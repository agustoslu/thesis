from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import torch
import opacus
from opacus import PrivacyEngine
import flwr
import logging
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from dpnlp_lib.src.utils import (
    MIMICTransformerEncoder,
    get_parameters,
    wrap_model_with_dp,
    save_metrics,
    append_row_to_csv,
    logger,
)
from dpnlp_lib.src.tasks import BaseTask
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DPSGDFedAvg(FedAvg):
    def __init__(
        self,
        fraction_fit: float = 0.5,  # only half of the hospitals selected for each training round
        fraction_evaluate: float = 1.0,  # only half of the hospitals selected for each evaluation round
        min_fit_clients: int = 2,  # minimum number of hospitals to train on
        min_evaluate_clients: int = 1,  # minimum number of hospitals to evaluate on
        min_available_clients: int = 2,  # minimum number of hospitals available for both training and evaluation
        task: BaseTask = None,
        fit_config: Optional[Dict[str, Union[str, float, int]]] = None,
        model_config: Optional[Dict[str, Union[str, float, int]]] = None,
        run_uuid: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.task = task
        self.metrics = getattr(task, "compute_metrics", None) if task else None
        self.fit_config = fit_config or {}
        self.model_config = model_config or {}
        self.run_uuid = run_uuid
        self.evaluate_metrics_aggregation_fn = (
            getattr(task, "compute_metrics", None) if task else None
        )
        self.max_epsilon = 0.0

    def __repr__(self) -> str:
        return "DPSGDFedAvg"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        model = MIMICTransformerEncoder(**self.model_config)
        model = wrap_model_with_dp(model)
        ndarrays = get_parameters(model)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = self.fit_config.copy()
        logger.info(
            f"[DPSGDFedAvg] Configuring fit for round {server_round} with config: {config}"
        )
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        fit_configurations = [
            (client, FitIns(parameters, config)) for client in clients
        ]
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None
        logger.info(
            f"[DPSGDFedAvg] Aggregating fit results for round {server_round} with {len(results)} results."
        )

        epsilons = []
        accepted_results = []
        for client, fit_res in results:
            logger.info(
                f"[DPSGDFedAvg] Client {client.cid} fit result: {fit_res.metrics}"
            )
            epsilon = fit_res.metrics.get("epsilon") if fit_res.metrics else None
            if epsilon is not None:
                epsilons.append(epsilon)
            if fit_res.metrics.get("accept", True):
                accepted_results.append((client, fit_res))

        if epsilons:
            self.max_epsilon = max(self.max_epsilon, max(epsilons))

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in accepted_results
        ]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        stats_aggregated = {
            "server_round": server_round,
            "num_clients": len(accepted_results),
            "max_epsilon": self.max_epsilon,
            "client_epsilons": epsilons,
            "accepted_clients": [client.cid for client, _ in accepted_results],
        }
        append_row_to_csv(
            stats_aggregated,
            file_name="federated_fit_stats.csv",
            run_uuid=self.run_uuid,
        )
        logger.info(f"Privacy budget ε at round {server_round}: {self.max_epsilon}")
        return parameters_aggregated, stats_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_list = [evaluate_res.metrics for _, evaluate_res in results]
        aggregated_metrics = {}
        for k in metrics_list[0].keys():
            values = [m[k] for m in metrics_list if k in m]
            if values:
                aggregated_metrics[k] = float(np.mean(values))

        metrics_aggregated = {
            "server_round": server_round,
            "num_clients": len(results),
            "eval_loss": loss_aggregated,
            **aggregated_metrics,
        }
        logger.info(
            f"[DPSGDFedAvg] Aggregated evaluation results for round {server_round}: {metrics_aggregated}"
        )
        save_metrics(
            metrics_aggregated,
            file_name=f"metrics_eval_round_{server_round}.json",
            run_uuid=self.run_uuid,
        )
        append_row_to_csv(
            metrics_aggregated,
            file_name="federated_eval_stats.csv",
            run_uuid=self.run_uuid,
        )
        return loss_aggregated, metrics_aggregated

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients


def train(
    model: MIMICTransformerEncoder,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 1,
    compute_metrics: Optional[
        Callable[[np.ndarray, np.ndarray], Dict[str, float]]
    ] = None,
) -> Tuple[MIMICTransformerEncoder, Dict[str, Scalar]]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_loss = running_loss / len(train_loader)

    metrics = {}
    if compute_metrics:
        metrics = compute_metrics(all_labels, all_preds)

    metrics.update(
        {
            "train_loss": final_loss,
        }
    )
    logger.info(f"Final Train Metrics: {metrics}")
    return model, metrics


def test(
    model: MIMICTransformerEncoder,
    testloader: torch.utils.data.DataLoader,
    config: Dict[str, Union[str, float, int]],
    device: torch.device,
    compute_metrics: Optional[
        Callable[[np.ndarray, np.ndarray], Dict[str, float]]
    ] = None,
) -> Tuple[float, int, Dict[str, Scalar]]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    running_loss, total_samples = 0.0, 0
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for features, labels in tqdm(testloader, desc="Evaluation..."):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            total_samples += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_loss = running_loss / len(testloader)

    metrics = {}
    if compute_metrics:
        metrics = compute_metrics(all_labels, all_preds)
        metrics.update(
            {
                "train_loss": final_loss,
            }
        )
    logger.info(f"Final Evaluation Metrics: {metrics}")

    return final_loss, total_samples, metrics
