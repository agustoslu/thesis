from typing import Dict, List, Optional, Tuple, Union
import torch
import opacus
from opacus import PrivacyEngine
import flwr
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from dpnlp_lib.src.utils import (
    MIMICGRU,
    get_parameters,
    set_parameters,
    get_target_delta,
    save_metrics_to_run_folder,
    log_gpu_usage,
    append_row_to_csv,
    logger,
)
from dpnlp_lib.src.tasks import BaseTask


class DPSGDFedAvg(Strategy):
    def __init__(
        self,
        fraction_fit: float = 0.5,  # only half of the hospitals selected for each training round
        fraction_evaluate: float = 0.5,  # only half of the hospitals selected for each evaluation round
        min_fit_clients: int = 2,  # minimum number of hospitals to train on
        min_evaluate_clients: int = 2,  # minimum number of hospitals to evaluate on
        min_available_clients: int = 2,  # minimum number of hospitals available for both training and evaluation
        task: BaseTask = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.task = task
        self.metrics = getattr(task, "compute_metrics", None) if task else None
        self.max_epsilon = 0.0

    def __repr__(self) -> str:
        return "DPSGDFedAvg"

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        model = MIMICGRU(input_dim=300, hidden_dim=128, num_layers=6, dropout=0.3)
        ndarrays = get_parameters(model)
        return ndarrays_to_parameters(ndarrays)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        config = {
            "metrics": self.metrics,
            "task_type": getattr(self.task, "task_type", None) if self.task else None,
            "round": server_round,
        }

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
            return None, {}

        epsilons = []
        accepted_results = []
        for client, fit_res in results:
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
        metrics_aggregated = {
            "server_round": server_round,
            "num_clients": len(accepted_results),
            "max_epsilon": self.max_epsilon,
            "client_epsilons": epsilons,
        }
        save_metrics_to_run_folder(
            metrics_aggregated, filename=f"metrics_round_{server_round}.json"
        )
        log_gpu_usage(context=f"Server Aggregate Round {server_round}")
        append_row_to_csv(metrics_aggregated, filename="federated_stats.csv")
        logger.info(f"Privacy budget ε at round {server_round}: {self.max_epsilon}")
        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        if self.fraction_evaluate == 0.0:
            return []
        config = {
            "metrics": self.metrics,
            "task_type": self.task.get_info.get("task_type") if self.task else None,
            "round": server_round,
        }
        evaluate_ins = EvaluateIns(parameters, config)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {
            "server_round": server_round,
            "num_clients": len(results),
            "eval_loss": loss_aggregated,
        }
        save_metrics_to_run_folder(
            metrics_aggregated, filename=f"eval_metrics_round_{server_round}.json"
        )
        log_gpu_usage(context=f"Server Aggregate Evaluate Round {server_round}")
        append_row_to_csv(metrics_aggregated, filename="eval_federated_stats.csv")
        return loss_aggregated, metrics_aggregated

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
