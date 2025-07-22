import os
import pickle
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, Dict, Tuple, Optional, Callable
import numpy as np
from flwr.client import NumPyClient
from flwr.common import Context
from omegaconf import DictConfig, OmegaConf
import opacus
from opacus import PrivacyEngine
import logging
from dpnlp_lib.src.utils import (
    set_parameters,
    get_parameters,
    get_target_delta,
    wrap_model_with_dp,
    MIMICTransformerEncoder,
    logger,
    load_accountant_state,
    save_accountant_state,
)
from dpnlp_lib.src.dp_sgd import train, test
# from dpnlp_lib.src.dp_ftrl import train, test

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        cid: str,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        compute_metrics: Optional[Callable[[np.ndarray, np.ndarray], Dict[str, float]]],
        client_fit_config: Optional[Dict[str, Any]] = None,
        run_uuid: Optional[str] = None,
    ):
        self.cid = cid
        self.device = device
        self.model = wrap_model_with_dp(model).to(self.device)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.compute_metrics = compute_metrics
        self.run_uuid = run_uuid

        cfg = client_fit_config or {}
        self.lr = cfg.get("lr", 0.01)
        self.epochs = cfg.get("epochs", 1)
        self.max_grad_norm = cfg.get("max_grad_norm", 1.0)
        self.target_epsilon = cfg.get("target_epsilon", 2.5)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.delta = get_target_delta(len(self.train_dataloader.dataset))

        self.privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_dataloader = (
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_dataloader,
                target_epsilon=self.target_epsilon,
                target_delta=self.delta,
                epochs=self.epochs,
                max_grad_norm=self.max_grad_norm,
                grad_accumulation_steps=2,
            )
        )
        load_accountant_state(self.privacy_engine, self.cid, self.run_uuid, logger)

    def fit(
        self, params: list[np.array], config: Dict[str, Any]
    ) -> Tuple[list[np.array], int, Dict[str, Any]]:
        logger.info(f"[Client {self.cid}] received fit instruction.")
        set_parameters(self.model, params)
        self.model, metrics = train(
            model=self.model,
            train_loader=self.train_dataloader,
            optimizer=self.optimizer,
            device=self.device,
            epochs=self.epochs,
            compute_metrics=self.compute_metrics,
        )

        save_accountant_state(self.privacy_engine, self.cid, self.run_uuid, logger)
        epsilon = self.privacy_engine.get_epsilon(self.delta)
        metrics["epsilon"] = epsilon
        logger.info(f"[Client {self.cid}] Training completed. Epsilon: {epsilon}")
        return get_parameters(self.model), len(self.train_dataloader.dataset), metrics

    def evaluate(
        self, params: list[np.array], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        logger.info(f"[Client {self.cid}] received evaluate instruction.")
        set_parameters(self.model, params)
        loss, num_samples, metrics = test(
            model=self.model,
            testloader=self.test_dataloader,
            config=config,
            device=self.device,
            compute_metrics=self.compute_metrics,
        )
        return float(loss), num_samples, metrics


def get_client_fn(
    cfg: DictConfig, task_obj, train_partitions, test_partitions, run_uuid
) -> Callable[[Context], NumPyClient]:
    """Return a function that creates a new Flower client for a given client ID."""

    def client_fn(context: Context) -> NumPyClient:
        """Create a Flower client instance for a simulation."""
        partition_id = context.node_config.get("partition-id", 0)
        print(f"context.node_config: {context.node_config}")
        print(f"Creating client function for partition {partition_id}...")
        if partition_id >= len(train_partitions) or partition_id >= len(
            test_partitions
        ):
            raise ValueError(
                f"Partition ID {partition_id} exceeds available partitions."
            )
        logger.info(f"Instantiating client {partition_id}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = MIMICTransformerEncoder(**cfg.task.model_config)

        train_dataloader = train_partitions[partition_id]
        test_dataloader = test_partitions[partition_id]
        logger.info(
            f"Client {partition_id} created with model {model} on device {device}."
        )
        logger.info(
            f"Train and test dataloaders created for client {train_dataloader} and {test_dataloader}."
        )

        logger.info(
            f"Client {partition_id} created with {len(train_dataloader.dataset)} training samples and {len(test_dataloader.dataset)} test samples."
        )
        logger.info(
            f"(train_dataloader: {train_dataloader}, test_dataloader: {test_dataloader})"
        )
        compute_metrics_fn = getattr(task_obj, "compute_metrics", None)

        return FlowerClient(
            cid=str(partition_id),
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            compute_metrics=compute_metrics_fn,
            client_fit_config=OmegaConf.to_container(
                cfg.dp_sgd.fit_config, resolve=True
            )
            if "dp_sgd" in cfg and "fit_config" in cfg.dp_sgd
            else {},
            run_uuid=run_uuid,
        ).to_client()

    return client_fn
