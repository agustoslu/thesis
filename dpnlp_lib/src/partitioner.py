"""adapted from https://github.com/LuminLiu/BinaryFL/blob/master/datasets/get_data.py#L19"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Any, List, Dict, Tuple, Optional
import logging
from utils import enable_info_logs

enable_info_logs()
logger = logging.getLogger(__name__)


class DatasetSplit(Dataset):
    def __init__(self, dataset: Dataset, client_idxs: List[int]) -> None:
        self.dataset = dataset
        self.client_idxs = client_idxs

    def __len__(self) -> int:
        return len(self.client_idxs)

    def __getitem__(self, item: int) -> Tuple[Any, Any]:
        feature, label = self.dataset[self.client_idxs[item]]
        return feature, label


def iid_esize_split(
    dataset: Dataset, num_clients: int, is_shuffle: bool = True
) -> List[DataLoader]:
    num_samples = len(dataset)
    num_samples_per_client = int(num_samples / num_clients)
    data_loaders: List[DataLoader] = [None] * num_clients
    dict_users: Dict[int, np.ndarray] = {}
    all_idxs: List[int] = list(range(num_samples))
    for i in range(num_clients):
        np.random.seed(42)
        dict_users[i] = np.random.choice(
            all_idxs, num_samples_per_client, replace=False
        )
        all_idxs = list(set(all_idxs) - set(dict_users[i]))
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]), batch_size=4, shuffle=is_shuffle
        )
    return data_loaders


def niid_esize_split(
    dataset: Dataset, num_clients: int, is_shuffle: bool = True
) -> List[DataLoader]:
    num_samples = len(dataset)
    num_shards = 2 * num_clients
    data_loaders: List[DataLoader] = [None] * num_clients
    num_features = int(num_samples / num_shards)
    idx_shard: List[int] = [i for i in range(num_shards)]
    dict_users: Dict[int, np.ndarray] = {i: np.array([]) for i in range(num_clients)}
    idxs: List[int] = list(range(num_samples))
    labels: List[Any] = [dataset.features[idx] for idx in idxs]
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    idxs = idxs.astype(int)

    for i in range(num_clients):
        np.random.seed(42)
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand * num_features : (rand + 1) * num_features]),
                axis=0,
            )
            dict_users[i] = dict_users[i].astype(int)

        logger.info(
            f"Client {i} assigned shards {rand_set}, total samples: {len(dict_users[i])}"
        )
        data_loaders[i] = DataLoader(
            DatasetSplit(dataset, dict_users[i]),
            batch_size=32,
            shuffle=is_shuffle,
        )
    return data_loaders


def dirichlet_split_shortage_handle(
    dataclass_index: List[List[int]],
    sampleOfClass: int,
    assigned_cls: int,
    sampleofID_c: List[int],
    random_seed: int = 42,
) -> Tuple[List[int], List[List[int]]]:
    np.random.seed(random_seed)
    if len(dataclass_index[assigned_cls]) >= sampleOfClass:
        select_ID = np.random.sample(dataclass_index[assigned_cls], sampleOfClass)
        dataclass_index[assigned_cls] = list(
            set(dataclass_index[assigned_cls]) - set(select_ID)
        )
        sampleofID_c += list(select_ID)
    else:
        shortage = sampleOfClass - len(dataclass_index[assigned_cls])
        select_ID = np.random.sample(
            dataclass_index[assigned_cls], len(dataclass_index[assigned_cls])
        )
        dataclass_index[assigned_cls] = list(
            set(dataclass_index[assigned_cls]) - set(select_ID)
        )
        sampleofID_c += list(select_ID)
        dataclass_num = [
            len(dataclass_index[cls]) for cls in range(2)
        ]  # binary classification
        max_cls = int(np.argmax(dataclass_num))
        sampleofID_c, dataclass_index = dirichlet_split_shortage_handle(
            dataclass_index=dataclass_index,
            sampleOfClass=shortage,
            assigned_cls=max_cls,
            sampleofID_c=sampleofID_c,
            random_seed=random_seed,
        )
    return sampleofID_c, dataclass_index


def dirichlet_split(
    dataset: Dataset,
    num_clients: int,
    is_shuffle: bool,
    double_stochastic: bool = True,
    alpha: float = 0.5,
) -> List[DataLoader]:
    num_samples = len(dataset)
    data_loaders: List[DataLoader] = [None] * num_clients
    datanumber = int(num_samples / num_clients)
    dataclass_index: List[List[int]] = [[] for _ in range(2)]
    idxs: List[int] = list(range(num_samples))
    labels: List[Any] = [dataset.features[idx] for idx in idxs]
    for idx in idxs:
        dataclass_index[dataset.features[idx]].append(idx)

    np.random.seed(42)
    dirichlet_label = np.random.dirichlet([alpha] * 2, num_clients)
    if double_stochastic:
        dirichlet_label = make_double_stochastic(dirichlet_label)

    sampleOfID: List[List[int]] = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        probs = dirichlet_label[client]
        sampleOfClass = [int(datanumber * prob) for prob in probs]
        for i in range(2):
            sampleOfID[client], dataclass_index = dirichlet_split_shortage_handle(
                dataclass_index=dataclass_index,
                sampleOfClass=sampleOfClass[i],
                assigned_cls=i,
                sampleofID_c=sampleOfID[client],
                random_seed=42,
            )
        data_loaders[client] = DataLoader(
            DatasetSplit(dataset, sampleOfID[client]), batch_size=32, shuffle=is_shuffle
        )
    return data_loaders


def make_double_stochastic(x: np.ndarray) -> np.ndarray:
    rsum: Optional[np.ndarray] = None
    csum: Optional[np.ndarray] = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1
    return x
 

def partition_data(
    dataset: Dataset, iid: int, num_clients: int, is_shuffle: bool = True
) -> List[DataLoader]:
    if is_shuffle:
        if iid == 1:
            data_loaders = iid_esize_split(dataset, num_clients, is_shuffle)
        elif iid == 0:
            data_loaders = niid_esize_split(dataset, num_clients, is_shuffle)
        elif iid == -1:
            data_loaders = dirichlet_split(dataset, num_clients, is_shuffle)
        else:
            raise ValueError(f"Data Distribution pattern `{iid}` not implemented ")
    else:
        data_loaders = iid_esize_split(dataset, is_shuffle)
    return data_loaders
