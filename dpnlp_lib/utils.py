"""adapted from https://github.com/LuminLiu/BinaryFL/blob/master/datasets/get_data.py#L19"""

import numpy as np
from torch.utils.data import DataLoader, Dataset
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.metrics import mean_squared_error

def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

enable_info_logs()
logger = logging.getLogger(__name__)

class DatasetSplit(Dataset):
    """ Think of it as a wrapper for the original dataset,
    which allows us to create a subset of the original dataset according to the sample indices that are found
    in dict_users. So each client will be having its own features and labels."""

    def __init__(self, dataset, client_idxs):
        """
        :dataset: original complete dataset
        :idxs: list of indexes of the subset
        """
        self.dataset = dataset
        self.client_idxs = client_idxs

    def __len__(self):
        return len(self.client_idxs)

    def __getitem__(self, item):
        """
        :param item: index in the split dataset
        :return: sample, label and the index in the original complete dataset
        """
        feature, label = self.dataset[self.client_idxs[item]]
        return feature, label


def iid_esize_split(dataset, num_clients, is_shuffle=True):
    """each client to receive a random subset of the dataset, 
    so that the distribution of classes and features in each client’s data matches the overall dataset.
    Why is this IID?
    Random Sampling:
    Each client gets a random selection of samples from the entire dataset, without replacement.
    No Sorting or Grouping by Label:
    You do not sort or group by label or any feature before splitting, so the class/feature proportions in each client’s data will, on average, match the global distribution.
    Equal Size:
    Each client gets the same number of samples, further supporting identical distribution.
    What does "identically distributed" mean here?
    If your dataset is, for example, 60% class 0 and 40% class 1, then (statistically) each client’s data will also be about 60% class 0 and 40% class 1.
    There may be small fluctuations due to randomness, but with large enough datasets, the distributions will be very close."""
    num_samples = len(dataset)
    num_samples_per_client = int(num_samples / num_clients) 
    data_loaders = [0] * num_clients 
    dict_users, all_idxs = {}, list(range(num_samples)) 
    for i in range(num_clients):
        np.random.seed(42)
        dict_users[i] = np.random.choice(all_idxs, num_samples_per_client, replace=False)  
        all_idxs = list(set(all_idxs) - set(dict_users[i]))  
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                      batch_size=32, 
                                      shuffle=is_shuffle)
        
    return data_loaders

def niid_esize_split(dataset, num_clients, is_shuffle=True):
    """ 40k patients 
     two classes label=0 survived, label=1 died
     num_samples_per_shard = 40k / num_shards = 10k
     Client 1: Shard 0, 2, (labels mixed)
     Client 2: Shard 1, 3 (labels mixed)
    each client has a different set of 2 shards, each shard has 10k samples
    thereby each has a different class distribution
    Non-IID	Grouped/sorted before splitting	Skewed, does not match global"""
    num_samples = len(dataset)  
    num_shards = 2 * num_clients  
    data_loaders = [0] * num_clients  
    num_features = int(len(num_samples) / num_shards)  
    idx_shard = [i for i in range(num_shards)]  
    dict_users = {i: np.array([]) for i in range(num_clients)}  
    idxs = list(range(num_samples))  
    # is_shuffle is used to differentiate between train and test set
    labels = [dataset.features[idx] for idx in idxs]  
    idxs_labels = np.vstack((idxs, labels))  
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]  
    idxs = idxs_labels[0, :]  
    idxs = idxs.astype(int) 

    # divide and assign
    for i in range(num_clients):
        np.random.seed(42)
        # randomly choose two shards for each client
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)  
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_features: (rand + 1) * num_features]), axis=0)
            dict_users[i] = dict_users[i].astype(int) 
            
        logger.info(f"Client {i} assigned shards {rand_set}, total samples: {len(dict_users[i])}")
        data_loaders[i] = DataLoader(DatasetSplit(dataset, dict_users[i]),
                                      batch_size=32,  # only for now
                                      shuffle=is_shuffle)
    return data_loaders

def dirichlet_split_shortage_handle(dataclass_index, sampleOfClass, assigned_cls, sampleofID_c):
    """
    Use self-invoking function to handle the sample shortage problem in dirichlet split
    :param dataclass_index: list of list, len = 10, each element is the list of index of each of the 10 clses
    :param sampleOfClass: int, is the number of data samples to be assigned for the usr for the assigned_cls
    :param assigned_cls: the current cls being assigned
    :param sampleofID_c: the list of the sample of ID for current client
    :param random_seed: seed
    :return select_ID : the selected indexes for assigned cls
    :return dataclass_index: the list of the remaining available idxes
    """
    np.random.seed(42)
    if len(dataclass_index[assigned_cls]) >= sampleOfClass:
        select_ID = np.random.sample(dataclass_index[assigned_cls],
                                  sampleOfClass)
        dataclass_index[assigned_cls] = list(set(dataclass_index[assigned_cls]) - set(select_ID))
        sampleofID_c += select_ID
    else:
        shortage = sampleOfClass - len(dataclass_index[assigned_cls])
        select_ID = np.random.sample(dataclass_index[assigned_cls], len(dataclass_index[assigned_cls]))
        dataclass_index[assigned_cls] = list(set(dataclass_index[assigned_cls]) - set(select_ID))
        sampleofID_c += select_ID
        dataclass_num = [len(dataclass_index[cls]) for cls in range(2)] # binary classification 
        max_cls = np.argmax(dataclass_num)
        sampleofID_c, dataclass_index = dirichlet_split_shortage_handle(dataclass_index= dataclass_index,
                                                                        sampleOfClass=shortage,
                                                                        assigned_cls = max_cls,
                                                                        sampleofID_c=sampleofID_c,
                                                                        random_seed=42)
    return sampleofID_c, dataclass_index


def dirichlet_split(dataset, num_clients, is_shuffle, double_stochastic=True, alpha=0.5):
    num_samples = len(dataset) 
    data_loaders = [0] * num_clients
    datanumber = int(len(num_samples)/ num_clients)
    # print(f'initial data sample number {datanumber}')
    dataclass_index = [[] for i in range(2)]
    idxs = num_samples
    labels = [dataset.features[idx] for idx in num_samples]
    for idx in num_samples:
        dataclass_index[dataset.features[idx]].append(idx)

    np.random.seed(42)
    dirichlet_label = np.random.dirichlet([alpha]*2,
                                          num_clients)
    # dirichlet_label: size (num_class * num_clients) matrix, rsum = 1
    if double_stochastic:
        dirichlet_label = make_double_stochstic(dirichlet_label)

    sampleOfID = [[] for _ in range(num_clients)]
    for client in range(num_clients):
        # np.random.seed(args.seed)
        probs = dirichlet_label[client]
        sampleOfClass = [int(datanumber*prob) for prob in probs]
        for i in range(2):
            sampleOfID[client], dataclass_index = dirichlet_split_shortage_handle(dataclass_index= dataclass_index,
                                                                        sampleOfClass=sampleOfClass[i],
                                                                        assigned_cls = i,
                                                                        sampleofID_c=sampleOfID[client],
                                                                        random_seed=42)
        data_loaders[client] = DataLoader(DatasetSplit(dataset, sampleOfID[client]),
                                          batch_size=32,
                                          shuffle=is_shuffle)
    return data_loaders

def make_double_stochstic(x):
# rsum = 0, which is a inherent property of the dirichlet matrix
# here, its kinda like to rescale each row, so that the csum are equal for all columns
# But I do not quite understand how it works
# It is like a puzzle
    rsum = None
    csum = None

    n = 0
    while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
        x /= x.sum(0)
        x = x / x.sum(1)[:, np.newaxis]
        rsum = x.sum(1)
        csum = x.sum(0)
        n += 1

    #x = x / x.sum(axis=0).reshape(1,-1)
    return x

def partition_data(dataset, iid, is_shuffle = True):
    
    if is_shuffle: # only train set are split in a non-iid way
        if iid == 1:
            data_loaders = iid_esize_split(dataset,is_shuffle)
        elif iid == 0:
            data_loaders = niid_esize_split(dataset, is_shuffle)
        elif iid == -1:
            data_loaders = dirichlet_split(dataset,is_shuffle)
        else:
            raise ValueError('Data Distribution pattern `{}` not implemented '.format(iid))
    else:
        data_loaders = iid_esize_split(dataset, is_shuffle)
    return data_loaders



# TODO:
# incremental checkpointing?
# metrics
# plotting functionss

# def save_to_results_files(df: pd.DataFrame) -> None:
#     df.to_json(results_file(), orient="records", lines=True, mode="a")
#     logger.info("added line to %s", results_file())

# keep all patient info, debug logs / terminal is only for different stages of the pipeline 
# def save_to_log_file(df: pd.DataFrame) -> None:
#     df.to_json(results_file(), orient="records", lines=True, mode="a")
#     logger.info("added line to %s", results_file())

# def results_file() -> Path:
#     return home_dir / "results.json"

# def log_file() -> Path:
#     return home_dir / "log.txt"