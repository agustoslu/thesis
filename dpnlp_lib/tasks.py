# elaborate on privacy-utility trade-off, benchmark DP-SGD and DP-FTRL

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dpnlp.hospital import HospitalUnit
from dpnlp.patient import Patient


# tasks will call hospital and get their data

# class BaseTask(ABC):
#     @abstractmethod
#     def get_dataset(self, hospital: HospitalUnit):
#         ...

#     @abstractmethod
#     def compute_metrics(self, y_true, y_pred):
#         ...

#     @property
#     @abstractmethod
#     def get_label(self):
#         ...

#     def run_task(self):
#         ...
#         raise NotImplementedError("Subclasses should implement this method.")


# class MortalityTask(BaseTask):
        
#     def get_dataset(self, patient: Patient, split_method="iid", num_clients= 2, alpha= 0.5):
#         features = []
#         labels = []
#         for patient in patients: # retrieves patient.events.timeseries table as a combination of lab measurements and demographics
#             timeseries = patient.events.timeseries
#             features = timeseries.from_df_to_tensor() 
#             label = patient.icustays_df["MORTALITY_INHOSPITAL"]

#             features.append(features)
#             labels.append(label)

#         dataset = MIMIC3Dataset(features, labels)
#         train_set = self.HospitalUnit.get_train_test_split(dataset, split_ratio=0.8)
#         logger.info(f"Trainset created with {len(train_set)} samples.")
#         client_datasets = self.HospitalUnit.partition_dataset(train_set, split_method=split_method, num_clients=num_clients, alpha=alpha)
#         data_distrbution = self.HospitalUnit.get_distribution(client_datasets)
#         logger.info(f"Data distribution across clients: {data_distrbution}")
#         return client_datasets
    
#     def compute_metrics(self, y_true, y_pred):
#         from sklearn.metrics import roc_auc_score, accuracy_score
#         return {
#             "auroc": roc_auc_score(y_true, y_pred),
#             "accuracy": accuracy_score(y_true, y_pred)
#         }
    
#     @property
#     def get_label(self):
#         return "MORTALITY_INHOSPITAL"
    
#     #def run_task(self, hospital: HospitalUnit):



# class DecompensationTask(BaseTask):
#     @property
#     def label_column(self):
#         return "DECOMPENSATION"

#     def get_dataset(self, hospital: HospitalUnit):
#         
#     def compute_metrics(self, y_true, y_pred):
#         # AUROC, accuracy, etc.



# class LengthOfStayTask(BaseTask):
#     @property
#     def label_column(self):
#         return "LENGTH_OF_STAY"

#     def get_dataset(self, hospital: HospitalUnit):
#         # Extract features, LOS labels
#         

#     def compute_metrics(self, y_true, y_pred):
#         # RMSE, MAE, etc. 


# class PhenotypeTask(BaseTask):
#     @property
#     def label_column(self):
#         return "PHENOTYPE"

#     def get_dataset(self, hospital: HospitalUnit):
#         # Multi-label phenotype matrix
#         

#     def compute_metrics(self, y_true, y_pred):
#         # Macro/micro F1, etc.
  

# you could rather think of it as a main function there is nothing task specific about it, just pass args and get task from registry
# def run_task(task_name: str, hospital: HospitalUnit):
#     task_cls = TASK_REGISTRY.get(task_name)
#     if task_cls is None:
#         raise ValueError(f"Task {task_name} not registered.")
    
#     task = task_cls()
#     dataset = task.get_dataset(hospital)
#     train and get preds
#     metrics = task.compute_metrics(y_true, y_pred)
#     print(metrics)