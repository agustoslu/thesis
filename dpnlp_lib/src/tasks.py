from abc import ABC, abstractmethod
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
)


class BaseTask(ABC):
    @abstractmethod
    def get_label(self): ...

    @abstractmethod
    def compute_metrics(self, y_true, y_pred): ...

    @property
    def get_info(self): ...


class MortalityTask(BaseTask):
    def get_label(self, patient):
        assert len(patient.icustays_df) == 1
        label = patient.icustays_df["MORTALITY_INHOSPITAL"].astype(int)
        return label

    def compute_metrics(self, y_true, y_pred):
        return {
            "roc-auc": roc_auc_score(y_true, y_pred),
            "pr-auc": average_precision_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    @property
    def get_info(self):
        return {"label_column": "MORTALITY_INHOSPITAL", "task_type": "binary"}


class PhenotypeTask(BaseTask):
    def get_label(self, patient):
        PHENOTYPE_LIST = sorted(
            [
                "Acute and unspecified renal failure",
                "Acute cerebrovascular disease",
                "Acute myocardial infarction",
                "Cardiac dysrhythmias",
                "Chronic kidney disease",
                "Chronic obstructive pulmonary disease",
                "Complications of surgical/medical care",
                "Conduction disorders",
                "Congestive heart failure; nonhypertensive",
                "Coronary atherosclerosis and related",
                "Diabetes mellitus with complications",
                "Diabetes mellitus without complication",
                "Disorders of lipid metabolism",
                "Essential hypertension",
                "Fluid and electrolyte disorders",
                "Gastrointestinal hemorrhage",
                "Hypertension with complications",
                "Other liver diseases",
                "Other lower respiratory disease",
                "Other upper respiratory disease",
                "Pleurisy; pneumothorax; pulmonary collapse",
                "Pneumonia",
                "Respiratory failure; insufficiency; arrest",
                "Septicemia (except in labor)",
                "Shock",
            ]
        )
        phenos = patient.events.phenotypes.copy()
        assert len(phenos) == 1

        for col in self.PHENOTYPE_LIST:
            if col not in phenos.columns:
                phenos[col] = 0

        phenos = phenos[self.PHENOTYPE_LIST]
        label = phenos.iloc[0].astype(int).values
        return label

    def compute_metrics(self, y_true, y_pred):
        return {
            "macro_roc_auc": roc_auc_score(y_true, y_pred, average="macro"),
            "micro_roc_auc": roc_auc_score(y_true, y_pred, average="micro"),
            "weighted_roc_auc": roc_auc_score(y_true, y_pred, average="weighted"),
            "macro_pr_auc": average_precision_score(y_true, y_pred, average="macro"),
            "micro_pr_auc": average_precision_score(y_true, y_pred, average="micro"),
            "weighted_pr_auc": average_precision_score(
                y_true, y_pred, average="weighted"
            ),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    @property
    def get_info(self):
        return {"label_column": "PHENOTYPE", "task_type": "multilabel - 25 classes"}
