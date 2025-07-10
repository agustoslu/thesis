from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING

import torch
from bert_score import score as b_score
from rouge_score import rouge_scorer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    roc_auc_score,
)
if TYPE_CHECKING:
    from dpnlp_lib.src.builder import Patient


class BaseTask(ABC):
    @abstractmethod
    def get_source_text(self, patient: Patient) -> Any:
        # violation of https://en.wikipedia.org/wiki/Interface_segregation_principle but still for practicality
        ...

    @abstractmethod
    def get_label(self, patient: Patient) -> Any: ...

    @abstractmethod
    def compute_metrics(self, y_true: Any, y_pred: Any) -> Dict[str, float]: ...

    @property
    @abstractmethod
    def get_info(self) -> Dict[str, Any]: ...


class MortalityTask(BaseTask):
    def get_source_text(self, patient: Patient) -> None:
        """Not applicable for this task."""
        return None

    def get_label(self, patient: Patient) -> Any:
        if hasattr(patient, "icustays_df") and len(patient.icustays_df) == 1:
            label = int(patient.icustays_df.iloc[0]["MORTALITY_INHOSPITAL"])
            return label

        timeseries = getattr(patient.events, "timeseries", None)
        if timeseries is not None and "HADM_ID" in timeseries.columns:
            hadm_id = timeseries["HADM_ID"].unique()[0]
            icustays_df = patient.icustays_df[patient.icustays_df["HADM_ID"] == hadm_id]
            label = int(icustays_df["MORTALITY_INHOSPITAL"].astype(int).values[0])
            return label

    def compute_metrics(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
        return {
            "roc-auc": roc_auc_score(y_true, y_pred),
            "pr-auc": average_precision_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
        }

    @property
    def get_info(self) -> Dict[str, str]:
        return {"label_column": "MORTALITY_INHOSPITAL", "task_type": "binary"}


class PhenotypeTask(BaseTask):
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

    def get_source_text(self, patient: Patient) -> None:
        """Not applicable for this task."""
        return None

    def get_label(self, patient: Patient) -> List[int]:
        phenos = patient.get_table("phenotypes").copy()
        assert len(phenos) == 1

        for col in self.PHENOTYPE_LIST:
            if col not in phenos.columns:
                phenos[col] = 0

        phenos = phenos[self.PHENOTYPE_LIST]
        label = phenos.iloc[0].astype(int).values.tolist()
        return label

    def compute_metrics(self, y_true: Any, y_pred: Any) -> Dict[str, float]:
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
    def get_info(self) -> Dict[str, str]:
        return {"label_column": "PHENOTYPE", "task_type": "multiclass"}


class SummaryTask(BaseTask):
    def get_source_text(self, patient: Patient) -> str:
        if hasattr(patient, "icustays_df") and len(patient.icustays_df) == 1:
            source_text = patient.icustays_df.iloc[0]["SOURCE_TEXT"]
            return source_text if source_text is not None else ""

    def get_label(self, patient: Patient) -> str:
        if hasattr(patient, "icustays_df") and len(patient.icustays_df) == 1:
            label = patient.icustays_df.iloc[0]["DISCHARGE_SUMMARY"]
            return label if label is not None else ""

    def compute_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
        rouge_types = ["rouge1", "rouge2", "rougeLsum"]
        scorer = rouge_scorer.RougeScorer(
            rouge_types, use_stemmer=True, split_summaries=True
        )

        rouge_scores: Dict[str, List[float]] = {
            rouge_type: [] for rouge_type in rouge_types
        }
        for ref, pred in zip(y_true, y_pred):
            score_dict = scorer.score(ref, pred)
            for rouge_type in rouge_types:
                rouge_scores[rouge_type].append(score_dict[rouge_type].fmeasure)

        avg_rouge = {
            f"{rouge_type}_fmeasure": sum(scores) / len(scores)
            for rouge_type, scores in rouge_scores.items()
        }

        P, R, F1 = b_score(
            cands=y_pred,
            refs=y_true,
            model_type="dmis-lab/biobert-v1.1",
            lang="en",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        bert_scores = {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }

        metrics = {**avg_rouge, **bert_scores}
        return metrics

    @property
    def get_info(self) -> Dict[str, Any]:
        return {"label_column": None, "task_type": "summary"}
