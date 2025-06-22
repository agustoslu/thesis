from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, ClassVar, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from pathlib import Path
import logging
from argparse import ArgumentParser
from functools import cached_property
import re
import yaml
from concurrent.futures import ProcessPoolExecutor
from download import DATASET_PATH_DEMO, DATASET_PATH, home_dir
from tasks import BaseTask, MortalityTask, PhenotypeTask
from partitioner import enable_info_logs, partition_data

enable_info_logs()
logger = logging.getLogger(__name__)

##################################################
# Building Patient Database and Creating Hospitals
##################################################


class PandasMixin:
    """Mixin class that adds pd DataFrame conversion capabilities.
    Assumes the child class has a list of dataclass-like object (e.g. ICUStays)."""

    def _to_clean_dict(self, obj) -> dict:
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}

    @staticmethod
    def standardize_columns(df: pd.DataFrame, case: str = "upper") -> pd.DataFrame:
        df.columns = df.columns.astype(str)
        if case == "upper":
            df.columns = (
                df.columns.str.strip()
                .str.upper()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )
        elif case == "lower":
            df.columns = (
                df.columns.str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("-", "_")
            )
        return df

    @staticmethod
    def to_long(
        df: pd.DataFrame,
        index_name: str = None,
        variable_name: str = "VARIABLE",
        value_name: str = "VALUE",
    ) -> pd.DataFrame:
        """
        e.g. long_phenotypes = patient.events.to_long(patient.events.phenotypes, index_name="ICUSTAY_ID")
        """
        if index_name is None:
            index_name = df.index.name
        df_reset = df.reset_index()
        return pd.melt(
            df_reset,
            id_vars=[index_name],
            var_name=variable_name,
            value_name=value_name,
        )

    @staticmethod
    def to_wide(
        df: pd.DataFrame,
        index_name: str = None,
        variable_name: str = "VARIABLE",
        value_name: str = "VALUE",
    ) -> pd.DataFrame:
        """
        e.g. wide_phenotypes = PandasMixin.to_wide(long_phenotypes, index_name="ICUSTAY_ID")
        """
        if index_name is None:
            index_name = df.index.name

        wide_df = df.pivot(index=index_name, columns=variable_name, values=value_name)
        wide_df = wide_df.reset_index()
        return wide_df

    def to_dataframe(self, records: List[object]) -> pd.DataFrame:
        df = pd.DataFrame([self._to_clean_dict(obj) for obj in records])
        return self.standardize_columns(df, case="upper")

    def csv_to_df(
        self, path: Path, header=0, index_col=None, case: str = "upper"
    ) -> pd.DataFrame:
        df = pd.read_csv(path, header=header, index_col=index_col)
        return self.standardize_columns(df, case=case)

    def csv_to_df_batch(
        self, path: Path, header=0, index_col=None, case: str = "upper"
    ) -> pd.DataFrame:
        chunks = pd.read_csv(path, header=header, index_col=index_col, chunksize=10**6, compression="gzip", low_memory=True)
        df = pd.concat(chunks, ignore_index=True, sort=False)
        return self.standardize_columns(df, case=case)

@dataclass
class DataManager(PandasMixin):
    """Class to manage data loading and merging for MIMIC-III dataset and meant as a bridge between other classes
    to prevent overloading them and also to cache mapping and making them read-only"""

    data_path: Path
    diagnoses_map_path: Path
    diagnoses_path: Path
    phenotype_definitions_path: Path
    hcup_ccs_2015_path: Path
    var_map_path: Path
    var_ranges_path: Path
    variable_column: str = "LEVEL2"

    # lazy loading
    _phenotype_definitions: Optional[Dict] = field(default=None, init=False)
    _diagnoses: Optional[pd.DataFrame] = field(default=None, init=False)
    _var_map: Optional[pd.DataFrame] = field(default=None, init=False)
    _var_ranges: Optional[pd.DataFrame] = field(default=None, init=False)

    def merge_on_subject(
        self, table1: pd.DataFrame, table2: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge two tables on SUBJECT_ID."""
        logger.warning(
            "Merging on SUBJECT_ID only. This may lead to incorrect results if the tables are not aligned by HADM_ID."
        )
        logger.warning("Table passed to merge_on_subject_admission: %s", table1.columns)
        logger.warning("Table passed to merge_on_subject_admission: %s", table2.columns)
        return table1.merge(
            table2, how="inner", left_on=["SUBJECT_ID"], right_on=["SUBJECT_ID"]
        )

    def merge_on_subject_admission(
        self, table1: pd.DataFrame, table2: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge two tables on SUBJECT_ID and HADM_ID."""
        logger.warning(
            "Merging on SUBJECT_ID and HADM_ID. Ensure both tables have these columns."
        )
        logger.warning("Table passed to merge_on_subject_admission: %s", table1.columns)
        logger.warning("Table passed to merge_on_subject_admission: %s", table2.columns)
        return table1.merge(
            table2,
            how="inner",
            left_on=["SUBJECT_ID", "HADM_ID"],
            right_on=["SUBJECT_ID", "HADM_ID"],
        )

    def load_all_tables(self, data_path: Path) -> Dict[str, pd.DataFrame]:
        """Load all .csv files from MIMIC directory."""
        dataframes = {}
        for file in data_path.glob("*.csv"): # demo
        #for file in data_path.glob("*.csv.gz"):
            logger.info(f"Loading {file.name}...")
            #df = self.csv_to_df_batch(file)
            df= self.csv_to_df(file)  # demo 
            dataframes[file.stem.lower()] = df
        
        # perform some early filtering and add them as additional tables
        if "icustays" in dataframes and "admissions" in dataframes:
            dataframes["stays_admits"] = self.merge_on_subject_admission(
                dataframes["icustays"], dataframes["admissions"]
            )
        if "stays_admits" in dataframes and "patients" in dataframes:
            dataframes["stays_admits_patients"] = self.merge_on_subject(
                dataframes["stays_admits"], dataframes["patients"]
            )

        if (
            "labevents" in dataframes
            and "ICUSTAY_ID" not in dataframes["labevents"].columns
        ):
            dataframes["labevents"]["ICUSTAY_ID"] = ""

        return dataframes

    @cached_property
    def diagnoses(self) -> pd.DataFrame:
        """Load and cache diagnoses data."""
        if self._diagnoses is None:
            codes = self.csv_to_df(self.diagnoses_map_path)  # D_ICD_DIAGNOSES.csv
            codes = self.standardize_columns(codes, case="upper")
            codes = codes[["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"]]
            diagnoses = self.csv_to_df(self.diagnoses_path)  # DIAGNOSES_ICD.csv
            diagnoses = self.standardize_columns(diagnoses, case="upper")
            diagnoses = diagnoses.merge(
                codes, how="inner", left_on="ICD9_CODE", right_on="ICD9_CODE"
            )
            diagnoses[["SUBJECT_ID", "HADM_ID", "SEQ_NUM"]] = diagnoses[
                ["SUBJECT_ID", "HADM_ID", "SEQ_NUM"]
            ].astype(int)
            self._diagnoses = diagnoses
        return self._diagnoses

    @cached_property
    def phenotype_definitions(self) -> Dict:
        if self._phenotype_definitions is None:
            with open(self.phenotype_definitions_path, "r") as file:
                self._phenotype_definitions = yaml.load(file, Loader=yaml.SafeLoader)
        return self._phenotype_definitions

    def add_hcup_ccs_2015_groups(self, diagnoses: pd.DataFrame) -> pd.DataFrame:
        def_map = {}
        for dx in self._phenotype_definitions:
            for code in self._phenotype_definitions[dx]["codes"]:
                def_map[code] = (
                    dx,
                    self._phenotype_definitions[dx]["use_in_benchmark"],
                )
        diagnoses["HCUP_CCS_2015"] = diagnoses["ICD9_CODE"].apply(
            lambda c: def_map[c][0] if c in def_map else None
        )
        diagnoses["USE_IN_BENCHMARK"] = diagnoses["ICD9_CODE"].apply(
            lambda c: int(def_map[c][1]) if c in def_map else None
        )
        return diagnoses

    def make_phenotype_label_matrix(self, phenotypes, stays=None) -> pd.DataFrame:
        phenotypes = (
            phenotypes[["ICUSTAY_ID", "HCUP_CCS_2015"]]
            .loc[phenotypes["USE_IN_BENCHMARK"] > 0]
            .drop_duplicates()
        )
        phenotypes["VALUE"] = 1
        phenotypes = phenotypes.pivot(
            index="ICUSTAY_ID", columns="HCUP_CCS_2015", values="VALUE"
        )
        if stays is not None:
            phenotypes = phenotypes.reindex(
                stays["ICUSTAY_ID"].sort_values(), fill_value=0
            )
        return phenotypes.astype(int).sort_index(axis=0).sort_index(axis=1)

    @cached_property
    def variable_map(self) -> pd.DataFrame:
        """Load and cache variable mapping."""
        var_map = (
            self.csv_to_df(self.var_map_path, index_col=None).fillna("").astype(str)
        )
        var_map = self.standardize_columns(var_map, case="upper")
        var_map["COUNT"] = var_map["COUNT"].astype(int)
        var_map = var_map.loc[
            (var_map[self.variable_column] != "") & (var_map["COUNT"] > 0)
        ]
        var_map = var_map.loc[(var_map["STATUS"] == "ready")]
        var_map["ITEMID"] = var_map["ITEMID"].astype(int)
        var_map = var_map[[self.variable_column, "ITEMID", "MIMIC_LABEL"]].set_index(
            "ITEMID"
        )
        self._var_map = var_map.rename(
            columns={self.variable_column: "VARIABLE", "MIMIC_LABEL": "MIMIC_LABEL"}
        )
        return self._var_map

    @cached_property
    def variable_ranges(self) -> pd.DataFrame:
        columns = [
            self.variable_column,
            "OUTLIER LOW",
            "VALID LOW",
            "IMPUTE",
            "VALID HIGH",
            "OUTLIER HIGH",
        ]
        to_rename = dict(zip(columns, [c.replace(" ", "_") for c in columns]))
        to_rename[self.variable_column] = "VARIABLE"
        var_ranges = self.csv_to_df(self.var_ranges_path, index_col=None)
        columns_std = [c.replace(" ", "_") for c in columns]
        var_ranges = var_ranges[columns_std]
        var_ranges.rename(
            columns={c.replace(" ", "_"): n for c, n in to_rename.items()}, inplace=True
        )
        var_ranges = var_ranges.drop_duplicates(subset="VARIABLE", keep="first")
        var_ranges.set_index("VARIABLE", inplace=True)
        self._var_ranges = var_ranges.loc[var_ranges.notnull().all(axis=1)]
        return self._var_ranges

    def filter_diagnoses_on_stays(
        self, diagnoses: pd.DataFrame, stays: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter diagnoses to only include those that have corresponding ICU stays."""
        return diagnoses.merge(
            stays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].drop_duplicates(),
            how="inner",
            left_on=["SUBJECT_ID", "HADM_ID"],
            right_on=["SUBJECT_ID", "HADM_ID"],
        )


@dataclass
class ICUStay:
    ICUSTAY_ID: int
    SUBJECT_ID: int
    HADM_ID: int  # hospital admission id
    FIRST_CAREUNIT: str
    LAST_CAREUNIT: str
    FIRST_WARDID: int
    LAST_WARDID: int
    INTIME: datetime
    OUTTIME: datetime
    LOS: float  # length of stay
    DOB: Optional[datetime] = None  # date of birth
    DOD: Optional[datetime] = None  # date of death
    DISCHTIME: Optional[datetime] = None  # discharge time
    ADMITTIME: Optional[datetime] = None  # admission time
    DEATHTIME: Optional[datetime] = None
    GENDER: Optional[str] = None
    ETHNICITY: Optional[str] = None
    AGE: Optional[int] = None
    MORTALITY_INUNIT: Optional[int] = None
    MORTALITY_INHOSPITAL: Optional[int] = None
    ICU_TYPE: Optional[int] = None

    def is_transfer_free(self) -> bool:
        return (
            self.FIRST_CAREUNIT == self.LAST_CAREUNIT
            and self.FIRST_WARDID == self.LAST_WARDID
        )

    def age_at_admission(self) -> Optional[float]:
        dob = self.DOB
        intime = self.INTIME

        # sanity check to avoid absurd dates due to anonymization
        if not (1900 <= dob.year <= 2200):
            return None

        age = (intime - dob).total_seconds() / (60 * 60 * 24 * 365)
        return round(90 if age < 0 else age)

    def is_adult(self, min_age: int = 18) -> bool:
        """Check if the patient is an adult based on the age at admission."""
        age = self.age_at_admission()
        return age is not None and age >= min_age

    def encode_gender(self) -> int:
        """Encode gender as an integer."""
        g_map = {"F": 1, "M": 2, "OTHER": 3, "": 0}
        return g_map.get((self.GENDER or "").strip().upper(), g_map["OTHER"])

    def encode_ethnicity(self) -> int:
        e_map = {
            "ASIAN": 1,
            "BLACK": 2,
            "HISPANIC": 3,
            "WHITE": 4,
            "OTHER": 5,
            "UNABLE TO OBTAIN": 0,
            "PATIENT DECLINED TO ANSWER": 0,
            "UNKNOWN": 0,
            "": 0,
        }

        def aggregate(ETHNICITY: str) -> str:
            return (
                ETHNICITY.replace(" OR ", "/")
                .split(" - ")[0]
                .split("/")[0]
                .strip()
                .upper()
            )

        return e_map.get(aggregate(self.ETHNICITY or ""), e_map["OTHER"])

    def encode_inunit_mortality(self) -> int:
        mortality: bool = False
        mortality = pd.notnull(self.DOD) and (
            (self.INTIME <= self.DOD) and (self.OUTTIME >= self.DOD)
        )
        mortality = mortality or (
            pd.notnull(self.DEATHTIME)
            and ((self.INTIME <= self.DEATHTIME) and (self.OUTTIME >= self.DEATHTIME))
        )
        return int(mortality)

    def encode_inhospital_mortality(self) -> int:
        mortality: bool = False
        mortality = pd.notnull(self.DOD) and (
            (self.ADMITTIME <= self.DOD) and (self.DISCHTIME >= self.DOD)
        )
        mortality = mortality or (
            pd.notnull(self.DEATHTIME)
            and (
                (self.ADMITTIME <= self.DEATHTIME)
                and (self.DISCHTIME >= self.DEATHTIME)
            )
        )
        return int(mortality)

    def encode_icutype(self) -> int:
        icu_type_map = {
            "MICU": 1,
            "SICU": 2,
            "CSRU": 3,
            "TSICU": 4,
            "CCU": 5,
            "NICU": 6,
            "PICU": 7,
            "OTHER": 8,
        }
        return icu_type_map.get(
            self.FIRST_CAREUNIT.strip().upper(), icu_type_map["OTHER"]
        )

    @classmethod
    def filter_admissions_on_nb_icustays(
        cls, icustays_df: pd.DataFrame, min_nb_stays=1, max_nb_stays=1
    ) -> List["ICUStay"]:
        if icustays_df.empty:
            return []

        to_keep = icustays_df.groupby("HADM_ID").count()[["ICUSTAY_ID"]].reset_index()

        to_keep = to_keep.loc[
            (to_keep.ICUSTAY_ID >= min_nb_stays) & (to_keep.ICUSTAY_ID <= max_nb_stays)
        ][["HADM_ID"]] 
        

        filtered_stays_df = icustays_df.merge(to_keep, how="inner", on="HADM_ID")
        valid_hadm_ids = set(filtered_stays_df.HADM_ID)

        return [
            ICUStay(**row.to_dict())
            for _, row in filtered_stays_df.iterrows()
            if row["HADM_ID"] in valid_hadm_ids
        ]


@dataclass
class Events(PandasMixin):
    data_manager: DataManager
    patient: Patient

    diagnoses: Optional[pd.DataFrame] = field(default=None, init=False)
    phenotypes: Optional[pd.DataFrame] = field(default=None, init=False)
    events: Optional[pd.DataFrame] = field(default=None, init=False)
    _cleaned_events: Optional[pd.DataFrame] = field(
        default=None, init=False
    )  # cleaned events converted to time-series
    _timeseries: Optional[pd.DataFrame] = field(
        default=None, init=False
    )  # time-series data indexed and sorted by CHARTTIME, there might be some duplicates due to some measurements being done at the same time

    def __post_init__(self):
        self.get_events()
        self.get_diagnoses()
        self.get_phenotypes()

    ################################################
    # Data loading functions
    ################################################

    @cached_property
    def get_icustays_df(self) -> pd.DataFrame:
        return self.to_dataframe(self.patient.icu_stays)

    def get_events(self) -> pd.DataFrame:
        event_header = [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "CHARTTIME",
            "ITEMID",
            "VALUE",
            "VALUEUOM",
        ]
        event_tables = []
        for table_name in ["chartevents", "labevents", "outputevents"]:
            df = self.patient.get_table(table_name)
            if df is not None and not df.empty:
                for col in event_header:
                    if col not in df.columns:
                        df[col] = ""
                df = df[event_header].copy()
                df["SOURCE"] = table_name
                event_tables.append(df)
        if event_tables:
            events = pd.concat(event_tables, ignore_index=True, sort=False)
        else:
            events = pd.DataFrame(columns=event_header + ["SOURCE"])
        self.events = events
        return self.events

    def get_diagnoses(self) -> pd.DataFrame:
        """Get diagnoses for the patient using HCUP CCS 2015 groups."""
        _ = self.data_manager.phenotype_definitions
        diagnoses = self.data_manager.diagnoses
        self.diagnoses = self.data_manager.add_hcup_ccs_2015_groups(diagnoses)
        return self.diagnoses

    def get_phenotypes(self) -> pd.DataFrame:
        """Get phenotypes for the patient."""
        if self.phenotypes is None:
            diagnoses = self.get_diagnoses()
            icustays_df = self.get_icustays_df
            diagnoses = self.data_manager.filter_diagnoses_on_stays(
                diagnoses, icustays_df
            )
            self.phenotypes = self.data_manager.make_phenotype_label_matrix(
                diagnoses, stays=icustays_df
            )
        return self.phenotypes

    clean_fns: ClassVar[Dict[str, Callable]] = {
        "Capillary refill rate": lambda df: Events.clean_crr(df),
        "Diastolic blood pressure": lambda df: Events.clean_dbp(df),
        "Systolic blood pressure": lambda df: Events.clean_sbp(df),
        "Fraction inspired oxygen": lambda df: Events.clean_fio2(df),
        "Oxygen saturation": lambda df: Events.clean_o2sat(df),
        "Glucose": lambda df: Events.clean_lab(
            df
        ),  # rows with mimic label Fingerstick Glucose have NaN values in VALUEUOM however we have measurements for these in VALUE col
        "pH": lambda df: Events.clean_lab(df),
        "Temperature": lambda df: Events.clean_temperature(df),
    }

    ##############################################
    # Mapping functions for time-series data
    ##############################################

    def map_itemids_to_variables(self, events: pd.DataFrame) -> pd.DataFrame:
        """Map ITEMIDs to standardized variable names."""
        mapped_events = events.merge(
            self.data_manager.variable_map,
            left_on="ITEMID",
            right_index=True,
            how="inner",
        )
        if mapped_events.empty:
            unmapped = events["ITEMID"].unique()
            logger.warning(
                f"No ITEMIDs were successfully mapped to variables. Unmapped ITEMIDs: {unmapped}"
            )
            return mapped_events
        self.events = mapped_events
        return self.events

    def remove_outliers_for_variable(
        self, events: pd.DataFrame, variable: str
    ) -> pd.DataFrame:
        """Remove outliers for a specific variable based on predefined ranges."""
        if variable not in self.data_manager.variable_ranges.index:
            return events

        idx = events["VARIABLE"] == variable

        V = pd.to_numeric(events.loc[idx, "VALUE"], errors="coerce").copy()

        outlier_low = self.data_manager.variable_ranges.at[variable, "OUTLIER_LOW"]
        outlier_high = self.data_manager.variable_ranges.at[variable, "OUTLIER_HIGH"]
        valid_low = self.data_manager.variable_ranges.at[variable, "VALID_LOW"]
        valid_high = self.data_manager.variable_ranges.at[variable, "VALID_HIGH"]

        V[V < outlier_low] = np.nan
        V[V > outlier_high] = np.nan
        V = V.clip(lower=valid_low, upper=valid_high)
        events.loc[idx, "VALUE"] = V
        self.events = events
        return events

    def impute_missing_values(
        self, events: pd.DataFrame, variable: str
    ) -> pd.DataFrame:
        if variable not in self.data_manager.variable_ranges.index:
            return events
        impute_value = self.data_manager.variable_ranges.at[variable, "IMPUTE"]
        idx = events["VARIABLE"] == variable
        events.loc[idx, "VALUE"] = (
            events.loc[idx, "VALUE"].fillna(impute_value).infer_objects(copy=False)
        )
        self.events = events
        return events

    def add_hours_since_admission(self, events: pd.DataFrame) -> pd.DataFrame:
        """Add hours since admission to the events DataFrame."""
        intime = self.patient.icu_stays[0].INTIME
        events = events.copy()
        events["CHARTTIME"] = pd.to_datetime(events["CHARTTIME"])
        events["HOURS"] = (events["CHARTTIME"] - intime).dt.total_seconds() / 3600
        events = events[
            events["HOURS"] >= 0
        ].copy()  # filter out negative hours, recorded before admission (might have been an artifact, maybe a data entry delay or the patient was in emergency)
        return events

    def convert_to_timeseries(
        self, events: pd.DataFrame, variable_column="VARIABLE", variables=[]
    ) -> pd.DataFrame:
        required_cols = ["CHARTTIME", "ICUSTAY_ID", "HOURS"]
        if events.empty or not all(col in events.columns for col in required_cols):
            logger.warning(
                "Events DataFrame is empty or missing required columns for timeseries conversion."
            )
            return pd.DataFrame()
        metadata = (
            events[["CHARTTIME", "ICUSTAY_ID", "HOURS"]]
            .sort_values(by=["CHARTTIME", "ICUSTAY_ID"])
            .drop_duplicates(keep="first")
            .set_index("CHARTTIME")
        )
        timeseries = (
            events[["CHARTTIME", variable_column, "VALUE"]]
            .sort_values(by=["CHARTTIME", variable_column, "VALUE"], axis=0)
            .drop_duplicates(subset=["CHARTTIME", variable_column], keep="last")
        )
        timeseries = (
            timeseries.pivot(index="CHARTTIME", columns=variable_column, values="VALUE")
            .merge(metadata, left_index=True, right_index=True)
            .sort_index(axis=0)
            .reset_index()
        )
        for variable in variables:
            if variable not in timeseries:
                timeseries[variable] = np.nan

        static_cols = [
            "ICUSTAY_ID",
            "HADM_ID",
            "SUBJECT_ID",
            "AGE",
            "GENDER",
            "ETHNICITY",
            "ICU_TYPE",
        ]
        static_wide = self.patient.icustays_df[static_cols].copy()
        if "ICUSTAY_ID" in static_wide.columns:
            static_wide = static_wide.set_index("ICUSTAY_ID")
        static_wide.index = static_wide.index.astype(int)
        icustay_id = int(timeseries["ICUSTAY_ID"].iloc[0])
        static_row = static_wide.loc[[icustay_id]].reset_index(drop=True)
        static_df = pd.concat([static_row] * len(timeseries), ignore_index=True)
        timeseries = pd.concat([timeseries.reset_index(drop=True), static_df], axis=1)
        self._timeseries = timeseries
        return self._timeseries

    @staticmethod
    def to_binned_timeseries(
        timeseries: pd.DataFrame,
        stats_csv: Path | str,
        n_timesteps=24,
        task="mortality",
    ) -> pd.DataFrame:
        """adapted from https://github.com/layer6ai-labs/DuETT/blob/master/physionet.py#L83
        they choose n_timesteps=32 in the paper
        we might increase this as well since we're passing only 1/3 of features
        and DP noise + small transformer might challenge results"""
        if timeseries.empty:
            logger.warning(
                "Timeseries DataFrame is empty, returning empty DataFrame for patient."
            )
            return pd.DataFrame()
        if task == "mortality":
            timeseries = timeseries[timeseries["HOURS"] <= 48]
        elif task == "phenotypes":
            pass

        feature_stats = pd.read_csv(stats_csv, index_col=0)
        means = feature_stats["mean"].to_dict()
        stds = feature_stats["std"].to_dict()
        expected_cols = list(
            means.keys()
        )  # when we discard hours > 48, we also lose some lab events cols we need to add them back

        for col in expected_cols:
            if col not in timeseries.columns:
                logger.warning(
                    f"Column {col} is missing in timeseries, filling with NaN."
                )
                timeseries[col] = np.nan

        timeseries["BIN"] = np.where(
            np.isclose(timeseries["HOURS"], max(timeseries["HOURS"])),
            n_timesteps - 1,
            (timeseries["HOURS"] / max(timeseries["HOURS"]) * n_timesteps).astype(int),
        )
        timeseries.drop(columns=["CHARTTIME"], inplace=True, errors="ignore")

        timeseries = timeseries[
            (timeseries["BIN"] >= 0)
            & (timeseries["BIN"] < n_timesteps)
            & (~timeseries["HOURS"].isna())
        ]

        normalize_cols = [col for col in expected_cols if col in timeseries.columns]

        for col in normalize_cols:
            mean = means.get(col, 0)
            std = stds.get(col, 1e-7)
            timeseries[col] = pd.to_numeric(timeseries[col], errors="coerce")
            timeseries[f"{col}_NORM"] = (timeseries[col] - mean) / (std + 1e-7)

        bin_edges = (np.arange(1, n_timesteps + 1) / n_timesteps) * max(
            timeseries["HOURS"]
        )
        bin_ends = pd.Series(
            bin_edges, name="BIN_END_HOURS", index=np.arange(n_timesteps)
        )
        norm_cols = [f"{col}_NORM" for col in expected_cols]  # keep the same cols order
        binned_timeseries = timeseries[["BIN"] + norm_cols].copy()
        binned_timeseries.columns = ["BIN"] + [
            col.replace("_NORM", "") for col in norm_cols
        ]
        binned_timeseries["BIN_END_HOURS"] = binned_timeseries["BIN"].map(bin_ends)
        cols = (
            ["BIN"]
            + [
                col
                for col in binned_timeseries.columns
                if col not in ["BIN", "BIN_END_HOURS"]
            ]
            + ["BIN_END_HOURS"]
        )
        binned_timeseries = binned_timeseries[cols]
        return binned_timeseries  # BIN, lab events, demographics, BIN_END_HOURS

    def validate_events(
        self, events_df: pd.DataFrame, icustays_df: pd.DataFrame
    ) -> bool:
        n_events = 0  # total number of events
        empty_hadm = 0  # HADM_ID is empty in events
        no_hadm_in_stay = 0  # HADM_ID does not appear in icu_stays
        no_icustay = 0  # ICUSTAY_ID is empty in events, we try to recover them
        recovered = (
            0  # empty ICUSTAY_IDs are recovered according to icu_stays (given HADM_ID)
        )
        could_not_recover = (
            0  # empty ICUSTAY_IDs that are not recovered. This should be zero.
        )
        icustay_missing_in_stays = (
            0  # ICUSTAY_ID does not appear in icu_stays, we exlude them
        )

        icustays_df = icustays_df.copy()
        n_events += events_df.shape[0]
        empty_hadm = events_df["HADM_ID"].isnull().sum()
        events_df = events_df.dropna(subset=["HADM_ID"])
        merged_df = events_df.merge(
            icustays_df,
            left_on=["HADM_ID"],
            right_on=["HADM_ID"],
            how="left",
            suffixes=["", "_r"],
            indicator=True,
        )

        no_hadm_in_stay += (merged_df["_merge"] == "left_only").sum()
        merged_df = merged_df[merged_df["_merge"] == "both"]

        cur_no_icustay = merged_df["ICUSTAY_ID"].isnull().sum()
        no_icustay += cur_no_icustay
        merged_df.loc[:, "ICUSTAY_ID"] = merged_df["ICUSTAY_ID"].fillna(
            merged_df["ICUSTAY_ID_r"]
        )
        recovered += cur_no_icustay - merged_df["ICUSTAY_ID"].isnull().sum()
        could_not_recover += merged_df["ICUSTAY_ID"].isnull().sum()
        merged_df = merged_df.dropna(subset=["ICUSTAY_ID"])

        icustay_missing_in_stays += (
            merged_df["ICUSTAY_ID"] != merged_df["ICUSTAY_ID_r"]
        ).sum()
        merged_df = merged_df[(merged_df["ICUSTAY_ID"] == merged_df["ICUSTAY_ID_r"])]

        final_df = merged_df[
            [
                "SUBJECT_ID",
                "HADM_ID",
                "ICUSTAY_ID",
                "CHARTTIME",
                "ITEMID",
                "VALUE",
                "VALUEUOM",
            ]
        ]

        assert could_not_recover == 0

        # logger.warning(
        # f"n_events: {n_events}, "
        # f"empty_hadm: {empty_hadm}, "
        # f"no_hadm_in_stay: {no_hadm_in_stay}, "
        # f"no_icustay: {no_icustay}, "
        # f"recovered: {recovered}, "
        # f"could_not_recover: {could_not_recover}, "
        # f"icustay_missing_in_stays: {icustay_missing_in_stays}"
        # )

        return final_df

    def clean_events(self, events: pd.DataFrame) -> pd.DataFrame:
        for variable, clean_fn in self.clean_fns.items():
            idx = events["VARIABLE"] == variable
            if idx.any():
                cleaned_values = clean_fn(events.loc[idx].copy())
                events.loc[idx, "VALUE"] = cleaned_values
        events = events.loc[events["VALUE"].notnull()].copy()
        return events

    def get_cleaned_events(self) -> pd.DataFrame:
        if self._cleaned_events is None:
            if self.events is None or self.events.empty:
                self._cleaned_events = pd.DataFrame()
                return self._cleaned_events

            self.events = self.validate_events(self.events, self.get_icustays_df)
            mapped_events = self.map_itemids_to_variables(self.events)
            cleaned_events = self.clean_events(mapped_events)

            for variable in cleaned_events["VARIABLE"].unique():
                cleaned_events = self.remove_outliers_for_variable(
                    cleaned_events, variable
                )
                cleaned_events = self.impute_missing_values(cleaned_events, variable)
                cleaned_events = self.add_hours_since_admission(cleaned_events)

            timeseries = self.convert_to_timeseries(
                cleaned_events,
                variable_column="VARIABLE",
                variables=cleaned_events["VARIABLE"].unique(),
            )

            self._cleaned_events = cleaned_events
            self._timeseries = timeseries

        return self._cleaned_events, self._timeseries

    @staticmethod
    def get_cleaned_events_static(events_obj):
        return events_obj.get_cleaned_events()

    @property  # since they are cached they will not call get_cleaned_events again
    def cleaned_events(self) -> pd.DataFrame:
        if self._cleaned_events is None:
            self.get_cleaned_events()
        return self._cleaned_events

    @property
    def timeseries(self) -> pd.DataFrame:
        if self._timeseries is None:
            self.get_cleaned_events()
        return self._timeseries

    #####################################
    # Time-series cleaning functions
    #####################################

    def clean_sbp(df):
        """Clean systolic blood pressure (SBP) values."""
        variable = df["VALUE"].astype(str)
        idx = variable.apply(lambda s: "/" in s)
        variable.loc[idx] = variable[idx].apply(
            lambda s: re.match("^(\d+)/(\d+)$", s).group(1)
        )
        return variable.astype(float)

    def clean_dbp(df):
        variable = df["VALUE"].astype(str)
        idx = variable.apply(lambda s: "/" in s)
        variable.loc[idx] = variable[idx].apply(
            lambda s: re.match("^(\d+)/(\d+)$", s).group(2)
        )
        return variable.astype(float)

    def clean_crr(df):
        """Clean capillary refill rate (CRR) values."""
        variable = pd.Series(np.nan, index=df.index)
        df["VALUE"] = df["VALUE"].astype(
            str
        )  # when df.VALUE is empty, dtype can be float and raises an exception, to fix this we change dtype to str
        variable.loc[(df["VALUE"] == "Normal <3 secs") | (df["VALUE"] == "Brisk")] = 0
        variable.loc[
            (df["VALUE"] == "Abnormal >3 secs") | (df["VALUE"] == "Delayed")
        ] = 1
        return variable

    def clean_fio2(df):
        """Clean fraction of inspired oxygen (FIO2) values."""
        variable = df["VALUE"].astype(float)
        is_str = np.array([isinstance(x, str) for x in df["VALUE"]], dtype=bool)
        idx = df["VALUEUOM"].fillna("").apply(lambda s: "torr" not in s.lower()) & (
            is_str | (~is_str & (variable > 1.0))
        )
        variable.loc[idx] = variable[idx] / 100.0
        return variable

    def clean_lab(df):
        """Clean lab values by replacing invalid entries with NaN."""
        variable = df["VALUE"].copy()
        idx = variable.apply(
            lambda s: isinstance(s, str) and not (re.match(r"^(\d+(\.\d*)?|\.\d+)$", s))
        )
        variable.loc[idx] = np.nan
        return pd.to_numeric(variable, errors="coerce")

    def clean_o2sat(df):
        """Clean oxygen saturation (O2SAT) values. O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale."""
        variable = df["VALUE"].copy()
        idx = variable.apply(
            lambda s: isinstance(s, str)
            and not bool(re.match(r"^(\d+(\.\d*)?|\.\d+)$", s))
        )
        variable.loc[idx] = np.nan
        variable = pd.to_numeric(variable, errors="coerce")
        idx = variable <= 1
        variable.loc[idx] = variable[idx] * 100.0
        return variable

    def clean_temperature(df):
        """Clean temperature values. Map Fahrenheit to Celsius, some ambiguous 50<x<80"""
        variable = df["VALUE"].astype(float)
        idx = (
            df["VALUEUOM"].fillna("").apply(lambda s: "F" in s.lower())
            | df["MIMIC_LABEL"].apply(lambda s: "F" in s.lower())
            | (variable >= 79)
        )
        variable.loc[idx] = (variable[idx] - 32) * 5.0 / 9
        return variable


@dataclass
class Patient(PandasMixin):
    subject_id: int
    data: Dict[str, pd.DataFrame] = field(
        default_factory=lambda: defaultdict(pd.DataFrame)
    )
    icu_stays: List[ICUStay] = field(
        default_factory=list
    )  # we might discard this since now we only do per hospital and remove multiple icu stays
    events: List[Events] = field(
        default_factory=list
    )  # filtered, cleaned time-series events

    def add_table(self, table_name: str, df: pd.DataFrame):
        self.data[table_name] = df.reset_index(drop=True)

    def get_table(self, table_name: str) -> pd.DataFrame:
        return self.data.get(table_name, pd.DataFrame())

    def add_icustay(self, icu_row: dict, merged_stays: pd.DataFrame):
        try:
            assert not merged_stays["ICUSTAY_ID"].isnull().any()
            assert not merged_stays["HADM_ID"].isnull().any()
            assert len(merged_stays["ICUSTAY_ID"].unique()) == len(
                merged_stays["ICUSTAY_ID"]
            )

            patient_data = merged_stays.loc[
                merged_stays["SUBJECT_ID"] == self.subject_id
            ]

            if patient_data.empty:
                raise ValueError(f"No data found for subject_id {self.subject_id}")

            stay = ICUStay(
                ICUSTAY_ID=int(icu_row["ICUSTAY_ID"]),
                SUBJECT_ID=int(icu_row["SUBJECT_ID"]),
                HADM_ID=int(icu_row["HADM_ID"]),
                FIRST_CAREUNIT=icu_row["FIRST_CAREUNIT"],
                LAST_CAREUNIT=icu_row["LAST_CAREUNIT"],
                FIRST_WARDID=icu_row["FIRST_WARDID"],
                LAST_WARDID=icu_row["LAST_WARDID"],
                INTIME=pd.to_datetime(icu_row["INTIME"]),
                OUTTIME=pd.to_datetime(icu_row["OUTTIME"]),
                LOS=icu_row["LOS"],
                DOB=pd.to_datetime(icu_row["DOB"]),
                DOD=pd.to_datetime(icu_row["DOD"]),
                DISCHTIME=pd.to_datetime(icu_row["DISCHTIME"]),
                ADMITTIME=pd.to_datetime(icu_row["ADMITTIME"]),
                DEATHTIME=pd.to_datetime(icu_row["DEATHTIME"]),
                GENDER=icu_row["GENDER"],
                ETHNICITY=icu_row["ETHNICITY"],
            )
            stay.GENDER = stay.encode_gender()
            stay.ETHNICITY = stay.encode_ethnicity()
            stay.AGE = stay.age_at_admission()
            stay.MORTALITY_INUNIT = stay.encode_inunit_mortality()
            stay.MORTALITY_INHOSPITAL = stay.encode_inhospital_mortality()
            stay.ICU_TYPE = stay.encode_icutype()
            self.icu_stays.append(stay)

        except Exception as e:
            logging.warning(
                f"Failed to parse ICU stay for subject {self.subject_id}: {e}"
            )

    def add_events(self, events: Events):
        self.events = events

    def filter_diagnoses_on_stays(diagnoses, icustays_df) -> pd.DataFrame:
        """Filter diagnoses based on stays."""
        return diagnoses.merge(
            icustays_df[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].drop_duplicates(),
            how="inner",
            left_on=["SUBJECT_ID", "HADM_ID"],
            right_on=["SUBJECT_ID", "HADM_ID"],
        )

    @cached_property
    def icustays_df(self) -> pd.DataFrame:
        return self.to_dataframe(self.icu_stays)

    @property
    def available_tables(self) -> List[str]:
        return list(self.data.keys())

    @staticmethod
    def process_patient(pid_patient_tuple):
        pid, patient = pid_patient_tuple

        stays = ICUStay.filter_admissions_on_nb_icustays(patient.icustays_df)
        if len(patient.icu_stays) > 1:
            hadm_ids = [stay.HADM_ID for stay in patient.icu_stays]
            logger.warning(
                f"Check multiple stays for the same admission: Patient {pid} has {len(patient.icu_stays)} ICU stays: HADM_IDs {hadm_ids}"
            )
            logger.warning(
                f"After filtering by ICU stay count, {len(stays)} stays remain for Patient {pid}"
            )

        removed_due_to_age = []
        removed_due_to_transfer = []
        filtered_stays = []

        for stay in stays:
            is_adult = stay.is_adult()
            is_transfer_free = stay.is_transfer_free()
            if not is_adult:
                removed_due_to_age.append(stay)
            if not is_transfer_free:
                removed_due_to_transfer.append(stay)
            if is_adult and is_transfer_free:
                filtered_stays.append(stay)

        stays = filtered_stays
        logger.warning(
            f"Removed {len(removed_due_to_age)} patients due to age restriction"
        )
        logger.warning(
            f"Removed {len(removed_due_to_transfer)} patients due to transfer restriction"
        )
        logger.warning(
            f"After applying age and transfer filters, {len(stays)} stays remain for Patient {pid}"
        )

        # sanity check: since we'd like to give per-patient privacy guarantees and to avoid linking multiple stays for the same patient which could increase reidentification risk
        if len(stays) == 1:
            patient.icu_stays = stays
            logger.info(
                f"Patient {pid} has exactly one valid ICU stay, keeping the patient."
            )
            return (pid, patient)
        else:
            logger.warning(
                f"Patient {pid} has {len(stays)} ICU stays, excluding from final list."
            )
            return None

    def batch_generator(pids: List[int], batch_size: int):
        for i in range(0, len(pids), batch_size):
            yield pids[i : i + batch_size]

    @classmethod
    def build_patient_database(
        cls,
        tables: Dict[str, pd.DataFrame],
        filter_single_stay: bool = True,
        csv_path: Optional[str] = None,
        batch_size: int = 1000,
        max_workers: int = 8,
    ) -> Tuple[Dict[int, "Patient"], Dict[int, "Patient"]]:

        merged_stays = tables.get("stays_admits_patients")
        if merged_stays is None:
            raise ValueError("stays_admits_patients table is missing.")

        subject_ids = merged_stays["SUBJECT_ID"].unique()
        all_patients = {}
        all_filtered_patients = {}

        for batch_subject_ids in cls.batch_generator(subject_ids, batch_size):
            batch_tables = {
                name: df[df["SUBJECT_ID"].isin(batch_subject_ids)]
                for name, df in tables.items()
                if "SUBJECT_ID" in df.columns
            }
            patients = defaultdict(lambda: Patient(None))
            
            for table_name, df in batch_tables.items():
                for subject_id, sub_df in df.groupby("SUBJECT_ID"):
                    if patients[subject_id].subject_id is None:
                        patients[subject_id].subject_id = subject_id
                    patients[subject_id].add_table(table_name, sub_df)
            
            for _, row in merged_stays[merged_stays["SUBJECT_ID"].isin(batch_subject_ids)].iterrows():
                subject_id = row["SUBJECT_ID"]
                if subject_id in patients:
                    patients[subject_id].add_icustay(row.to_dict(), merged_stays)
            
            logger.info(f"Filtering patients for batch {batch_subject_ids[0]}-{batch_subject_ids[-1]}...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(cls.process_patient, patients.items()))
            filtered_patients = {
                pid: patient
                for result in results
                if result is not None
                for pid, patient in [result]
            }
            logger.info(f"Batch filtered: {len(filtered_patients)} patients")
            all_filtered_patients.update(filtered_patients)
            
            del patients
            del filtered_patients
        all_patients = None    

        logger.info(f"Total patients after all batches: {len(all_filtered_patients)}")
        return all_patients, all_filtered_patients

            
    @staticmethod
    def get_stats(all_timeseries: list, columns_of_interest: list) -> pd.DataFrame:
        if not all_timeseries:
            return pd.DataFrame()

        df = pd.concat(all_timeseries, ignore_index=True)
        df = df[columns_of_interest].apply(pd.to_numeric, errors="coerce")

        stats = []
        for col in columns_of_interest:
            vals = df[col].values.flatten()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                stats.append(
                    {
                        "VARIABLE": col,
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "max": np.nan,
                        "count": 0,
                    }
                )
            else:
                stats.append(
                    {
                        "VARIABLE": col,
                        "mean": np.round(np.mean(vals), 3),
                        "std": np.round(np.std(vals), 3),
                        "min": np.round(np.min(vals), 3),
                        "max": np.round(np.max(vals), 3),
                        "count": len(vals),
                    }
                )
        stats_df = pd.DataFrame(stats)
        return stats_df

    def __repr__(self):
        careunits = (
            list(self.icustays_df["FIRST_CAREUNIT"].unique())
            if not self.icustays_df.empty
            else []
        )
        return f"<Patient {self.subject_id}, {len(self.data)} tables, {len(self.icu_stays)} ICU stays, Careunits: {careunits}>"


class MIMIC3Dataset(Dataset):
    def __init__(self, features, labels):
        self.features = [
            torch.tensor(f, dtype=torch.float32) if not torch.is_tensor(f) else f
            for f in features
        ]

        if isinstance(labels[0], (list, tuple, np.ndarray, pd.Series)):
            self.labels = [torch.tensor(l, dtype=torch.float32) for l in labels]
        else:
            self.labels = torch.tensor(
                labels,
                dtype=torch.float32 if isinstance(labels[0], float) else torch.long,
            )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label


class HospitalUnit:
    def __init__(self, patients, padded_seq):
        self.patients = patients
        self.padded_seq = padded_seq

    def get_features(self):
        return [padded_seq for padded_seq in self.padded_seq]

    def get_labels(self, task):
        return [task.get_label(patient) for patient in self.patients]

    def build_dataset(self, task):
        features = self.get_features()
        labels = self.get_labels(task)
        return MIMIC3Dataset(features, labels)

    def split_dataset(self, dataset, split_ratio=0.8, seed=42):
        total_len = len(dataset)
        train_len = int(total_len * split_ratio)
        dev_len = int((total_len - train_len) / 2)
        test_len = total_len - train_len - dev_len
        lengths = [train_len, dev_len, test_len]
        generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, lengths, generator=generator)

    def partition_dataset(
        self, dataset, split_method=1, num_clients=2, alpha=0.5, is_shuffle=True
    ):
        train_set, dev_set, test_set = self.split_dataset(dataset)
        client_datasets = partition_data(
            train_set,
            split_method,
            num_clients=num_clients,
            #alpha=alpha,
            is_shuffle=is_shuffle,
        )
        return client_datasets, dev_set, test_set

    @staticmethod
    def get_hospital_unit_info(ds):
        if hasattr(ds, "dataset"):
            dataset = ds.dataset
        else:
            dataset = ds

        num_patients = len(dataset)
        labels = []
        for i in range(num_patients):
            sample = dataset[i]
            if isinstance(sample, (tuple, list)) and len(sample) > 1:
                labels.append(sample[1].item() if hasattr(sample[1], "item") else sample[1])
            elif isinstance(sample, dict) and "label" in sample:
                labels.append(sample["label"])
        label_counts = {}
        if labels:
            unique, counts = np.unique(labels, return_counts=True)
            label_counts = dict(zip(unique, counts))
        return {
            "num_patients": num_patients,
            "label_distribution": label_counts,
            "example_labels": labels[:5],
        }
    
    @classmethod
    def show_distribution(cls, client_datasets, num_clients, split_method="iid"):
        logger.info(f"Created {num_clients} hospital units.")
        logger.info(f"Used split method: {split_method}")
        for unit_id in range(num_clients):
            ds = client_datasets[unit_id]
            info = cls.get_hospital_unit_info(ds)
            logger.info(
                f"Hospital Unit {unit_id}: {info['num_patients']} patients, "
                f"Label distribution: {info['label_distribution']}, "
                f"Example labels: {info['example_labels']}"
            )


if __name__ == "__main__":
    logger.info("Loading all MIMIC tables...")

    data_manager = DataManager(
        data_path=DATASET_PATH_DEMO,
        diagnoses_map_path=DATASET_PATH_DEMO
        / "D_ICD_DIAGNOSES.csv",
        diagnoses_path=DATASET_PATH_DEMO
        / "DIAGNOSES_ICD.csv",
        phenotype_definitions_path=Path("/home/tanalp/thesis/dpnlp")
        / "mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml",
        hcup_ccs_2015_path=Path("/home/tanalp/thesis/dpnlp")
        / "mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml",
        var_map_path=Path("/home/tanalp/thesis/dpnlp") / "mimic3benchmark/resources/itemid_to_variable_map.csv",
        var_ranges_path=Path("/home/tanalp/thesis/dpnlp") / "mimic3benchmark/resources/variable_ranges.csv",
        variable_column="LEVEL2",
    )
    
    #################### FULL MIMIC-III ###########################

    # data_manager = DataManager(
    #     data_path=DATASET_PATH,
    #     diagnoses_map_path=DATASET_PATH
    #     / "D_ICD_DIAGNOSES.csv.gz",
    #     diagnoses_path=DATASET_PATH
    #     / "DIAGNOSES_ICD.csv.gz",
    #     phenotype_definitions_path=Path("/home/tanalp/thesis/dpnlp")
    #     / "mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml",
    #     hcup_ccs_2015_path=Path("/home/tanalp/thesis/dpnlp")
    #     / "mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml",
    #     var_map_path=Path("/home/tanalp/thesis/dpnlp") / "mimic3benchmark/resources/itemid_to_variable_map.csv",
    #     var_ranges_path=Path("/home/tanalp/thesis/dpnlp") / "mimic3benchmark/resources/variable_ranges.csv",
    #     variable_column="LEVEL2",
    # )
    ################################################
    # Build Patient Database
    ################################################

    tables = data_manager.load_all_tables(data_manager.data_path)
    logger.info("Building patient database...")
    full_db, filtered_db = Patient.build_patient_database(
        tables, filter_single_stay=True
    )
    logger.info(f"Total patients: {len(full_db)}")
    logger.info(f"Filtered patients (single stay, no transfers): {len(filtered_db)}")

    ################################################
    # Timeseries Data
    ################################################

    all_timeseries = []
    binned_timeseries = []
    binned_tensors = []
    concat_tensors = []
    lengths = []
    padded_seq = []
    # quick fix for now to add missing variables to timeseries if the patient has them missing
    columns_of_interest = [
        "Capillary refill rate",
        "Diastolic blood pressure",
        "Fraction inspired oxygen",
        "Glascow coma scale eye opening",
        "Glascow coma scale motor response",
        "Glascow coma scale total",
        "Glascow coma scale verbal response",
        "Glucose",
        "Heart Rate",
        "Mean blood pressure",
        "Oxygen saturation",
        "Respiratory rate",
        "Systolic blood pressure",
        "Temperature",
        "pH",
        "AGE",
        "GENDER",
        "ETHNICITY",
        "ICU_TYPE",
    ]
    events_objs = []
    for idx, (pid, patient) in enumerate(filtered_db.items(), 1):
        logger.info(f"Loading events for {pid} ({idx}/{len(filtered_db)})...")
        events = Events(
            data_manager=data_manager,
            patient=patient,
        )
        patient.add_events(events)
        events_objs.append(events)

    with ProcessPoolExecutor(max_workers=8) as executor:
        cleaned_results = list(
            executor.map(Events.get_cleaned_events_static, events_objs)
        )


# factor out this inside HospitalUnit class - also do the padding with collate

    for events in events_objs:
        timeseries = events.timeseries
        binned_timeseries_df = Events.to_binned_timeseries(
            timeseries,
            stats_csv="/home/tanalp/thesis/dpnlp/thesis/dpnlp_lib/src/dataset/features_stats_3.csv",
            n_timesteps=24,
            task="mortality",
        )
        input_cols = [col for col in binned_timeseries_df.columns]
        mask = ~binned_timeseries_df[input_cols].isna()
        mask_tensor = torch.tensor(mask.values, dtype=torch.float32)
        mask_impute = binned_timeseries_df[input_cols] == 0.0
        mask_impute = torch.tensor(mask_impute.values, dtype=torch.float32)
        input_tensor = torch.tensor(
            binned_timeseries_df[input_cols].fillna(0).values, dtype=torch.float32
        )
        binned_timeseries.append(binned_timeseries_df)
        binned_tensors.append(
            {
                "input_tensor": input_tensor,
                "mask_tensor": mask_tensor,
                "mask_impute": mask_impute,
            }
        )
        concat_tensor = torch.cat(
        [input_tensor, mask_tensor.float(), mask_impute.float()], dim=-1
        )
        concat_tensors.append(concat_tensor)
        lengths.append(concat_tensor.shape[0])
    
        
    valid_concat_tensors = [t for t in concat_tensors if t is not None and t.shape[0] > 0]
    padded = pad_sequence(valid_concat_tensors, batch_first=True, padding_value=0.0)
    for t in padded:
        padded_seq.append(t)


    ####################################################
    # Create HospitalUnit and Arrange Data Heterogeneity
    ####################################################

    task = MortalityTask() # pass config args
    hospital = HospitalUnit(list(filtered_db.values()), padded_seq)
    dataset = hospital.build_dataset(task=task)
    train, test, dev = hospital.split_dataset(dataset, split_ratio=0.8, seed=42)  
    client_datasets, dev_set, test_set = hospital.partition_dataset(train, split_method=1, num_clients=2, alpha=0.5, is_shuffle=True)
    hospital.show_distribution(client_datasets, num_clients=2, split_method="iid")
    breakpoint()

    ################################################
    # Test cases
    ################################################

    patients = list(filtered_db.values())
    num_patients = len(patients)
    num_icustays = sum(len(p.icu_stays) for p in patients)
    num_events = sum(
        len(p.events.events) for p in patients if hasattr(p.events, "events")
    )

    expected_count_patient = 33798
    expected_count_icustay = 42276
    expected_count_events = 250_000_000

    assert num_patients == expected_count_patient, (
        f"Expected {expected_count_patient} patients, found {num_patients}"
    )
    assert num_icustays == expected_count_icustay, (
        f"Expected {expected_count_icustay} ICU stays, found {num_icustays}"
    )
    assert num_events > expected_count_events, (
        f"Expected more than {expected_count_events} events, found {num_events}"
    )

    logger.info(f"Number of patients: {num_patients}")
    logger.info(f"Number of ICU stays: {num_icustays}")
    logger.info(f"Number of events: {num_events}")
    logger.info("All data checks passed successfully!")


   
    breakpoint()
    # for col in columns_of_interest:
    #     if col not in timeseries.columns:
    #         timeseries[col] = np.nan

    # if not timeseries.empty:
    #      all_timeseries.append(timeseries)

    # stats_df = Patient.get_stats(all_timeseries, columns_of_interest=columns_of_interest)
    # stats_df.to_csv(DATASET_PATH / "features_stats_3.csv", index=False)
