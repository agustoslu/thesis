import pytest
import logging
import pandas as pd
from builder import Patient, Events, DataManager
from utils import enable_info_logs

@pytest.fixture
def patient_and_events():
    events = Events(data_manager=data_manager, patient=patient)
    return patient, events

def check_number_of_patients(data_manager, expected_count):
    """ expected number patients https://arxiv.org/abs/1703.07771 """
    patients = data_manager.get_patients()
    assert len(patients) == expected_count, f"Expected {expected_count} patients, found {len(patients)}"
    logger.debug(f"Number of patients check passed: {len(patients)}")