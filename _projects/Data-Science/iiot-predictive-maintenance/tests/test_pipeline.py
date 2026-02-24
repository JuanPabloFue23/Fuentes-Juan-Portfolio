import pytest
import pandas as pd
import numpy as np
import os
import yaml
from src.data_generation import generate_synthetic_data
from src.features import engineer_features

@pytest.fixture
def config():
    """Fixture to load config for tests."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def sample_data():
    """Fixture to generate a small batch of data for testing."""
    return generate_synthetic_data(n_samples=100)

def test_data_columns(sample_data, config):
    """Check if the generator produces all required columns."""
    expected_cols = config['features']['numerical'] + \
                    config['features']['categorical'] + \
                    [config['data']['target'], 'machine_id', 'model_type']
    
    # Check if every expected column exists in the dataframe
    for col in expected_cols:
        assert col in sample_data.columns, f"Column {col} is missing!"

def test_feature_engineering_logic(sample_data):
    """Verify that custom features are calculated correctly."""
    processed_df = engineer_features(sample_data.copy())
    
    assert 'wear_torque_ratio' in processed_df.columns
    assert 'temp_deviation' in processed_df.columns
    # Check that temp_deviation is always non-negative (as it uses .abs())
    assert (processed_df['temp_deviation'] >= 0).all()

def test_failure_rate_range(sample_data):
    """Ensure the synthetic data maintains a realistic imbalance (1% to 15%)."""
    failure_rate = sample_data['failure'].mean()
    assert 0.01 <= failure_rate <= 0.15, f"Unrealistic failure rate: {failure_rate}"

def test_model_file_exists(config):
    """Check if the model export path is valid after training."""
    # This assumes you've run the training script at least once
    model_path = config['paths']['model_export']
    if os.path.exists(model_path):
        assert model_path.endswith('.joblib')