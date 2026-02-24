import pandas as pd
import numpy as np
from faker import Faker
import yaml
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_synthetic_data(n_samples=5000, seed=42):
    """Generates a realistic IIoT dataset for predictive maintenance."""
    np.random.seed(seed)
    fake = Faker()
    
    logging.info(f"Generating {n_samples} samples of synthetic industrial data...")

    # 1. Base Features
    data = {
        'machine_id': [fake.bothify(text='ID-####') for _ in range(n_samples)],
        'model_type': np.random.choice(['Alpha_v1', 'Beta_v2', 'Sigma_X'], n_samples, p=[0.5, 0.3, 0.2]),
        'provider': np.random.choice(['SKF', 'Bosch', 'Siemens', 'ABB'], n_samples),
        
        # Normal distribution: Ambient temperature around 25°C
        'ambient_temp': np.random.normal(25, 5, n_samples),
        
        # Exponential distribution: Most tools are new, few are very old
        'tool_wear_min': np.random.exponential(scale=60, size=n_samples),
        
        # Poisson distribution: Torque pulses/cycles
        'torque_nm': np.random.poisson(lam=45, size=n_samples)
    }

    df = pd.DataFrame(data)

    # 2. Physics-Based Interaction Logic
    # We define a 'risk_score' where tool wear and torque amplify each other
    # failure_risk = (wear * 0.04) + (torque * 0.02) + (interaction * 0.01) - bias
    interaction = (df['tool_wear_min'] * df['torque_nm']) / 100
    logit = (df['tool_wear_min'] * 0.05) + (df['torque_nm'] * 0.08) + (interaction * 0.2) - 18
    
    # Sigmoid function to get probability [0, 1]
    prob_failure = 1 / (1 + np.exp(-logit))
    
    # 3. Inject Noise and Class Imbalance
    # We want a rare failure rate (approx 3-7%)
    df['failure'] = np.random.binomial(1, prob_failure)
    
    # 4. Add "Sensor Drift" (Noise)
    df['ambient_temp'] += np.random.normal(0, 1, n_samples)
    
    logging.info(f"Data generation complete. Failure rate: {df.failure.mean():.2%}")
    return df

if __name__ == "__main__":
    # Load config to get paths
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    df = generate_synthetic_data(n_samples=10000)
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(config['data']['raw_path']), exist_ok=True)
    
    df.to_csv(config['data']['raw_path'], index=False)
    logging.info(f"Dataset saved to {config['data']['raw_path']}")