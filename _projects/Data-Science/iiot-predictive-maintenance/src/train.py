import pandas as pd
import yaml
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Local imports
from preprocessing import get_preprocessing_pipeline
from features import engineer_features

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_training():
    # 1. Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. Load and Prepare Data
    logging.info("Loading synthetic dataset...")
    df = pd.read_csv(config['data']['raw_path'])
    df = engineer_features(df) # Apply our custom physics features
    
    X = df.drop(columns=[config['data']['target'], 'machine_id'])
    y = df[config['data']['target']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['data']['test_size'], random_state=config['data']['random_state']
    )

    # 3. Initialize Preprocessing
    preprocessor = get_preprocessing_pipeline(config)

    # 4. Define Models (Baseline vs Challenger)
    models = {
        "Baseline_Logistic": LogisticRegression(class_weight='balanced'),
        "Challenger_RF": RandomForestClassifier(
            n_estimators=config['model']['params']['n_estimators'],
            max_depth=config['model']['params']['max_depth'],
            class_weight='balanced'
        )
    }

    best_f1 = 0
    best_model = None

    # 5. Training Loop
    for name, model in models.items():
        logging.info(f"Training {name}...")
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        full_pipeline.fit(X_train, y_train)
        
        # Quick Evaluation
        preds = full_pipeline.predict(X_test)
        score = f1_score(y_test, preds)
        logging.info(f"{name} F1-Score: {score:.4f}")

        if score > best_f1:
            best_f1 = score
            best_model = full_pipeline

    # 6. Serialize the Champion Model
    logging.info(f"Saving champion model to {config['paths']['model_export']}")
    joblib.dump(best_model, config['paths']['model_export'])

if __name__ == "__main__":
    run_training()