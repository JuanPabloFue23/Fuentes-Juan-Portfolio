import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score
)
from features import engineer_features

def evaluate_model():
    # 1. Load Config and Model
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model = joblib.load(config['paths']['model_export'])
    
    # 2. Load and Prep Data
    df = pd.read_csv(config['data']['raw_path'])
    df = engineer_features(df)
    
    # Note: In a real project, you'd pull the exact test split used in train.py
    # For this portfolio, we'll demonstrate on the whole set or a held-out slice
    X = df.drop(columns=[config['data']['target'], 'machine_id'])
    y = df[config['data']['target']]
    
    # 3. Generate Predictions
    y_pred = model.predict(X)
    y_probs = model.predict_proba(X)[:, 1]

    # --- VISUALIZATION 1: Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Failure', 'Failure'], 
                yticklabels=['No Failure', 'Failure'])
    plt.title('Confusion Matrix: Predictive Maintenance')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()

    # --- VISUALIZATION 2: Precision-Recall Curve ---
    # Better than ROC for imbalanced data
    precision, recall, _ = precision_recall_curve(y, y_probs)
    ap_score = average_precision_score(y, y_probs)

    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (Average Precision = {ap_score:.2f})')
    plt.savefig('reports/precision_recall_curve.png')
    plt.close()

    # 4. Save Text Report
    report = classification_report(y, y_pred)
    print("Evaluation Report:\n", report)
    with open("reports/model_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    import os
    os.makedirs('reports', exist_ok=True)
    evaluate_model()