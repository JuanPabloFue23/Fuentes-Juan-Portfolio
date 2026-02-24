import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_preprocessing_pipeline(config):
    """
    Creates a production-grade preprocessing pipeline.
    - Numerical: Impute missing -> RobustScale (handles outliers better than Standard)
    - Categorical: Impute missing -> OneHotEncode (ignore unknown categories in production)
    """
    num_features = config['features']['numerical']
    cat_features = config['features']['categorical']

    # 1. Numerical Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()) 
    ])

    # 2. Categorical Pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop' # Drop machine_id and other non-features
    )

    return preprocessor

if __name__ == "__main__":
    # Test block
    config = load_config()
    pipe = get_preprocessing_pipeline(config)
    print("✅ Preprocessing Pipeline Initialized.")
    print(f"Targeting Numerical: {config['features']['numerical']}")
    print(f"Targeting Categorical: {config['features']['categorical']}")