from pydantic import BaseModel, Field, validator
import pandas as pd

class SensorSchema(BaseModel):
    ambient_temp: float = Field(..., ge=-50, le=150)
    tool_wear_min: float = Field(..., ge=0)
    torque_nm: int = Field(..., ge=0)
    failure: int = Field(..., ge=0, le=1)

def validate_dataframe(df: pd.DataFrame):
    """Iteratively validate rows to ensure data integrity."""
    for record in df.to_dict(orient='records'):
        SensorSchema(**record)
    return True