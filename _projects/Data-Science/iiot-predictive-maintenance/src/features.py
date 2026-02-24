# def add_features(df):
#     df["loan_to_income"] = df["loan_amount"] / df["income"]
#     df["credit_risk_band"] = pd.cut(
#         df["credit_score"],
#         bins=[300,580,670,740,850],
#         labels=["poor","fair","good","excellent"]
#     )
#     return df

def engineer_features(df):
    """
    Create domain-specific features based on IIOT physics.
    """
    # Power Factor: Interaction between Wear and Torque
    df['wear_torque_ratio'] = df['tool_wear_min'] * df['torque_nm']
    
    # Temperature deviation from 'safe' 25 degrees
    df['temp_deviation'] = (df['ambient_temp'] - 25).abs()
    
    return df