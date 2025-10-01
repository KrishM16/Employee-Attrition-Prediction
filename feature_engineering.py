from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    # Encode categorical variables
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
