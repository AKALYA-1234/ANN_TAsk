import gradio as gr
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# -------------------------------
# 1. Load Model & Encoders
# -------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ann_model.h5")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "bank_dataset_cleaned.csv")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

# Load Keras model
model = load_model(MODEL_PATH)

# Load dataset to extract columns and encoders
df = pd.read_csv(DATASET_PATH, encoding="latin1")

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
if "y" in categorical_cols:
    categorical_cols.remove("y")  # target column
numeric_cols = [col for col in df.columns if col not in categorical_cols + ["y"]]

# Fit encoders for categorical columns
encoders = {}
for col in categorical_cols + ["y"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

target_encoder = encoders["y"]

# -------------------------------
# 2. Prediction Function
# -------------------------------
def predict_func(*inputs):
    try:
        data = {}
        i = 0
        # Encode categorical columns
        for col in categorical_cols:
            le = encoders[col]
            if inputs[i] not in le.classes_:
                return {"Error": f"Invalid input for {col}: {inputs[i]}"}
            data[col] = le.transform([inputs[i]])[0]
            i += 1
        # Numerical columns
        for col in numeric_cols:
            data[col] = float(inputs[i])
            i += 1

        X = pd.DataFrame([data])
        # Ensure numeric dtype
        X = X.astype(float)

        # Predict probability
        prob = float(model.predict(X)[0][0])

        # Return formatted probabilities
        return {"yes": round(prob, 4), "no": round(1 - prob, 4)}

    except Exception as e:
        return {"Error": str(e)}

# -------------------------------
# 3. Build Gradio UI
# -------------------------------
inputs = []

# Dropdowns for categorical columns
for col in categorical_cols:
    options = sorted(encoders[col].classes_.tolist())
    inputs.append(gr.Dropdown(choices=options, label=col))

# Number inputs for numerical columns
for col in numeric_cols:
    inputs.append(gr.Number(label=col))

output = gr.Label(label="Prediction")

app = gr.Interface(
    fn=predict_func,
    inputs=inputs,
    outputs=output,
    title="Bank Term Deposit Prediction",
    description="Predict whether a customer will subscribe to a term deposit (yes/no)."
)

# -------------------------------
# 4. Launch
# -------------------------------
if __name__ == "__main__":
    app.launch()
