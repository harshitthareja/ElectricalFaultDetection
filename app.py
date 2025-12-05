import streamlit as st
st.set_page_config(page_title="Electrical Fault Detection", layout="centered")

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_units):
        super(NeuralNetwork, self).__init__()
        layers = []
        in_size = input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_size, hidden_units))
            layers.append(nn.ReLU())
            in_size = hidden_units
        layers.append(nn.Linear(in_size, 1))
        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.model(x))

@st.cache_resource
def load_all():
    feature_names = joblib.load("feature_names.pkl")
    scaler = joblib.load("scaler.pkl")
    input_size = len(feature_names)
    HIDDEN_LAYERS = 3
    HIDDEN_UNITS = 128
    model = NeuralNetwork(input_size, HIDDEN_LAYERS, HIDDEN_UNITS)
    state_dict = torch.load("fault_model_state.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model, scaler, feature_names

model, scaler, feature_names = load_all()

st.title("Electrical Fault Detection System")

st.sidebar.header("Navigation")
app_mode = st.sidebar.radio("Select Mode", ["Single Prediction", "Batch Prediction"])

if app_mode == "Single Prediction":
    st.subheader("Enter Feature Values")
    user_inputs = {}
    cols = st.columns(2)
    for i, col in enumerate(feature_names):
        with cols[i % 2]:
            val = st.number_input(col, value=0.0, format="%.4f")
            user_inputs[col] = val

    if st.button("Predict Fault"):
        df = pd.DataFrame([user_inputs])
        scaled = scaler.transform(df)
        x_tensor = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            prob = model(x_tensor).item()
        pred = 1 if prob >= 0.5 else 0
        st.write(f"Probability: `{prob:.4f}`")
        st.write(f"Class: `{pred}`")
        if pred == 1:
            st.error("Fault Detected")
        else:
            st.success("No Fault Detected")

else:
    st.subheader("Batch Prediction from CSV")
    file = st.file_uploader("Upload CSV File", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head())
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            if st.button("Run Batch Prediction"):
                X = scaler.transform(df[feature_names])
                X_tensor = torch.tensor(X, dtype=torch.float32)
                with torch.no_grad():
                    probs = model(X_tensor).view(-1).numpy()
                preds = (probs >= 0.5).astype(int)
                result = df.copy()
                result["fault_probability"] = probs
                result["predicted_class"] = preds
                st.dataframe(result.head())
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "fault_predictions.csv", "text/csv")
