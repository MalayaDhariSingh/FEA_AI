import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.preprocessing import StandardScaler

# --- 1. SETUP & MODEL DEFINITION ---
# We must reproduce the exact model structure to load weights
class FEA_Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# --- 2. LOAD DATA & REBUILD SCALERS ---
# We need to recreate the scalers used during training
@st.cache_data # Cache this so it doesn't reload on every slider move
def load_data_and_scalers():
    try:
        df = pd.read_csv('data_log.txt')
        df = df.dropna()
        
        X = df[['Radius', 'Load']].values
        y = df[['MaxStress']].values
        
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        scaler_X.fit(X)
        scaler_y.fit(y)
        
        return scaler_X, scaler_y
    except FileNotFoundError:
        st.error("Could not find data_log.txt. Make sure it is in the same folder!")
        return None, None

scaler_X, scaler_y = load_data_and_scalers()

# --- 3. LOAD MODEL WEIGHTS ---
@st.cache_resource
def load_model():
    model = FEA_Surrogate()
    # Load the saved weights (ensure the file name matches what you saved)
    try:
        model.load_state_dict(torch.load("fea_surrogate_model.pth"))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Model file 'fea_surrogate_model.pth' not found. Did you run train_surrogate.py?")
        return None

model = load_model()

# --- 4. STREAMLIT UI ---
st.title("🔩 Real-Time FEA Surrogate Model")
st.markdown("Predicting **Von Mises Stress** instantly using PyTorch.")

# Sidebar Controls
st.sidebar.header("Design Parameters")
radius = st.sidebar.slider("Hole Radius (mm)", 5.0, 15.0, 10.0, step=0.1)
load = st.sidebar.slider("Applied Load (Pressure)", 50.0, 200.0, 100.0, step=1.0)

# --- 5. PREDICTION LOGIC ---
if model and scaler_X and scaler_y:
    # Prepare input
    input_data = np.array([[radius, load]])
    
    # Scale input
    input_scaled = scaler_X.transform(input_data)
    input_tensor = torch.FloatTensor(input_scaled)
    
    # Predict
    with torch.no_grad():
        prediction_scaled = model(input_tensor)
    
    # Inverse scale output to get real MPa
    prediction_real = scaler_y.inverse_transform(prediction_scaled.numpy())
    predicted_stress = float(prediction_real[0][0])

    # --- 6. VISUALIZATION ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Geometry Preview")
        fig_geom, ax_geom = plt.subplots(figsize=(4, 2))
        
        # Draw Plate
        rect = patches.Rectangle((0, 0), 100, 50, linewidth=2, edgecolor='black', facecolor='lightgray')
        ax_geom.add_patch(rect)
        
        # Draw Hole
        circle = patches.Circle((50, 25), radius, linewidth=2, edgecolor='red', facecolor='white')
        ax_geom.add_patch(circle)
        
        # Draw Load Arrows
        ax_geom.arrow(105, 25, 10, 0, head_width=3, head_length=3, fc='blue', ec='blue')
        ax_geom.text(118, 25, f"Load: {load}", color='blue', va='center')

        ax_geom.set_xlim(-10, 140)
        ax_geom.set_ylim(-10, 60)
        ax_geom.axis('off')
        st.pyplot(fig_geom)

    with col2:
        st.subheader("AI Prediction")
        
        # Color logic: Green if safe (<350), Red if high stress
        delta_color = "normal"
        if predicted_stress > 400:
            delta_color = "inverse" # Red indicator usually
        
        st.metric(
            label="Max Von Mises Stress", 
            value=f"{predicted_stress:.2f} MPa",
            delta=f"Radius: {radius} mm"
        )
        
        if predicted_stress > 450:
            st.error("⚠️ CRITICAL STRESS LEVEL")
        elif predicted_stress < 200:
            st.success("✅ Design is Safe")
        else:
            st.warning("⚠️ Moderate Stress")

    # --- 7. CONTEXT PLOT (Real-time Trend) ---
    st.divider()
    st.subheader("Stress Trend Analysis")
    st.caption(f"How stress changes with Radius (at fixed Load = {load})")
    
    # Generate a curve for the current Load
    r_range = np.linspace(5, 15, 50)
    l_fixed = np.full_like(r_range, load)
    
    # Prepare batch for model
    batch_input = np.column_stack((r_range, l_fixed))
    batch_scaled = scaler_X.transform(batch_input)
    batch_tensor = torch.FloatTensor(batch_scaled)
    
    with torch.no_grad():
        batch_pred_scaled = model(batch_tensor)
    
    batch_pred_real = scaler_y.inverse_transform(batch_pred_scaled.numpy())
    
    # Plot
    fig_trend, ax_trend = plt.subplots(figsize=(8, 3))
    ax_trend.plot(r_range, batch_pred_real, color='blue', label='AI Prediction curve')
    
    # Mark user's current point
    ax_trend.scatter([radius], [predicted_stress], color='red', s=100, label='Current Design', zorder=5)
    
    ax_trend.set_xlabel("Hole Radius (mm)")
    ax_trend.set_ylabel("Max Stress (MPa)")
    ax_trend.grid(True, alpha=0.3)
    ax_trend.legend()
    
    st.pyplot(fig_trend)