import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. LOAD DATA
print("Loading FEA Data...")
try:
    df = pd.read_csv('data_log.txt')
    # Filter out any accidental error lines or empty rows
    df = df.dropna()
    print(f"Loaded {len(df)} simulation samples.")
except:
    print("Error: data_log.txt not found. Run manager.py first!")
    exit()

# Inputs: Radius, Load | Output: MaxStress
X = df[['Radius', 'Load']].values
y = df[['MaxStress']].values

# 2. PREPROCESS (Crucial for Neural Networks)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_scaled)
y_tensor = torch.FloatTensor(y_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2)

# 3. DEFINE THE SURROGATE MODEL
# A simple MLP that approximates the FEA solver
class FEA_Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),   # 2 Inputs (Radius, Load)
            nn.ReLU(),
            nn.Linear(64, 64),  # Hidden Layer
            nn.ReLU(),
            nn.Linear(64, 1)    # 1 Output (Stress)
        )
        
    def forward(self, x):
        return self.net(x)

model = FEA_Surrogate()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# 4. TRAINING LOOP
print("Training AI Solver...")
epochs = 1000
losses = []

for i in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    
    if i % 100 == 0:
        print(f"Epoch {i}: Loss {loss.item():.5f}")

# 5. EVALUATION (The "Recruiter Proof")
model.eval()
with torch.no_grad():
    test_pred_scaled = model(X_test)
    # Convert back to real units (MPa)
    test_pred_real = scaler_y.inverse_transform(test_pred_scaled.numpy())
    test_actual_real = scaler_y.inverse_transform(y_test.numpy())

# Plot: Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(test_actual_real, test_pred_real, alpha=0.7, color='blue')
plt.plot([test_actual_real.min(), test_actual_real.max()], 
         [test_actual_real.min(), test_actual_real.max()], 'r--', lw=2) # Ideal line
plt.xlabel("Actual Abaqus Stress (MPa)")
plt.ylabel("AI Predicted Stress (MPa)")
plt.title("AI Surrogate Accuracy: Real-Time Physics Prediction")
plt.grid(True)
plt.show()

# 6. SAVE MODEL (Optional)
torch.save(model.state_dict(), "fea_surrogate_model.pth")
print("Model Saved.")