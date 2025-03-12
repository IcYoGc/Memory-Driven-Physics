import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Ensure the output folder exists
output_folder = "Experiments/open"
os.makedirs(output_folder, exist_ok=True)

# --- Conductivity Simulation ---
def conductivity_decay(t, tau, noise=0.0):
    """Model conductivity stabilization as an exponential decay process with optional noise."""
    decay = np.exp(-t / tau) + np.random.normal(0, noise, len(t))
    return np.clip(decay, 0, 1)

t_values = np.linspace(0, 100, 1000)
control_stabilization = conductivity_decay(t_values, 20, noise=0.02)
memory_stabilization = conductivity_decay(t_values, 50, noise=0.01)
entropy_stabilization = conductivity_decay(t_values, 10, noise=0.05)

results = {
    "time": list(t_values),
    "control": list(control_stabilization),
    "memory_field": list(memory_stabilization),
    "entropy": list(entropy_stabilization)
}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"{output_folder}/conductivity_results_{timestamp}.json"
with open(output_file, "w") as f:
    json.dump(results, f)
print(f"✅ Conductivity results saved to {output_file}")

# --- Entanglement Simulation ---
def entanglement_decay(t, tau):
    """Model entanglement decay as an exponential function."""
    return np.exp(-t / tau)

control_decay = entanglement_decay(t_values, 20)
memory_decay = entanglement_decay(t_values, 50)
entropy_decay = entanglement_decay(t_values, 10)

results_entanglement = {
    "time": list(t_values),
    "control": list(control_decay),
    "memory_field": list(memory_decay),
    "entropy": list(entropy_decay)
}
output_file_entanglement = f"{output_folder}/entanglement_results_{timestamp}.json"
with open(output_file_entanglement, "w") as f:
    json.dump(results_entanglement, f)
print(f"✅ Entanglement results saved to {output_file_entanglement}")

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.plot(t_values, control_stabilization, label="Control (Standard Stabilization)", linestyle='dashed', color='blue')
plt.plot(t_values, memory_stabilization, label="Memory Field (Enhanced Stabilization)", linestyle='solid', color='green')
plt.plot(t_values, entropy_stabilization, label="High-Entropy (Slower Stabilization)", linestyle='dotted', color='red')
plt.xlabel("Time (arbitrary units)")
plt.ylabel("Relative Conductivity")
plt.title("Conductivity Decay under Memory Field Influence")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t_values, control_decay, label="Control (Standard Decoherence)", linestyle='dashed', color='blue')
plt.plot(t_values, memory_decay, label="Memory Field (Extended Coherence)", linestyle='solid', color='green')
plt.plot(t_values, entropy_decay, label="High-Entropy (Accelerated Decoherence)", linestyle='dotted', color='red')
plt.xlabel("Time (arbitrary units)")
plt.ylabel("Entanglement Fidelity")
plt.title("Memory-Driven Entanglement Stability Simulation")
plt.legend()
plt.grid()
plt.show()
