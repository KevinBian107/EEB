import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from LCModels import LCNECortexFCN, LCNECortexRNN

def run_simulation(model, device, time_steps=100, condition="baseline"):
    '''Running simulations for all types of models'''
    LC_vals, NE_vals, C_vals = [], [], []

    if isinstance(model, LCNECortexFCN):
        LC_t = torch.tensor(0.1, device=device)  # Initial LC activity
        C_t = torch.tensor(0.1, device=device)  # Initial Cortex activity
        LC_hidden, Cortex_hidden = None, None  # Not used in FCN

    elif isinstance(model, LCNECortexRNN):
        LC_hidden = torch.zeros(1, model.hidden_size, device=device)  # RNN hidden state for LC
        Cortex_hidden = torch.zeros(1, model.hidden_size, device=device)  # RNN hidden state for Cortex
        LC_t = torch.zeros(1, model.hidden_size, device=device)  # Initial LC state
        C_t = torch.zeros(1, model.hidden_size, device=device)  # Initial Cortex state

    for t in range(time_steps):
        x_t = torch.randn(1, device=device) * 0.1  # Sensory stimulus

        # Define stressor conditions
        if condition == "acute_stress":
            stress_t = torch.tensor([1.0 if t == 50 else 0.0], device=device)
        elif condition == "chronic_stress":
            stress_t = torch.tensor([0.1 * torch.exp(torch.tensor(-0.02 * t, device=device))], device=device)
        elif condition == "top_down_control":
            stress_t = torch.tensor([0.5], device=device)
            if isinstance(model, LCNECortexRNN):  
                Cortex_hidden -= 0.05  # Cortical suppression effect
        else:
            stress_t = torch.tensor([0.0], device=device)

        # Forward pass based on model type
        if isinstance(model, LCNECortexFCN):
            LC_t, NE_t, C_t = model(x_t, stress_t, LC_t, C_t)

        elif isinstance(model, LCNECortexRNN):
            LC_t, NE_t, C_t, LC_hidden, Cortex_hidden = model(x_t, stress_t, LC_hidden, Cortex_hidden)

        # Store values
        LC_vals.append(LC_t.item())
        NE_vals.append(NE_t.item())
        C_vals.append(C_t.item())

    return LC_vals, NE_vals, C_vals

def render_plot(results):
    '''Rendering helper functions'''
    plt.figure(figsize=(12, 6))
    for cond, (LC_vals, NE_vals, C_vals) in results.items():
        plt.plot(LC_vals, label=f"LC Activity - {cond}")
    plt.xlabel("Time Steps")
    plt.ylabel("LC Activity")
    plt.legend()
    plt.title("Recurrent LC Activity Under Different Conditions")
    plt.show()

    plt.figure(figsize=(12, 6))
    for cond, (LC_vals, NE_vals, C_vals) in results.items():
        plt.plot(NE_vals, label=f"NE Release - {cond}")
    plt.xlabel("Time Steps")
    plt.ylabel("NE Release")
    plt.legend()
    plt.title("Recurrent NE Release Under Different Conditions")
    plt.show()

    plt.figure(figsize=(12, 6))
    for cond, (LC_vals, NE_vals, C_vals) in results.items():
        plt.plot(C_vals, label=f"Cortical Activity - {cond}")
    plt.xlabel("Time Steps")
    plt.ylabel("Cortical Activity")
    plt.legend()
    plt.title("Recurrent Cortical Activity Under Different Conditions")
    plt.show()