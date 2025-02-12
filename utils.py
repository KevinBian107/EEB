import pandas as pd
import numpy as np

import torch.optim as optim
import torch.nn as nn
import torch

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from models.LCModels import LCNECortexFCN, LCNECortexRNN

def run_simulation(model, device, time_steps=100, condition="baseline"):
    '''Running simulations for all types of untrained models and then graph'''
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

        LC_vals.append(LC_t.item())
        NE_vals.append(NE_t.item())
        C_vals.append(C_t.item())
        
    results = LC_vals, NE_vals, C_vals
    plt.figure(figsize=(12, 6))
    for cond, (LC_vals, NE_vals, C_vals) in results.items():
        plt.plot(LC_vals, label=f"LC Activity - {cond}")
        plt.xlabel("Time Steps")
        plt.ylabel("LC Activity")
        plt.legend()
        plt.title("LC Activity Under Different Conditions")
        plt.show()

        plt.figure(figsize=(12, 6))
        for cond, (LC_vals, NE_vals, C_vals) in results.items():
            plt.plot(NE_vals, label=f"NE Release - {cond}")
        plt.xlabel("Time Steps")
        plt.ylabel("NE Release")
        plt.legend()
        plt.title("NE Release Under Different Conditions")
        plt.show()

        plt.figure(figsize=(12, 6))
        for cond, (LC_vals, NE_vals, C_vals) in results.items():
            plt.plot(C_vals, label=f"Cortical Activity - {cond}")
        plt.xlabel("Time Steps")
        plt.ylabel("Cortical Activity")
        plt.legend()
        plt.title("Cortical Activity Under Different Conditions")
        plt.show()


def perform_pca_and_plot(activations, title, hue_labels):
    '''do batch pca plotting'''
    
    pca = PCA(n_components=2)
    act_pca = pca.fit_transform(activations)
    explained_variance = pca.explained_variance_ratio_ * 100

    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=hue_labels, palette="viridis", alpha=0.7)
    plt.title(f"{title}\nExplained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
    plt.xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
    plt.ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()