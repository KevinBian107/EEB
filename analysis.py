import pandas as pd
import numpy as np
import torch

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns


def pca_gadget(model, X_tensor, df_clean):
    '''PCA analysis for LSTM Gadget model'''
    model.eval()
    with torch.no_grad():
        X_test = X_tensor.unsqueeze(1)  # Ensure (batch, seq_len, features)
        
        Pupil_pred, LC_act, NE_act, forget_gate, input_gate, output_gate = model(X_test)

    act_lc = LC_act.cpu().numpy()
    act_ne = NE_act.cpu().numpy()
    forget_gate_np = forget_gate.cpu().numpy()
    input_gate_np = input_gate.cpu().numpy()
    output_gate_np = output_gate.cpu().numpy()
    pupil_pred = Pupil_pred.cpu().numpy().squeeze()
    pupil_actual = df_clean["Event_PupilDilation"].values

    min_length = min(len(pupil_actual), len(pupil_pred))
    pupil_actual = pupil_actual[:min_length]
    pupil_pred = pupil_pred[:min_length]

    df_activations = pd.DataFrame({
        'LC_Mean': act_lc.mean(axis=1), 'LC_Var': act_lc.var(axis=1),
        'NE_Mean': act_ne.mean(axis=1), 'NE_Var': act_ne.var(axis=1),
        'ForgetGate_Mean': forget_gate_np.mean(axis=1), 'ForgetGate_Var': forget_gate_np.var(axis=1),
        'InputGate_Mean': input_gate_np.mean(axis=1), 'InputGate_Var': input_gate_np.var(axis=1),
        'OutputGate_Mean': output_gate_np.mean(axis=1), 'OutputGate_Var': output_gate_np.var(axis=1),
        'PupilPred': pupil_pred, 'ActualPupil': pupil_actual
    })

    activations_list = [act_lc, act_ne, forget_gate_np, input_gate_np, output_gate_np]
    labels = ["LC", "NE", "Forget Gate", "Input Gate", "Output Gate"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for i, (activation, label) in enumerate(zip(activations_list, labels)):
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(activation)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        ax = axes[i]
        sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=ax)
        ax.set_title(f"{label} Activations\nPC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
        ax.set_xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
        ax.set_ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")

    plt.tight_layout()
    plt.show()

    corr_lc = pearsonr(df_activations['LC_Mean'], df_activations['ActualPupil'])[0]
    corr_ne = pearsonr(df_activations['NE_Mean'], df_activations['ActualPupil'])[0]
    corr_forget = pearsonr(df_activations['ForgetGate_Mean'], df_activations['ActualPupil'])[0]
    corr_input = pearsonr(df_activations['InputGate_Mean'], df_activations['ActualPupil'])[0]
    corr_output = pearsonr(df_activations['OutputGate_Mean'], df_activations['ActualPupil'])[0]
    corr_pupil = pearsonr(df_activations['PupilPred'], df_activations['ActualPupil'])[0]

    print("Pearson Correlation with Actual Pupil Dilation:")
    print(f"LC Activation: {corr_lc:.3f}")
    print(f"NE Activation: {corr_ne:.3f}")
    print(f"Forget Gate: {corr_forget:.3f}")
    print(f"Input Gate: {corr_input:.3f}")
    print(f"Output Gate: {corr_output:.3f}")
    print(f"Predicted Pupil Dilation: {corr_pupil:.3f}")
    

def pca_lcne_lstm(model, X_tensor, df_clean):
    '''PCA analysis for LCNE LSTM model'''
    model.eval()
    with torch.no_grad():
        prev_LC = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        prev_Cortex = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        cell_state = torch.zeros(X_tensor.shape[0], model.hidden_dim)

        LC_act, NE_act, C_act, Pupil_pred, forget_gate, input_gate, output_gate, cell_state = model(
            X_tensor, prev_LC, prev_Cortex, cell_state, return_activations=True
        )

    act_lc = LC_act.cpu().numpy()
    act_ne = NE_act.cpu().numpy()
    act_cortex = C_act.cpu().numpy()
    pupil_pred = Pupil_pred.cpu().numpy().squeeze()
    forget_gate_np = forget_gate.cpu().numpy()
    input_gate_np = input_gate.cpu().numpy()
    output_gate_np = output_gate.cpu().numpy()
    cell_state_np = cell_state.cpu().numpy()

    df_activations = pd.DataFrame({
        'LC_Mean': act_lc.mean(axis=1), 'LC_Var': act_lc.var(axis=1),
        'NE_Mean': act_ne.mean(axis=1), 'NE_Var': act_ne.var(axis=1),
        'Cortex_Mean': act_cortex.mean(axis=1), 'Cortex_Var': act_cortex.var(axis=1),
        'ForgetGate_Mean': forget_gate_np.mean(axis=1), 'ForgetGate_Var': forget_gate_np.var(axis=1),
        'InputGate_Mean': input_gate_np.mean(axis=1), 'InputGate_Var': input_gate_np.var(axis=1),
        'OutputGate_Mean': output_gate_np.mean(axis=1), 'OutputGate_Var': output_gate_np.var(axis=1),
        'CellState_Mean': cell_state_np.mean(axis=1), 'CellState_Var': cell_state_np.var(axis=1),
        'PupilPred': pupil_pred.mean(axis=1),
    })
    
    print (act_lc.mean(axis=1).shape, input_gate_np.mean(axis=1).shape, pupil_pred.shape)
    
    activations_list = [act_lc, act_ne, act_cortex, input_gate_np, output_gate_np, cell_state_np]
    labels = ["LC", "NE", "Cortex", "Input Gate", "Output Gate", "Cell State"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    axes = axes.flatten()

    for i, (activation, label) in enumerate(zip(activations_list, labels)):
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(activation)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        ax = axes[i]
        sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=ax)
        ax.set_title(f"{label}\nExplained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
        ax.set_xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
        ax.set_ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")
        ax.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()
    


def pca_feed_forward(model, X_tensor, df_behavior):
    """
    Extract ff_nn activations, apply PCA and t-SNE, and visualize them side by side.
    """
    with torch.no_grad():
        predictions, act1, act2 = model(X_tensor, return_activations=True)

    act1_np = act1.cpu().numpy()
    act2_np = act2.cpu().numpy()
    df_filtered = df_behavior.iloc[:X_tensor.shape[0]].copy()

    pca1 = PCA(n_components=2)
    act1_pca = pca1.fit_transform(act1_np)
    explained_variance1 = pca1.explained_variance_ratio_ * 100

    pca2 = PCA(n_components=2)
    act2_pca = pca2.fit_transform(act2_np)
    explained_variance2 = pca2.explained_variance_ratio_ * 100

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    act1_tsne = tsne.fit_transform(act1_np)
    act2_tsne = tsne.fit_transform(act2_np)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PCA Layer 1
    sns.scatterplot(x=act1_pca[:, 0], y=act1_pca[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[0, 0])
    axes[0, 0].set_title(f"PCA Projection of Layer 1 Activations\nExplained Variance: PC1={explained_variance1[0]:.2f}%, PC2={explained_variance1[1]:.2f}%")
    axes[0, 0].set_xlabel(f"PCA Component 1 ({explained_variance1[0]:.2f}% Variance)")
    axes[0, 0].set_ylabel(f"PCA Component 2 ({explained_variance1[1]:.2f}% Variance)")

    # PCA Layer 2
    sns.scatterplot(x=act2_pca[:, 0], y=act2_pca[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[0, 1])
    axes[0, 1].set_title(f"PCA Projection of Layer 2 Activations\nExplained Variance: PC1={explained_variance2[0]:.2f}%, PC2={explained_variance2[1]:.2f}%")
    axes[0, 1].set_xlabel(f"PCA Component 1 ({explained_variance2[0]:.2f}% Variance)")
    axes[0, 1].set_ylabel(f"PCA Component 2 ({explained_variance2[1]:.2f}% Variance)")

    # t-SNE Layer 1
    sns.scatterplot(x=act1_tsne[:, 0], y=act1_tsne[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[1, 0])
    axes[1, 0].set_title("t-SNE Projection of Layer 1 Activations")
    axes[1, 0].set_xlabel("t-SNE Component 1")
    axes[1, 0].set_ylabel("t-SNE Component 2")

    # t-SNE Layer 2
    sns.scatterplot(x=act2_tsne[:, 0], y=act2_tsne[:, 1], hue=df_filtered["Condition"],
                    palette="viridis", alpha=0.7, ax=axes[1, 1])
    axes[1, 1].set_title("t-SNE Projection of Layer 2 Activations")
    axes[1, 1].set_xlabel("t-SNE Component 1")
    axes[1, 1].set_ylabel("t-SNE Component 2")

    plt.tight_layout()
    plt.show()



def pca_lcne(model, X_tensor, df_clean):
    """
    Extracts activations, performs PCA and t-SNE, applies clustering, and visualizes results.
    """

    # Run model inference and extract activations
    with torch.no_grad():
        prev_LC = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        prev_Cortex = torch.zeros(X_tensor.shape[0], model.hidden_dim)
        
        LC_act, NE_act, C_act, Pupil_pred, LC_raw, NE_raw, C_raw = model(X_tensor, prev_LC, prev_Cortex, return_activations=True)

    # Convert activations to numpy arrays
    act_lc = LC_act.cpu().numpy()
    act_ne = NE_act.cpu().numpy()
    act_cortex = C_act.cpu().numpy()

    # Create DataFrame for analysis
    df_activations = pd.DataFrame({
        'LC_Mean': act_lc.mean(axis=1),
        'NE_Mean': act_ne.mean(axis=1),
        'Cortex_Mean': act_cortex.mean(axis=1),
        'PupilPred': Pupil_pred.cpu().numpy().squeeze(),
        'ActualPupil': df_clean['Event_PupilDilation'].values  
    })

    # Concatenate activations for combined analysis
    act_combined = np.hstack([act_lc, act_ne, act_cortex])
    
    # Standardize data
    scaler = StandardScaler()
    act_combined_scaled = scaler.fit_transform(act_combined)

    # Perform PCA
    pca = PCA(n_components=2)
    act_pca = pca.fit_transform(act_combined_scaled)
    explained_variance = pca.explained_variance_ratio_ * 100

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    act_tsne = tsne.fit_transform(act_pca)

    # Apply K-Means clustering
    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters_pca = kmeans.fit_predict(act_pca)
    clusters_tsne = kmeans.predict(act_tsne)

    # Plot PCA and t-SNE projections
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    activations_list = [act_lc, act_ne, act_cortex]
    labels = ["LC", "NE", "Cortex"]

    for i, (activation, label) in enumerate(zip(activations_list, labels)):
        pca = PCA(n_components=2)
        act_pca = pca.fit_transform(activation)
        explained_variance = pca.explained_variance_ratio_ * 100
        
        # PCA Projection
        sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=df_clean["Condition"], palette="viridis", alpha=0.7, ax=axes[i])
        axes[i].set_title(f"{label} PCA\nExplained Variance: PC1={explained_variance[0]:.2f}%, PC2={explained_variance[1]:.2f}%")
        axes[i].set_xlabel(f"PCA Component 1 ({explained_variance[0]:.2f}% Variance)")
        axes[i].set_ylabel(f"PCA Component 2 ({explained_variance[1]:.2f}% Variance)")

    # K-Means Clustering
    sns.scatterplot(x=act_pca[:, 0], y=act_pca[:, 1], hue=clusters_pca, palette="tab10", alpha=0.7, ax=axes[3])
    axes[3].set_title("PCA Clustering")

    sns.scatterplot(x=act_tsne[:, 0], y=act_tsne[:, 1], hue=clusters_tsne, palette="tab10", alpha=0.7, ax=axes[4])
    axes[4].set_title("t-SNE Clustering")

    plt.tight_layout()
    plt.show()

def firing_lcne(model, X_test, df_clean):
    '''Firing rate analysis for LCNE model'''

    with torch.no_grad():
        LC_activations, NE_activations, C_activations, Pupil_pred = model(
            X_test, torch.zeros(X_test.shape[0], 8), torch.zeros(X_test.shape[0], 8)
        )

    act_lc = LC_activations.cpu().numpy().squeeze()
    act_ne = NE_activations.cpu().numpy().squeeze()
    act_cortex = C_activations.cpu().numpy().squeeze()
    pupil_pred = Pupil_pred.cpu().numpy().squeeze()

    pupil_actual = df_clean["Event_PupilDilation"].values
    time_axis = np.arange(len(pupil_actual))  # Time index

    scaler_pupil = MinMaxScaler(feature_range=(0, 1))
    pupil_actual_scaled = scaler_pupil.fit_transform(pupil_actual.reshape(-1, 1)).squeeze()
    pupil_pred_scaled = scaler_pupil.transform(pupil_pred.reshape(-1, 1)).squeeze()

    scaler_lc = MinMaxScaler(feature_range=(0, 1))
    act_lc_scaled = scaler_lc.fit_transform(act_lc.reshape(-1, 1)).squeeze()

    scaler_ne = MinMaxScaler(feature_range=(0, 1))
    act_ne_scaled = scaler_ne.fit_transform(act_ne.reshape(-1, 1)).squeeze()

    scaler_cortex = MinMaxScaler(feature_range=(0, 1))
    act_cortex_scaled = scaler_cortex.fit_transform(act_cortex.reshape(-1, 1)).squeeze()

    # Ensure all variables have the same length
    min_length = min(len(time_axis), len(pupil_actual_scaled), len(pupil_pred_scaled), 
                    len(act_lc_scaled), len(act_ne_scaled), len(act_cortex_scaled))

    time_axis = time_axis[:min_length]
    pupil_actual_scaled = pupil_actual_scaled[:min_length]
    pupil_pred_scaled = pupil_pred_scaled[:min_length]
    act_lc_scaled = act_lc_scaled[:min_length]
    act_ne_scaled = act_ne_scaled[:min_length]
    act_cortex_scaled = act_cortex_scaled[:min_length]


    plt.figure(figsize=(12, 6))

    sns.lineplot(x=time_axis, y=pupil_actual_scaled, label="Actual Pupil Dilation", color='blue', linestyle="dashed", alpha=0.8)
    sns.lineplot(x=time_axis, y=act_lc_scaled, label="LC Activation", color='green', alpha=0.5)
    # sns.lineplot(x=time_axis, y=act_ne_scaled, label="NE Activation", color='purple', alpha=0.5)
    # sns.lineplot(x=time_axis, y=act_cortex_scaled, label="Cortex Activation", color='orange', alpha=0.5)

    plt.xlabel("Time (Trials)")
    plt.ylabel("Normalized Activation")
    plt.title("Model Activations vs. Real Pupil Dilation Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    lc_corr = pearsonr(act_lc_scaled, pupil_actual_scaled)[0]
    ne_corr = pearsonr(act_ne_scaled, pupil_actual_scaled)[0]
    cortex_corr = pearsonr(act_cortex_scaled, pupil_actual_scaled)[0]
    pupil_pred_corr = pearsonr(pupil_pred_scaled, pupil_actual_scaled)[0]
    print(f"Correlation with Actual Pupil Dilation:")
    print(f"LC Activation: {lc_corr:.3f}")
    print(f"NE Activation: {ne_corr:.3f}")
    print(f"Cortex Activation: {cortex_corr:.3f}")
    print(f"Predicted Pupil Dilation: {pupil_pred_corr:.3f}")