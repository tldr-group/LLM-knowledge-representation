import os
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from math import acos, degrees

from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

import json  # Added for saving and loading data

# ---------------------------- Configuration ---------------------------- #

@dataclass
class ModelConfig:
    """
    Configuration for a machine learning model.
    """
    name: str
    activation_path_template: str  # Use {layer} as placeholder for layer index
    num_layers: int

# Example configurations for different models and features
MODEL_CONFIGS: Dict[str, List[ModelConfig]] = {
    "Atomic Number": [
        ModelConfig(
            name='Meta-Llama-3.1-70B',
            activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number/atomic number.last.11_templates.{layer}.pt',
            num_layers=
            80  
        ),
    ],
    "Atomic Mass": [
        ModelConfig(
            name='Meta-Llama-3.1-70B',
            activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic mass/atomic mass.last.11_templates.{layer}.pt',
            num_layers=80  
        ),
        # Add more models if needed
    ],
    "Group": [
        ModelConfig(
            name='Meta-Llama-3.1-70B',
            activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group/group.last.11_templates.{layer}.pt',
            num_layers=80  
        ),
        # Add more models if needed
    ],
    "Period": [
        ModelConfig(
            name='Meta-Llama-3.1-70B',
            activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/period/period.last.11_templates.{layer}.pt',
            num_layers=80  
        ),
        # Add more models if needed
    ],
    "Electronegativity": [
        ModelConfig(
            name='Meta-Llama-3.1-70B',
            activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/electronegativity/electronegativity.last.11_templates.{layer}.pt',
            num_layers=80  
        ),

    ]
}

# Global Constants
PROMPT_TEMPLATE_NUMBER = 11
OUTPUT_DIR = 'Results/cosine_distance'  # Updated output directory name

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'angles'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'pearson_correlations'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'angle_differences'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'angles_pca'), exist_ok=True)  # New directory for PCA angles
os.makedirs(os.path.join(OUTPUT_DIR, 'pearson_correlations_pca'), exist_ok=True)  # New directory for PCA Pearson
os.makedirs(os.path.join(OUTPUT_DIR, 'angle_differences_pca'), exist_ok=True)  # New directory for PCA angle differences
os.makedirs(os.path.join(OUTPUT_DIR, 'data'), exist_ok=True)  # Directory to store computed data

# ---------------------------- Data Handling ---------------------------- #

def load_data(file_path: str, label_columns: List[str], repeat_factor: int = PROMPT_TEMPLATE_NUMBER) -> Dict[str, np.ndarray]:
    """
    Loads the dataset and returns the labels with missing values filled with -inf.

    Args:
        file_path (str): Path to the dataset CSV file.
        label_columns (List[str]): List of columns to use as regression labels.
        repeat_factor (int): Number of times to repeat each label.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping each label to its repeated values.
    """
    periodic_table = pd.read_csv(file_path)
    print(f"Loaded dataset with columns: {periodic_table.columns.tolist()}")

    labels_repeated_dict = {}
    for label in label_columns:
        labels = periodic_table[label].fillna(-np.inf).astype(float).values
        labels_repeated = np.repeat(labels, repeat_factor)
        labels_repeated_dict[label] = labels_repeated

    return labels_repeated_dict

def split_data_middle_group(labels_repeated: np.ndarray, group_size: int = PROMPT_TEMPLATE_NUMBER) -> Tuple[List[int], List[int]]:
    """
    Splits data by selecting the middle group for testing.

    Args:
        labels_repeated (np.ndarray): Array of repeated labels.
        group_size (int): Number of samples per group.

    Returns:
        Tuple[List[int], List[int]]: Training and testing indices.
    """
    train_indices = []
    test_indices = []

    for label in np.unique(labels_repeated):
        if label == -np.inf:
            continue  # Skip labels filled with -inf
        label_indices = np.where(labels_repeated == label)[0]
        n_groups = len(label_indices) // group_size
        if n_groups == 0:
            continue  # Not enough samples for grouping
        middle_group = n_groups // 2
        start_idx = middle_group * group_size
        end_idx = start_idx + group_size

        test_indices.extend(label_indices[start_idx:end_idx])
        train_indices.extend(np.delete(label_indices, np.arange(start_idx, end_idx)))

    return train_indices, test_indices

def split_data_first_group(labels_repeated: np.ndarray, group_size: int = PROMPT_TEMPLATE_NUMBER) -> Tuple[List[int], List[int]]:
    """
    Splits data by selecting the first group for testing.

    Args:
        labels_repeated (np.ndarray): Array of repeated labels.
        group_size (int): Number of samples per group.

    Returns:
        Tuple[List[int], List[int]]: Training and testing indices.
    """
    train_indices = []
    test_indices = []

    for label in np.unique(labels_repeated):
        if label == -np.inf:
            continue  # Skip labels filled with -inf
        label_indices = np.where(labels_repeated == label)[0]
        test_indices.extend(label_indices[:group_size])  # First group as test set
        train_indices.extend(label_indices[group_size:])  # Remaining as training set

    return train_indices, test_indices

def split_data_group_shuffle(labels_repeated: np.ndarray, group_size: int = PROMPT_TEMPLATE_NUMBER, test_size: float = 0.2, random_state: int = 100) -> Tuple[List[int], List[int]]:
    """
    Splits data using group shuffle split.

    Args:
        labels_repeated (np.ndarray): Array of repeated labels.
        group_size (int): Number of samples per group.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed used by the random number generator.

    Returns:
        Tuple[List[int], List[int]]: Training and testing indices.
    """
    valid_indices = np.isfinite(labels_repeated)
    valid_labels = labels_repeated[valid_indices]

    groups = np.repeat(np.arange(len(valid_labels) // group_size), group_size)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.arange(len(valid_labels)), groups=groups))

    train_indices = np.where(valid_indices)[0][train_idx]
    test_indices = np.where(valid_indices)[0][test_idx]

    return train_indices.tolist(), test_indices.tolist()

def split_data(labels_repeated: np.ndarray, method: str) -> Tuple[List[int], List[int]]:
    """
    Splits data based on the specified method.

    Args:
        labels_repeated (np.ndarray): Array of repeated labels.
        method (str): Splitting method ('middle', 'first', 'group_shuffle').

    Returns:
        Tuple[List[int], List[int]]: Training and testing indices.
    """
    if method == 'middle':
        return split_data_middle_group(labels_repeated)
    elif method == 'first':
        return split_data_first_group(labels_repeated)
    elif method == 'group_shuffle':
        return split_data_group_shuffle(labels_repeated)
    else:
        raise ValueError(f"Unknown split method: {method}")

def load_activation_data(layer: int, activation_path_template: str, labels_repeated: np.ndarray, split_method: str = 'middle') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads activation data for a specific layer and splits it into training and testing sets.

    Args:
        layer (int): Layer index.
        activation_path_template (str): Template path for activation files.
        labels_repeated (np.ndarray): Array of repeated labels.
        split_method (str): Method to split data.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    file_path = activation_path_template.format(layer=layer)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")

    activation_data = torch.load(file_path, weights_only=True).cpu().numpy()

    if activation_data.shape[0] != len(labels_repeated):
        raise ValueError(f"Inconsistent number of samples: {activation_data.shape[0]} features, {len(labels_repeated)} labels.")

    train_indices, test_indices = split_data(labels_repeated, split_method)

    X_train, X_test = activation_data[train_indices], activation_data[test_indices]
    y_train, y_test = labels_repeated[train_indices], labels_repeated[test_indices]

    return X_train, X_test, y_train, y_test

# ---------------------------- Model Training and Evaluation ---------------------------- #

def train_svr_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, repeat_factor: int = PROMPT_TEMPLATE_NUMBER) -> Tuple[float, np.ndarray, List[np.ndarray]]:
    """
    Trains an SVR model using GroupKFold cross-validation and returns average R² and predictions.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        n_splits (int): Number of splits for cross-validation.
        repeat_factor (int): Number of samples per group.

    Returns:
        Tuple[float, np.ndarray, List[np.ndarray]]: Average R² score, all predictions, and list of coefficients.
    """
    groups = np.repeat(np.arange(len(y) // repeat_factor), repeat_factor)

    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []
    y_pred_all = np.zeros_like(y)
    coef_list = []

    for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        sample_weights = compute_sample_weight('balanced', y_train)
        svr_model = SVR(kernel='linear', C=2)
        svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        y_pred = svr_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

        y_pred_all[test_index] = y_pred

        # Extract coefficients and store
        coef = svr_model.coef_
        coef_list.append(coef)

        print(f"Fold {fold}: R² = {r2:.4f}")

    avg_r2 = np.mean(r2_scores)
    print(f"Average R² across {n_splits} folds: {avg_r2:.4f}")
    return avg_r2, y_pred_all, coef_list

# ---------------------------- Cosine Distance Calculation ---------------------------- #

def calculate_cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine distance between two vectors.

    Args:
        vec1 (np.ndarray): First vector.
        vec2 (np.ndarray): Second vector.

    Returns:
        float: Cosine distance between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # Prevent division by zero
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the vectors is zero-length.")
    cosine_similarity = dot_product / (norm1 * norm2)
    # Clamp cosine_similarity to the valid range [-1, 1] to avoid numerical issues
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

# ---------------------------- Plotting Functions ---------------------------- #

def plot_cosine_distance_results(angle_dict: Dict[str, float], label_columns: List[str]):
    """
    Plots the cosine distances between decision boundaries for different models.

    Args:
        angle_dict (Dict[str, float]): Dictionary with model-label pairs as keys and cosine distances as values.
        label_columns (List[str]): List of label columns.
    """
    if not angle_dict:
        print("No cosine distances to plot. Please check if the angle_dict is empty.")
        return

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")

    models = list(angle_dict.keys())
    distances = list(angle_dict.values())

    # Ensure there is data to plot
    if not models or not distances:
        print("No valid cosine distances were found for plotting.")
        return

    sns.barplot(x=models, y=distances, palette="viridis")
    plt.xlabel('Model - Feature Pair', fontsize=14)
    plt.ylabel('Cosine Distance Between Decision Boundaries', fontsize=14)
    plt.title('Cosine Distance Between SVR Decision Boundaries for Different Feature Pairs', fontsize=16)
    plt.ylim(0, 2)  # Cosine distance ranges from 0 to 2

    for index, value in enumerate(distances):
        plt.text(index, value + 0.02, f"{value:.4f}", ha='center', va='bottom', fontsize=12)

    # Add shadow area between 0.9486 to 1.0514
    plt.axhspan(0.9486, 1.0514, color='grey', alpha=0.3, label='Non-Meaningful Area')

    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'angles', 'decision_boundary_cosine_distances.png'), dpi=300)
    plt.show()

def plot_feature_pair_cosine_distances(cosine_distance_dict: Dict[str, List[float]], num_layers: int, model_name: str, output_dir: str = 'results_cosine_distance/angles'):
    """
    Plots the cosine distances between feature pairs across layers and saves the figure.

    Args:
        cosine_distance_dict (Dict[str, List[float]]): Dictionary with feature pairs as keys and cosine distances as values across layers.
        num_layers (int): Number of layers.
        model_name (str): Name of the model to include in the filename.
        output_dir (str): Directory to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    layers = np.arange(num_layers)  # Layers from 0 to num_layers - 1

    for feature_pair, distances in cosine_distance_dict.items():
        plt.plot(layers, distances, marker='o', label=feature_pair)

    # Add a shaded area between 0.9486 to 1.0514
    plt.axhspan(0.9486, 1.0514, color='grey', alpha=0.3, label='Non-Meaningful Area')

    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Distance Between Decision Boundaries')
    plt.title(f'Cosine Distance Between SVR Decision Boundaries for Feature Pairs Across Layers - {model_name}')
    plt.ylim(0, 2)  # Cosine distance ranges from 0 to 2
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Define the path to save the plot with the model name
    output_path = os.path.join(output_dir, f'feature_pair_cosine_distances_across_layers_{model_name}.png')
    plt.savefig(output_path)
    plt.show()

    print(f"Cosine distance plot saved at {output_path}")

def plot_feature_pair_pearson(pearson_dict: Dict[str, List[float]], num_layers: int, model_name: str, output_dir: str = 'results_cosine_distance/pearson_correlations'):
    """
    Plots the Pearson correlations between feature pairs across layers and saves the figure.

    Args:
        pearson_dict (Dict[str, List[float]]): Dictionary with feature pairs as keys and Pearson correlations as values across layers.
        num_layers (int): Number of layers.
        model_name (str): Name of the model to include in the filename.
        output_dir (str): Directory to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    layers = np.arange(num_layers)  # Layers from 0 to num_layers - 1

    for feature_pair, corrs in pearson_dict.items():
        plt.plot(layers, corrs, marker='o', label=feature_pair)

    # Add horizontal lines at correlation = 0 and 1 for reference
    plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, label='0 Correlation')
    plt.axhline(y=1, color='green', linestyle='--', linewidth=1, label='1 Correlation')

    plt.xlabel('Layer Index')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.title(f'Pearson Correlation Between SVR Coefficients for Feature Pairs Across Layers - {model_name}')
    plt.ylim(-1.1, 1.1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Define the path to save the plot with the model name
    output_path = os.path.join(output_dir, f'feature_pair_pearson_across_layers_{model_name}.png')
    plt.savefig(output_path)
    plt.show()

    print(f"Pearson correlation plot saved at {output_path}")

def compute_feature_pair_cosine_distances(coefficients_dict: Dict[str, Dict[str, List[np.ndarray]]],
                                         feature_pairs: List[Tuple[str, str]],
                                         label_model_mapping: Dict[str, List[ModelConfig]],
                                         num_layers: int) -> Dict[str, List[float]]:
    """
    Computes cosine distances between feature pairs across all layers, adapting to different models for each feature.

    Args:
        coefficients_dict (Dict[str, Dict[str, List[np.ndarray]]]): Coefficients organized by label and model.
        feature_pairs (List[Tuple[str, str]]): List of feature pairs to compare.
        label_model_mapping (Dict[str, List[ModelConfig]]): Mapping from labels to model configurations.
        num_layers (int): Number of layers.

    Returns:
        Dict[str, List[float]]: Dictionary where keys are feature pairs and values are lists of cosine distances across layers.
    """
    cosine_distance_dict = {}

    for feature1, feature2 in feature_pairs:
        key = f"{feature1} vs {feature2}"
        cosine_distance_dict[key] = []

        for layer in range(num_layers):
            # Retrieve the first model for each feature (assuming one model per feature for simplicity)
            model1 = label_model_mapping[feature1][0].name
            model2 = label_model_mapping[feature2][0].name

            coef1 = coefficients_dict.get(feature1, {}).get(model1, [None] * num_layers)[layer]
            coef2 = coefficients_dict.get(feature2, {}).get(model2, [None] * num_layers)[layer]

            if coef1 is not None and coef2 is not None:
                try:
                    distance = calculate_cosine_distance(coef1, coef2)
                except ValueError:
                    distance = np.nan
                cosine_distance_dict[key].append(distance)
            else:
                cosine_distance_dict[key].append(np.nan)  # Handle missing data gracefully

    return cosine_distance_dict

def compute_feature_pair_pearson(coefficients_dict: Dict[str, Dict[str, List[np.ndarray]]],
                                 feature_pairs: List[Tuple[str, str]],
                                 label_model_mapping: Dict[str, List[ModelConfig]],
                                 num_layers: int) -> Dict[str, List[float]]:
    """
    Computes Pearson correlation coefficients between feature pairs across all layers.

    Args:
        coefficients_dict (Dict[str, Dict[str, List[np.ndarray]]]): Coefficients organized by label and model.
        feature_pairs (List[Tuple[str, str]]): List of feature pairs to compare.
        label_model_mapping (Dict[str, List[ModelConfig]]): Mapping from labels to model configurations.
        num_layers (int): Number of layers.

    Returns:
        Dict[str, List[float]]: Dictionary where keys are feature pairs and values are lists of Pearson correlations across layers.
    """
    pearson_dict = {}

    for feature1, feature2 in feature_pairs:
        key = f"{feature1} vs {feature2}"
        pearson_dict[key] = []

        for layer in range(num_layers):
            # Retrieve the first model for each feature (assuming one model per feature for simplicity)
            model1 = label_model_mapping[feature1][0].name
            model2 = label_model_mapping[feature2][0].name

            coef1 = coefficients_dict.get(feature1, {}).get(model1, [None] * num_layers)[layer]
            coef2 = coefficients_dict.get(feature2, {}).get(model2, [None] * num_layers)[layer]

            if coef1 is not None and coef2 is not None:
                try:
                    corr, _ = pearsonr(coef1, coef2)
                except ValueError:
                    corr = np.nan
                pearson_dict[key].append(corr)
            else:
                pearson_dict[key].append(np.nan)  # Handle missing data gracefully

    return pearson_dict

# ---------------------------- PCA-Based Cosine Distance Calculation ---------------------------- #

def compute_feature_pair_cosine_distances_pca(coefficients_dict: Dict[str, Dict[str, List[np.ndarray]]],
                                             feature_pairs: List[Tuple[str, str]],
                                             label_model_mapping: Dict[str, List[ModelConfig]],
                                             num_layers: int,
                                             n_components: int = 10) -> Dict[str, List[float]]:
    """
    Computes cosine distances between feature pairs across all layers using PCA-transformed coefficients.

    Args:
        coefficients_dict (Dict[str, Dict[str, List[np.ndarray]]]): Coefficients organized by label and model.
        feature_pairs (List[Tuple[str, str]]): List of feature pairs to compare.
        label_model_mapping (Dict[str, List[ModelConfig]]): Mapping from labels to model configurations.
        num_layers (int): Number of layers.
        n_components (int): Number of principal components for PCA.

    Returns:
        Dict[str, List[float]]: Dictionary where keys are feature pairs and values are lists of cosine distances across layers.
    """
    cosine_distance_dict_pca = {}

    # Collect all coefficients to fit PCA
    all_coefs = []
    for label, models in coefficients_dict.items():
        for model_name, coefs in models.items():
            for coef in coefs:
                if coef is not None:
                    all_coefs.append(coef)

    all_coefs = np.array(all_coefs)
    print(f"Total coefficients for PCA (Cosine Distances): {all_coefs.shape}")

    # Fit PCA on all coefficients
    pca = PCA(n_components=n_components)
    pca.fit(all_coefs)
    print(f"PCA explained variance ratio for cosine distances: {pca.explained_variance_ratio_}")

    # Transform coefficients using PCA
    coefficients_pca = {}
    for label, models in coefficients_dict.items():
        coefficients_pca[label] = {}
        for model_name, coefs in models.items():
            coefficients_pca[label][model_name] = []
            for coef in coefs:
                if coef is not None:
                    transformed = pca.transform(coef.reshape(1, -1)).flatten()
                    coefficients_pca[label][model_name].append(transformed)
                else:
                    coefficients_pca[label][model_name].append(None)

    # Compute cosine distances using PCA-transformed coefficients
    for feature1, feature2 in feature_pairs:
        key = f"{feature1} vs {feature2}"
        cosine_distance_dict_pca[key] = []

        for layer in range(num_layers):
            # Retrieve the first model for each feature (assuming one model per feature for simplicity)
            model1 = label_model_mapping[feature1][0].name
            model2 = label_model_mapping[feature2][0].name

            coef1 = coefficients_pca.get(feature1, {}).get(model1, [None] * num_layers)[layer]
            coef2 = coefficients_pca.get(feature2, {}).get(model2, [None] * num_layers)[layer]

            if coef1 is not None and coef2 is not None:
                try:
                    distance = calculate_cosine_distance(coef1, coef2)
                except ValueError:
                    distance = np.nan
                cosine_distance_dict_pca[key].append(distance)
            else:
                cosine_distance_dict_pca[key].append(np.nan)  # Handle missing data gracefully

    return cosine_distance_dict_pca

# ---------------------------- Plotting Functions for PCA-Based Cosine Distances ---------------------------- #

def plot_feature_pair_cosine_distances_pca(cosine_distance_dict_pca: Dict[str, List[float]], num_layers: int, model_name: str, output_dir: str = 'results_cosine_distance/angles_pca'):
    """
    Plots the cosine distances between feature pairs across layers using PCA-transformed coefficients and saves the figure.

    Args:
        cosine_distance_dict_pca (Dict[str, List[float]]): Dictionary with feature pairs as keys and PCA-based cosine distances as values across layers.
        num_layers (int): Number of layers.
        model_name (str): Name of the model to include in the filename.
        output_dir (str): Directory to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    layers = np.arange(num_layers)  # Layers from 0 to num_layers - 1

    for feature_pair, distances in cosine_distance_dict_pca.items():
        plt.plot(layers, distances, marker='o', label=feature_pair)

    # Add a shaded area between 0.9486 to 1.0514
    plt.axhspan(0.9486, 1.0514, color='grey', alpha=0.3, label='Non-Meaningful Area')

    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Distance Between Decision Boundaries')
    plt.title(f'PCA-Based Cosine Distance Between SVR Decision Boundaries for Feature Pairs Across Layers - {model_name}')
    plt.ylim(0, 2)  # Cosine distance ranges from 0 to 2
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Define the path to save the plot with the model name
    output_path = os.path.join(output_dir, f'feature_pair_cosine_distances_pca_across_layers_{model_name}.png')
    plt.savefig(output_path)
    plt.show()

    print(f"PCA-based cosine distance plot saved at {output_path}")

# ---------------------------- Compute Angle/Cosine Distance Differences ---------------------------- #

def compute_feature_cosine_distance_differences(coefficients_dict: Dict[str, Dict[str, List[np.ndarray]]],
                                              label_model_mapping: Dict[str, List[ModelConfig]],
                                              num_layers: int) -> Dict[str, List[float]]:
    """
    Computes the cosine distance differences between each layer's coefficients and the last layer's coefficients for each feature.

    Args:
        coefficients_dict (Dict[str, Dict[str, List[np.ndarray]]]): Coefficients organized by label and model.
        label_model_mapping (Dict[str, List[ModelConfig]]): Mapping from labels to model configurations.
        num_layers (int): Number of layers.

    Returns:
        Dict[str, List[float]]: Dictionary where keys are feature names and values are lists of cosine distance differences across layers.
    """
    cosine_diff_dict = {}

    for feature, models in label_model_mapping.items():
        model_name = models[0].name  # Assuming one model per feature
        coefs = coefficients_dict.get(feature, {}).get(model_name, [])

        if not coefs:
            print(f"No coefficients found for feature '{feature}' with model '{model_name}'. Skipping.")
            continue

        last_layer_coef = coefs[-1]
        cosine_diff_dict[feature] = []

        for layer in range(num_layers):
            current_coef = coefs[layer]
            if current_coef is not None and last_layer_coef is not None:
                try:
                    distance = calculate_cosine_distance(current_coef, last_layer_coef)
                except ValueError:
                    distance = np.nan
                cosine_diff_dict[feature].append(distance)
            else:
                cosine_diff_dict[feature].append(np.nan)  # Handle missing data gracefully

    return cosine_diff_dict

def plot_feature_cosine_distance_differences(cosine_diff_dict: Dict[str, List[float]], num_layers: int, model_name: str, output_dir: str = 'results_cosine_distance/angle_differences'):
    """
    Plots the cosine distance differences between each layer's coefficients and the last layer's coefficients for each feature.

    Args:
        cosine_diff_dict (Dict[str, List[float]]): Dictionary with feature names as keys and cosine distance differences as values across layers.
        num_layers (int): Number of layers.
        model_name (str): Name of the model to include in the filename.
        output_dir (str): Directory to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    layers = np.arange(num_layers)  # Layers from 0 to num_layers - 1

    for feature, distance_diffs in cosine_diff_dict.items():
        plt.plot(layers, distance_diffs, marker='o', label=feature)

    # Add a shaded area between 0.9486 to 1.0514
    plt.axhspan(0.9486, 1.0364, color='grey', alpha=0.3, label='Non-Meaningful Area')

    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Distance Difference to Last Layer')
    plt.title(f'Cosine Distance Differences of SVR Coefficients per Feature Across Layers - {model_name}')
    plt.ylim(0, 2)  # Cosine distance ranges from 0 to 2
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Define the path to save the plot with the model name
    output_path = os.path.join(output_dir, f'feature_cosine_distance_differences_{model_name}.png')
    plt.savefig(output_path)
    plt.show()

    print(f"Cosine distance difference plot saved at {output_path}")

# ---------------------------- PCA-Based Cosine Distance Calculation ---------------------------- #

def compute_feature_pair_cosine_distances_pca(coefficients_dict: Dict[str, Dict[str, List[np.ndarray]]],
                                             feature_pairs: List[Tuple[str, str]],
                                             label_model_mapping: Dict[str, List[ModelConfig]],
                                             num_layers: int,
                                             n_components: int = 10) -> Dict[str, List[float]]:
    """
    Computes cosine distances between feature pairs across all layers using PCA-transformed coefficients.

    Args:
        coefficients_dict (Dict[str, Dict[str, List[np.ndarray]]]): Coefficients organized by label and model.
        feature_pairs (List[Tuple[str, str]]): List of feature pairs to compare.
        label_model_mapping (Dict[str, List[ModelConfig]]): Mapping from labels to model configurations.
        num_layers (int): Number of layers.
        n_components (int): Number of principal components for PCA.

    Returns:
        Dict[str, List[float]]: Dictionary where keys are feature pairs and values are lists of cosine distances across layers.
    """
    cosine_distance_dict_pca = {}

    # Collect all coefficients to fit PCA
    all_coefs = []
    for label, models in coefficients_dict.items():
        for model_name, coefs in models.items():
            for coef in coefs:
                if coef is not None:
                    all_coefs.append(coef)

    all_coefs = np.array(all_coefs)
    print(f"Total coefficients for PCA (Cosine Distances): {all_coefs.shape}")

    # Fit PCA on all coefficients
    pca = PCA(n_components=n_components)
    pca.fit(all_coefs)
    print(f"PCA explained variance ratio for cosine distances: {pca.explained_variance_ratio_}")

    # Transform coefficients using PCA
    coefficients_pca = {}
    for label, models in coefficients_dict.items():
        coefficients_pca[label] = {}
        for model_name, coefs in models.items():
            coefficients_pca[label][model_name] = []
            for coef in coefs:
                if coef is not None:
                    transformed = pca.transform(coef.reshape(1, -1)).flatten()
                    coefficients_pca[label][model_name].append(transformed)
                else:
                    coefficients_pca[label][model_name].append(None)

    # Compute cosine distances using PCA-transformed coefficients
    for feature1, feature2 in feature_pairs:
        key = f"{feature1} vs {feature2}"
        cosine_distance_dict_pca[key] = []

        for layer in range(num_layers):
            # Retrieve the first model for each feature (assuming one model per feature for simplicity)
            model1 = label_model_mapping[feature1][0].name
            model2 = label_model_mapping[feature2][0].name

            coef1 = coefficients_pca.get(feature1, {}).get(model1, [None] * num_layers)[layer]
            coef2 = coefficients_pca.get(feature2, {}).get(model2, [None] * num_layers)[layer]

            if coef1 is not None and coef2 is not None:
                try:
                    distance = calculate_cosine_distance(coef1, coef2)
                except ValueError:
                    distance = np.nan
                cosine_distance_dict_pca[key].append(distance)
            else:
                cosine_distance_dict_pca[key].append(np.nan)  # Handle missing data gracefully

    return cosine_distance_dict_pca

# ---------------------------- Save and Load Functions ---------------------------- #

def save_json(data: Dict, filepath: str):
    """
    Saves a dictionary to a JSON file.

    Args:
        data (Dict): Data to save.
        filepath (str): Path to the JSON file.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f)
    print(f"Data saved to {filepath}")

def load_json(filepath: str) -> Dict:
    """
    Loads a dictionary from a JSON file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        Dict: Loaded data.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Data loaded from {filepath}")
    return data

# ---------------------------- Main Orchestration Function ---------------------------- #

def main(label_model_mapping: Dict[str, List[ModelConfig]], methods: List[str], split_method: str = 'middle', data_file: str = 'periodic_table_dataset.csv'):
    """
    Main function to orchestrate the regression and cosine distance analysis for multiple labels and methods,
    using different models for each label.

    Args:
        label_model_mapping (Dict[str, List[ModelConfig]]): Mapping from labels to model configurations.
        methods (List[str]): List of regression methods to use (e.g., ['svr_cv']).
        split_method (str): Method to split the data ('middle', 'first', 'group_shuffle').
        data_file (str): Path to the dataset CSV file.
    """
    label_columns = list(label_model_mapping.keys())
    labels_repeated_dict = load_data(data_file, label_columns)

    # Dictionaries to store results
    r2_scores_dict: Dict[str, Dict[str, List[float]]] = {}
    coefficients_dict: Dict[str, Dict[str, List[np.ndarray]]] = {}  # Stores a list of coefficients for each layer

    # Iterate over each label and its corresponding model configurations
    for label, models in label_model_mapping.items():
        print(f"\nProcessing Label: {label}")
        labels_repeated = labels_repeated_dict[label]
        r2_scores_dict[label] = {}
        coefficients_dict[label] = {}

        for model in models:
            print(f"\nProcessing Model: {model.name}")
            r2_scores_dict[label][model.name] = []

            # Store coefficients for all layers
            coefficients_dict[label][model.name] = []

            for layer in range(model.num_layers):
                print(f"Loading data for Layer {layer}/{model.num_layers - 1}")
                try:
                    X_train, X_test, y_train, y_test = load_activation_data(
                        layer=layer,
                        activation_path_template=model.activation_path_template,
                        labels_repeated=labels_repeated,
                        split_method=split_method
                    )
                except FileNotFoundError as e:
                    print(e)
                    continue
                except ValueError as e:
                    print(e)
                    continue

                for method in methods:
                    if method == 'svr_cv':
                        print(f"Training SVR with cross-validation for Layer {layer}")
                        avg_r2, y_pred_all, coef_list = train_svr_cv(X_train, y_train)
                        r2_scores_dict[label][model.name].append(avg_r2)

                        if coef_list:
                            # Flatten each coefficient array before averaging
                            flattened_coefs = [coef.flatten() for coef in coef_list]
                            avg_coef = np.mean(flattened_coefs, axis=0)
                            coefficients_dict[label][model.name].append(avg_coef)  # Store coefficients for each layer
                            print(f"Obtained coefficients for {model.name} - {label} - Layer {layer}: {avg_coef[:5]}...")  # Print first few elements
                            print(f"The shape of the coefficients is: {avg_coef.shape}")
                        else:
                            print(f"Failed to obtain coefficients for {model.name} - {label} - Layer {layer}")

    # Save the coefficients_dict to a JSON file for future use
    coefficients_json_path = os.path.join(OUTPUT_DIR, 'data', 'coefficients.json')
    # Convert numpy arrays to lists for JSON serialization
    coefficients_serializable = {
        label: {
            model: [coef.tolist() for coef in layers]
            for model, layers in models.items()
        }
        for label, models in coefficients_dict.items()
    }
    with open(coefficients_json_path, 'w') as f:
        json.dump(coefficients_serializable, f)
    print(f"Coefficients saved to {coefficients_json_path}")

    # Define the feature pairs to compare
    feature_pairs = [
        ('Atomic Number', 'Atomic Mass'),
        ('Atomic Number', 'Group'),
        ('Atomic Number', 'Period'),
        ('Atomic Number', 'Electronegativity'),
        ('Atomic Mass', 'Group'),
        ('Atomic Mass', 'Period'),
        ('Atomic Mass', 'Electronegativity'),
        ('Group', 'Period'),
        ('Group', 'Electronegativity'),
        ('Period', 'Electronegativity')
    ]

    # Determine the maximum number of layers across all models
    num_layers = max([model.num_layers for models in label_model_mapping.values() for model in models])

    # Compute cosine distances between feature pairs across layers
    cosine_distances = compute_feature_pair_cosine_distances(
        coefficients_dict=coefficients_dict,
        feature_pairs=feature_pairs,
        label_model_mapping=label_model_mapping,
        num_layers=num_layers
    )

    # Save cosine_distances to a JSON file
    cosine_distance_json_path = os.path.join(OUTPUT_DIR, 'data', 'cosine_distances.json')
    cosine_distances_serializable = {pair: distances for pair, distances in cosine_distances.items()}
    with open(cosine_distance_json_path, 'w') as f:
        json.dump(cosine_distances_serializable, f)
    print(f"Cosine distances saved to {cosine_distance_json_path}")

    # Plot the cosine distances between feature pairs across layers
    plot_feature_pair_cosine_distances(
        cosine_distance_dict=cosine_distances,
        num_layers=num_layers,
        model_name='Meta-Llama-3.1-70B',  # Updated model name to match current configuration
        output_dir=os.path.join(OUTPUT_DIR, 'angles')
    )

    # Compute Pearson correlations between feature pairs across layers
    pearson_correlations = compute_feature_pair_pearson(
        coefficients_dict=coefficients_dict,
        feature_pairs=feature_pairs,
        label_model_mapping=label_model_mapping,
        num_layers=num_layers
    )

    # Save Pearson correlations to a JSON file
    pearson_json_path = os.path.join(OUTPUT_DIR, 'data', 'pearson_correlations.json')
    pearson_serializable = {pair: corrs for pair, corrs in pearson_correlations.items()}
    with open(pearson_json_path, 'w') as f:
        json.dump(pearson_serializable, f)
    print(f"Pearson correlations saved to {pearson_json_path}")

    # Plot the Pearson correlations between feature pairs across layers
    plot_feature_pair_pearson(
        pearson_dict=pearson_correlations,
        num_layers=num_layers,
        model_name='Meta-Llama-3.1-70B',  # Updated model name to match current configuration
        output_dir=os.path.join(OUTPUT_DIR, 'pearson_correlations')
    )

    # Compute cosine distance differences for each feature
    cosine_distance_differences = compute_feature_cosine_distance_differences(
        coefficients_dict=coefficients_dict,
        label_model_mapping=label_model_mapping,
        num_layers=num_layers
    )

    # Save cosine distance differences to a JSON file
    cosine_diff_json_path = os.path.join(OUTPUT_DIR, 'data', 'cosine_distance_differences.json')
    cosine_distance_differences_serializable = {feature: diffs for feature, diffs in cosine_distance_differences.items()}
    with open(cosine_diff_json_path, 'w') as f:
        json.dump(cosine_distance_differences_serializable, f)
    print(f"Cosine distance differences saved to {cosine_diff_json_path}")

    # Plot cosine distance differences for each feature
    plot_feature_cosine_distance_differences(
        cosine_diff_dict=cosine_distance_differences,
        num_layers=num_layers,
        model_name='Meta-Llama-3.1-70B',  # Updated model name to match current configuration
        output_dir=os.path.join(OUTPUT_DIR, 'angle_differences')
    )

    # ---------------------------- PCA-Based Cosine Distance Calculations ---------------------------- #

    # Compute PCA-based cosine distances between feature pairs across layers
    cosine_distances_pca = compute_feature_pair_cosine_distances_pca(
        coefficients_dict=coefficients_dict,
        feature_pairs=feature_pairs,
        label_model_mapping=label_model_mapping,
        num_layers=num_layers,
        n_components=10  # You can adjust the number of PCA components as needed
    )

    # Save PCA-based cosine distances to a JSON file
    cosine_distance_pca_json_path = os.path.join(OUTPUT_DIR, 'data', 'cosine_distances_pca.json')
    cosine_distances_pca_serializable = {pair: distances for pair, distances in cosine_distances_pca.items()}
    with open(cosine_distance_pca_json_path, 'w') as f:
        json.dump(cosine_distances_pca_serializable, f)
    print(f"PCA-based cosine distances saved to {cosine_distance_pca_json_path}")

    # Plot the PCA-based cosine distances between feature pairs across layers
    plot_feature_pair_cosine_distances_pca(
        cosine_distance_dict_pca=cosine_distances_pca,
        num_layers=num_layers,
        model_name='Meta-Llama-3.1-70B',
        output_dir=os.path.join(OUTPUT_DIR, 'angles_pca')
    )

# ---------------------------- Execute the Main Function ---------------------------- #

if __name__ == "__main__":
    # Define the regression methods to use, e.g., ['svr_cv'] for cross-validation
    regression_methods = ['svr_cv']  
    
    # Run the comparison using group_shuffle as the default data split method
    main(
        label_model_mapping=MODEL_CONFIGS,
        methods=regression_methods,
        split_method='group_shuffle',  # Options: 'middle', 'first', 'group_shuffle'
        data_file='periodic_table_dataset.csv'
    )
