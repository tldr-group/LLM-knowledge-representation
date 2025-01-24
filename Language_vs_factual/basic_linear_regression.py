import torch
import os
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass
from typing import List, Dict, Tuple

# ---------------------------- Model Configuration ---------------------------- #

@dataclass
class ModelConfig:
    name: str
    activation_path_template: str  # Use {layer} as placeholder for layer index
    num_layers: int

# Example configurations for different models
MODEL_CONFIGS: List[ModelConfig] = [

    #  ModelConfig(
    #     name='Llama-3.1-8B',
    #     activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/group/group.last.11_templates.{layer}.pt',
    #     num_layers=32
    # ),
    
    # ModelConfig(
    #     name='Meta-Llama-3.1-70B (description)',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group/group.last.11_templates.{layer}.pt',
    #     num_layers=80
    # ),

    #     ModelConfig(
    #     name='Llama-2-7b-hf',
    #     activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/group/group.last.11_templates.{layer}.pt',
    #     num_layers=32
    # ),
    # ModelConfig(
    #     name='Llama-2-7b-hf (question prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/group question/group question.last.11_templates_questions.{layer}.pt',
    #     num_layers=32
    # ),
   
    # ModelConfig(
    #     name='Llama-3.1-8B',
    #     activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/group/group.last.11_templates.{layer}.pt',
    #     num_layers=32
    # ),

    #     ModelConfig(
    #     name='Llama-3.1-8B (question prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/group question/group question.last.11_templates_questions.{layer}.pt',
    #     num_layers=32  # Example layer count
    # ),

    #     ModelConfig(
    #     name='Meta-Llama-3.1-70B',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group/group.last.11_templates.{layer}.pt',
    #     num_layers=80  # Example layer count
    # ),

    #     ModelConfig(
    #     name='Meta-Llama-3.1-70B (question prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group question/group question.last.11_templates_questions.{layer}.pt',
    #     num_layers=80  # Example layer count
    # ),



    # ModelConfig(
    #     name='Llama-3.1-8B-Instruct',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3-8B-Instruct/atomic number/atomic number.last.11_templates.{layer}.pt',
    #     num_layers=32  # Example layer count
    # ),


    # ModelConfig(
    #     name='Llama-2-7b-hf',
    #     activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/group/group.last.11_templates_quesiontions.{layer}.pt',
    #     num_layers=32
    # ),

    # ModelConfig(
    #     name='Llama-2-7b-hf',
    #     activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/atomic number/atomic number.last.11_templates.{layer}.pt',
    #     num_layers=32
    # ),
    


    # ModelConfig(
    #     name='Meta-Llama-3.1-70B',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number/atomic number.last.11_templates.{layer}.pt',
    #     num_layers=80  # Example layer count
    # ),


    # ModelConfig(
    #     name='Meta-Llama-3.1-70B',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/following atomic number/following atomic number.last.11_templates_following.{layer}.pt',
    #     num_layers=80  # Example layer count
    # )

    ModelConfig(
        name='Llama-2-7b-hf',
        activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/atomic number/atomic number.last.11_templates.{layer}.pt',
        num_layers=32
    ),

    ModelConfig(
        name='Llama-3.1-8B',
        activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/atomic number/atomic number.last.11_templates.{layer}.pt',
        num_layers=32
    ),


    ModelConfig(
        name='Meta-Llama-3.1-70B',
        activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number/atomic number.last.11_templates.{layer}.pt',
        num_layers=80
    ),


]

# ---------------------------- Global Variables ---------------------------- #

prompt_template_number = 11
predictions_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}  # Nested dict: model -> layer -> predictions
r2_scores_dict: Dict[str, List[float]] = {}  # Dict: model -> list of R² scores

# ---------------------------- Data Loading and Splitting ---------------------------- #

def load_data(file_path: str, label_column: str = 'Group') -> np.ndarray:
    """
    Loads the periodic table dataset and returns the labels with missing values filled with -inf.
    
    Args:
    - file_path: Path to the periodic table dataset.
    - label_column: The column to use as regression labels.

    Returns:
    - labels_repeated: The labels repeated according to prompt_template_number.
    """
    # Load the dataset
    periodic_table = pd.read_csv(file_path)
    print(f"Loaded dataset with columns: {periodic_table.columns.tolist()}")
    
    # Fill missing values (NaN) with -inf 
    labels = periodic_table[label_column].fillna(-np.inf).astype(float).values
    
    # Repeat the labels according to the prompt_template_number
    labels_repeated = np.repeat(labels, prompt_template_number)
    
    return labels_repeated

def split_data_middle_group(labels_repeated: np.ndarray) -> (List[int], List[int]):
    """
    Splits the data by selecting the middle group as the test set.
    
    Args:
    - labels_repeated: Repeated labels for the dataset.

    Returns:
    - train_indices, test_indices: Indices for training and test sets.
    """
    train_indices = []
    test_indices = []

    for label in np.unique(labels_repeated):
        label_indices = np.where(labels_repeated == label)[0]
        n_groups = len(label_indices) // prompt_template_number
        middle_group = n_groups // 2
        start_idx = middle_group * prompt_template_number
        end_idx = start_idx + prompt_template_number

        test_indices.extend(label_indices[start_idx:end_idx])
        train_indices.extend(np.delete(label_indices, np.arange(start_idx, end_idx))) 

    return train_indices, test_indices

def split_data_first_group(labels_repeated: np.ndarray) -> (List[int], List[int]):
    """
    Splits the data by selecting the first group as the test set.
    
    Args:
    - labels_repeated: Repeated labels for the dataset.

    Returns:
    - train_indices, test_indices: Indices for training and test sets.
    """
    train_indices = []
    test_indices = []

    for label in np.unique(labels_repeated):
        label_indices = np.where(labels_repeated == label)[0]
        test_indices.extend(label_indices[:prompt_template_number])  # First group as test set
        train_indices.extend(label_indices[prompt_template_number:])  # Remaining as training set

    return train_indices, test_indices

def split_data_group_shuffle(labels_repeated: np.ndarray) -> (List[int], List[int]):
    """
    Splits the data randomly using GroupShuffleSplit, excluding rows with abnormal values.
    
    Args:
    - labels_repeated: Repeated labels for the dataset.

    Returns:
    - train_indices, test_indices: Indices for training and test sets.
    """
    # Exclude abnormal values (e.g., -np.inf)
    valid_indices = np.isfinite(labels_repeated)  # This keeps only finite values, excluding -np.inf, NaN, etc.
    valid_labels = labels_repeated[valid_indices]
    
    # Generate group labels
    groups = np.repeat(np.arange(len(valid_labels) // prompt_template_number), prompt_template_number)
    
    # Perform group shuffle split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=100)
    train_idx, test_idx = next(gss.split(np.arange(len(valid_labels)), groups=groups))
    
    # Map back to original indices
    train_indices = np.where(valid_indices)[0][train_idx]
    test_indices = np.where(valid_indices)[0][test_idx]

    return train_indices.tolist(), test_indices.tolist()

def split_data(labels_repeated: np.ndarray, method: str) -> (List[int], List[int]):
    """
    Splits the data using the specified method.
    
    Args:
    - labels_repeated: Repeated labels for the dataset.
    - method: The split method ('middle', 'first', 'group_shuffle').

    Returns:
    - train_indices, test_indices: Indices for training and test sets.
    """
    if method == 'middle':
        return split_data_middle_group(labels_repeated)
    elif method == 'first':
        return split_data_first_group(labels_repeated)
    elif method == 'group_shuffle':
        return split_data_group_shuffle(labels_repeated)
    else:
        raise ValueError(f"Unknown split method: {method}")

def load_activation_data(layer: int, activation_path_template: str, labels_repeated: np.ndarray, split_method: str = 'middle') -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Loads activation data for a given layer and splits it into train and test sets using a specified method.
    
    Args:
    - layer: The layer index.
    - activation_path_template: The template path for activation files with {layer} as placeholder.
    - labels_repeated: The repeated labels.
    - split_method: Method for splitting ('middle', 'first', 'group_shuffle').

    Returns:
    - X_train, X_test: Training and testing features.
    - y_train, y_test: Training and testing labels.
    """
    file_path = activation_path_template.format(layer=layer)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    
    activation_data = torch.load(file_path, weights_only=True).cpu().numpy()
    
    if activation_data.shape[0] != len(labels_repeated):
        raise ValueError(f"Inconsistent number of samples: {activation_data.shape[0]} features, {len(labels_repeated)} labels.")
    
    # Select the appropriate split method
    train_indices, test_indices = split_data(labels_repeated, split_method)

    X_train, X_test = activation_data[train_indices], activation_data[test_indices]
    y_train, y_test = labels_repeated[train_indices], labels_repeated[test_indices]

    return X_train, X_test, y_train, y_test

# ---------------------------- Model Training and Evaluation ---------------------------- #

def train_svr(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> (float, np.ndarray):
    """
    Trains an SVR model and evaluates its performance using R² score.
    
    Args:
    - X_train, X_test: Training and testing features.
    - y_train, y_test: Training and testing labels.

    Returns:
    - r2_svr: R² score of the model.
    - y_pred_svr: Predicted labels for the test set.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sample_weights = compute_sample_weight('balanced', y_train)
    svr_model = SVR(kernel='linear', C=2)
    svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    y_pred_svr = svr_model.predict(X_test_scaled)
    r2_svr = r2_score(y_test, y_pred_svr)

    return r2_svr, y_pred_svr

def train_svr_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> (float, np.ndarray):
    """
    Trains an SVR model using 5-fold grouped cross-validation where augmented data is kept in the same group.

    Args:
    - X: Feature matrix.
    - y: Labels.
    - n_splits: Number of splits for cross-validation (default: 5).

    Returns:
    - avg_r2: Average R² score across all folds.
    - y_pred_all: Predicted labels for the test sets across all folds.
    """
    groups = np.repeat(np.arange(len(y) // prompt_template_number), prompt_template_number)
    
    gkf = GroupKFold(n_splits=n_splits)
    r2_scores = []
    y_pred_all = np.zeros_like(y)

    for train_index, test_index in gkf.split(X, y, groups=groups):
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

    avg_r2 = np.mean(r2_scores)
    return avg_r2, y_pred_all

def train_random_forest(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> (float, np.ndarray):
    """
    Trains a Random Forest model and evaluates its performance using R² score.
    
    Args:
    - X_train, X_test: Training and testing features.
    - y_train, y_test: Training and testing labels.

    Returns:
    - r2_rf: R² score of the model.
    - y_pred_rf: Predicted labels for the test set.
    """
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)

    return r2_rf, y_pred_rf

def train_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, method: str = 'svr') -> (float, np.ndarray):
    """
    Trains a regression model based on the specified method and evaluates its performance using R² score.
    
    Args:
    - X_train, X_test: Training and testing features.
    - y_train, y_test: Training and testing labels.
    - method: Regression method ('svr', 'random_forest', 'svr_cv').

    Returns:
    - r2: R² score of the model.
    - y_pred: Predicted labels for the test set.
    """
    if method == 'svr':
        return train_svr(X_train, X_test, y_train, y_test)
    elif method == 'random_forest':
        return train_random_forest(X_train, X_test, y_train, y_test)
    elif method == 'svr_cv':
        return train_svr_cv(X_train, y_train)  # Cross-validation doesn't need X_test/y_test
    else:
        raise ValueError(f"Unknown method: {method}")

# ---------------------------- Plotting Functions ---------------------------- #

def plot_r2_trends_across_models(r2_scores_dict: Dict[str, List[float]], models: List[ModelConfig], label_column: str):
    """
    Plots R² score trends across normalized layer depth for multiple models and highlights the best layer.

    Args:
    - r2_scores_dict: Dictionary with model names as keys and their respective R² scores.
    - models: List of ModelConfig instances.
    - label_column: The label used for regression (e.g., 'Atomic Number').
    """
    output_dir = 'Results/r2_trends_basic'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    # Set plot style
    plt.figure(figsize=(5, 3))
    sns.set(style="whitegrid")  # Use seaborn whitegrid style for a cleaner look

    colors = sns.color_palette("husl", len(models))  # Generate a color palette with distinct colors

    legend_handles = []

    for i, model in enumerate(models):
        r2_scores = r2_scores_dict[model.name]
        num_layers = model.num_layers
        
        if not r2_scores:  # Check if r2_scores is empty
            print(f"Warning: No R² scores found for model {model.name}. Skipping this model.")
            continue

        # Normalize layer index to be between 0 and 1
        normalized_layers = [layer / num_layers for layer in range(num_layers)]
        
        # Ensure that the lengths match before plotting
        if len(normalized_layers) != len(r2_scores):
            print(f"Error: Model {model.name} has a mismatch between normalized layers ({len(normalized_layers)}) and R² scores ({len(r2_scores)}). Skipping this model.")
            continue
        
        # Plot the R² score trend line
        line, = plt.plot(normalized_layers, r2_scores, 
                         marker='o', linestyle='-', color=colors[i], 
                         label=f'{model.name}', markersize=3, linewidth=1)
        
        # Collect the legend handles
        legend_handles.append(line)
        
        # Find the best layer (the one with the highest R² score)
        best_layer_index = r2_scores.index(max(r2_scores))
        best_layer_normalized = best_layer_index / num_layers
        
        # Add a vertical dashed line at the best layer
        best_layer_line = plt.axvline(x=best_layer_normalized, color=colors[i], linestyle='--', linewidth=1.5, 
                                      label=f'{model.name} Best Layer')
        legend_handles.append(best_layer_line)

    # Set y-axis limit from 0 to 1
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.ylim(0, 1)
    
    # Customize labels and title with label_column
    plt.xlabel('Layer Depth Proportion', fontsize=14)
    plt.ylabel('R² Score', fontsize=14)
    plt.title(f'{label_column}', fontsize=13)

    # Set grid and tick parameters
    plt.grid(True, which='major', linestyle='--', linewidth=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save the main plot without the legend
    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_trends_comparison_normalized_{label_column}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create a separate figure for the legend
    fig_legend = plt.figure(figsize=(12, 1))  # Very wide and short
    plt.figlegend(handles=legend_handles, loc='center', fontsize=10, ncol=len(legend_handles)//2)
    fig_legend.savefig(f'{output_dir}/r2_trends_legend_{label_column}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_legend)






def plot_results(r2_scores: List[float], predictions_dict: Dict[int, Dict[str, np.ndarray]], best_layer: int, label_column: str, activation_filename: str, model_name: str, model_num_layers: int):
    """
    Plots the R² scores and the predictions for the best layer and saves the results with the model name.

    Args:
    - r2_scores: List of R² scores for each layer.
    - predictions_dict: Dictionary containing predictions for each layer.
    - best_layer: The index of the layer with the highest R² score.
    - label_column: The label used for regression (e.g., 'Group').
    - activation_filename: Part of the filename for activation data (e.g., 'element.last.11_templates').
    - model_name: The name of the model (e.g., 'llama2').
    - model_num_layers: Total number of layers in the model.
    """
    y_test_best = predictions_dict[best_layer]['y_test']
    y_pred_best = predictions_dict[best_layer]['y_pred']

    # Create directory for results if it doesn't exist
    output_dir = 'Results/r2_trends_basic'
    os.makedirs(output_dir, exist_ok=True)

    # Define base file name with model name
    base_filename = f'{label_column}_{activation_filename}_layer_{best_layer}_model_{model_name}'

    # Plot R² score trends across layers
    plt.figure(figsize=(5,3))
    plt.plot(range(model_num_layers), r2_scores, marker='o', linestyle='-', color='b', label='R² Score')
    plt.axvline(best_layer, color='r', linestyle='--', label=f'Max R² at Layer {best_layer}')
    plt.ylim(0, 1)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title(f'R² Score Trend Across Layers - {model_name}', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/{base_filename}_r2_trend.png')
    plt.show()

    # Plot prediction results
    plt.figure(figsize=(18, 6))

    # 1. Scatter plot of true vs predicted values
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_best, y_pred_best, color='blue', label='SVR Predictions', alpha=0.7)
    plt.plot([y_test_best.min(), y_test_best.max()], [y_test_best.min(), y_test_best.max()], 'k--', lw=2)
    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'SVR (Layer {best_layer}): True vs Predicted - {model_name}', fontsize=14)
    plt.legend()

    # 2. Residual plot
    plt.subplot(1, 3, 2)
    residuals_best = y_test_best - y_pred_best
    sns.histplot(residuals_best, kde=True, color='orange', bins=12)
    plt.axvline(0, color='k', linestyle='--', lw=2)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Residual Distribution (Layer {best_layer}) - {model_name}', fontsize=14)

    # 3. True vs predicted values with error visualization
    plt.subplot(1, 3, 3)
    plt.scatter(np.arange(len(y_test_best)), y_test_best, label='True Values', color='green', alpha=0.6)
    plt.scatter(np.arange(len(y_test_best)), y_pred_best, label='Predicted Values', color='blue', alpha=0.6)
    plt.fill_between(np.arange(len(y_test_best)), y_test_best, y_pred_best, color='gray', alpha=0.3)
    plt.xlabel('Sample Index', fontsize=12)
    plt.ylabel('Values', fontsize=12)
    plt.title(f'True vs Predicted Values (Layer {best_layer}) - {model_name}', fontsize=14)
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{base_filename}_predictions.png')
    print(f"Saved results for model: {model_name}, layer: {best_layer}")
    plt.show()

def plot_r2_difference_between_models(r2_scores_dict: Dict[str, List[float]], models: List[ModelConfig], model_pairs: List[Tuple[str, str]], label_column: str):
    """
    Plots the R² difference (delta) between pairs of models across normalized layer depth proportion.

    Args:
    - r2_scores_dict: Dictionary with model names as keys and their respective R² scores.
    - models: List of ModelConfig instances.
    - model_pairs: List of tuples containing pairs of model names to compare.
    - label_column: The label used for regression (e.g., 'Atomic Number').
    """
    output_dir = 'Results/r2_trends_basic'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(5,3))
    sns.set(style="whitegrid")

    colors = sns.color_palette("husl", len(model_pairs))

    # Create a dictionary to map model names to their configurations
    model_config_dict = {model.name: model for model in models}

    for i, (model_name1, model_name2) in enumerate(model_pairs):
        r2_scores1 = r2_scores_dict.get(model_name1, [])
        r2_scores2 = r2_scores_dict.get(model_name2, [])

        if not r2_scores1 or not r2_scores2:
            print(f"Warning: Missing R² scores for model pair ({model_name1}, {model_name2}). Skipping this pair.")
            continue

        num_layers1 = model_config_dict[model_name1].num_layers
        num_layers2 = model_config_dict[model_name2].num_layers

        # Normalize layer indices for both models
        normalized_layers1 = [layer / num_layers1 for layer in range(len(r2_scores1))]
        normalized_layers2 = [layer / num_layers2 for layer in range(len(r2_scores2))]

        # Interpolate R² scores onto a common set of normalized layers
        common_normalized_layers = np.linspace(0, 1, 100)  # 100 points between 0 and 1

        interp_r2_scores1 = np.interp(common_normalized_layers, normalized_layers1, r2_scores1)
        interp_r2_scores2 = np.interp(common_normalized_layers, normalized_layers2, r2_scores2)

        # Compute delta R²
        delta_r2 = interp_r2_scores1 - interp_r2_scores2

        plt.plot(common_normalized_layers, delta_r2, marker='o', linestyle='-', color=colors[i],
                 label=f'{model_name1} - {model_name2}', markersize=4, linewidth=1.5)

    plt.xlabel('Layer Depth Proportion', fontsize=14)
    plt.ylabel('Δ R² Score', fontsize=14)
    plt.title('R² Difference Between Models Across Normalized Layer Depth', fontsize=13)
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_difference_between_models_{label_column}.png', dpi=300)
    plt.show()

# ---------------------------- Main Orchestration Function ---------------------------- #

def main(models: List[ModelConfig], methods: List[str], split_method: str = 'middle', label_column: str = 'Group'):
    """
    Main function to orchestrate the regression and R² score analysis for multiple models and methods.

    Args:
    - models: List of ModelConfig instances.
    - methods: List of regression methods to use (e.g., ['svr', 'random_forest', 'svr_cv']).
    - split_method: The method used to split the data (default: 'middle').
    - label_column: The column used as the regression target (e.g., 'Group').
    """
    labels_repeated = load_data('periodic_table_dataset.csv', label_column)

    for model in models:
        print(f"\nProcessing Model: {model.name}")
        r2_scores_dict[model.name] = []
        predictions_dict[model.name] = {}
        activation_filename = 'last.11_templates'

        for layer in range(model.num_layers):
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
                print(f"Training with method: {method} for model: {model.name}, layer: {layer}")
                
                # Check for 'svr_cv' in the methods list
                if method == 'svr_cv':
                    r2, y_pred = train_svr_cv(X_train, y_train)  # Only train with CV on the training data
                    y_test_to_store = y_train  # Since we're using CV, the "y_test" will be from the full training set
                else:
                    r2, y_pred = train_model(X_train, X_test, y_train, y_test, method=method)
                    y_test_to_store = y_test  # Regular test data when not using CV

                # Store R² score for the current model and layer
                r2_scores_dict[model.name].append(r2)

                # Store predictions for the current layer
                predictions_dict[model.name][layer] = {
                    'y_test': y_test_to_store,
                    'y_pred': y_pred
                }

        # Determine the best layer for the model
        if r2_scores_dict[model.name]:
            best_layer = np.argmax(r2_scores_dict[model.name])
            best_r2 = np.max(r2_scores_dict[model.name])
            print(f"Model: {model.name}, Best R² score at layer {best_layer}, Score: {best_r2:.4f}")

            # Plot results for the best layer
            plot_results(
                r2_scores=r2_scores_dict[model.name],
                predictions_dict=predictions_dict[model.name],
                best_layer=best_layer,
                label_column=label_column,
                activation_filename=activation_filename,
                model_name=model.name,
                model_num_layers=model.num_layers
            )
        else:
            print(f"No valid R² scores computed for model: {model.name}")

    # Plot R² trends across all models
    plot_r2_trends_across_models(r2_scores_dict, models, label_column)

    model_pairs_to_compare = [
        ('Meta-Llama-3.1-70B', 'Meta-Llama-3.1-70B (question prompt)'),
        ('Llama-3.1-8B', 'Llama-3.1-8B (question prompt)'),
        ('Llama-2-7b-hf', 'Llama-2-7b-hf (question prompt)')
        # Add more model pairs as needed
    ]

    plot_r2_difference_between_models(r2_scores_dict, models, model_pairs_to_compare, label_column)


# ---------------------------- Execute the Main Function ---------------------------- #

if __name__ == "__main__":
    # Define the models to compare with their configurations
    # Ensure that the activation_path_template contains a '{layer}' placeholder
    models_to_compare = MODEL_CONFIGS  # List of ModelConfig instances defined above
    
    # Define the regression methods to use
    regression_methods = ["svr_cv"]  # Add or remove methods as needed
    
    # run the main function
    main(
        models=models_to_compare,
        methods=regression_methods,
        split_method='group_shuffle',  # Options: 'middle', 'first', 'group_shuffle'
        label_column='Random'  # Change to desired label column
    )
