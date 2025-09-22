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
from typing import List, Dict, Tuple, Any
import yaml
import argparse

# ---------------------------- Model Configuration ---------------------------- #

@dataclass
class ModelConfig:
    name: str
    activation_path_template: str  # Use {layer} as placeholder for layer index
    num_layers: int
    enabled: bool = True

@dataclass
class Config:
    """Configuration class to hold all settings from YAML file"""
    models: List[ModelConfig]
    experiment: Dict[str, Any]
    output: Dict[str, Any]
    training: Dict[str, Any]
    plotting: Dict[str, Any]
    comparison: Dict[str, Any]
    data_processing: Dict[str, Any]
    logging: Dict[str, Any]

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config_data = yaml.safe_load(file)
    
    # Convert model configurations to ModelConfig objects
    models = []
    for model_data in config_data['models']:
        if model_data.get('enabled', True):  # Only include enabled models
            models.append(ModelConfig(
                name=model_data['name'],
                activation_path_template=model_data['activation_path_template'],
                num_layers=model_data['num_layers'],
                enabled=model_data.get('enabled', True)
            ))
    
    return Config(
        models=models,
        experiment=config_data['experiment'],
        output=config_data['output'],
        training=config_data['training'],
        plotting=config_data['plotting'],
        comparison=config_data['comparison'],
        data_processing=config_data['data_processing'],
        logging=config_data['logging']
    )

# ---------------------------- Global Variables ---------------------------- #

# These will be set from config
config: Config = None
prompt_template_number: int = 11
predictions_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}  # Nested dict: model -> layer -> predictions
r2_scores_dict: Dict[str, List[float]] = {}  # Dict: model -> list of R² scores

# ---------------------------- Data Loading and Splitting ---------------------------- #

def load_data(file_path: str, label_column: str = 'Group', missing_fill_value: float = -np.inf) -> np.ndarray:
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
    
    # Fill missing values (NaN) with configured value
    labels = periodic_table[label_column].fillna(missing_fill_value).astype(float).values
    
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

def split_data_group_shuffle(labels_repeated: np.ndarray, test_size: float = 0.2, random_state: int = 100) -> (List[int], List[int]):
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
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.arange(len(valid_labels)), groups=groups))
    
    # Map back to original indices
    train_indices = np.where(valid_indices)[0][train_idx]
    test_indices = np.where(valid_indices)[0][test_idx]

    return train_indices.tolist(), test_indices.tolist()

def split_data(labels_repeated: np.ndarray, method: str, test_size: float = 0.2, random_state: int = 100) -> (List[int], List[int]):
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
        return split_data_group_shuffle(labels_repeated, test_size, random_state)
    else:
        raise ValueError(f"Unknown split method: {method}")

def load_activation_data(layer: int, activation_path_template: str, labels_repeated: np.ndarray, split_method: str = 'middle', test_size: float = 0.2, random_state: int = 100, check_consistency: bool = True) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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
    
    if check_consistency and activation_data.shape[0] != len(labels_repeated):
        raise ValueError(f"Inconsistent number of samples: {activation_data.shape[0]} features, {len(labels_repeated)} labels.")
    
    # Select the appropriate split method
    train_indices, test_indices = split_data(labels_repeated, split_method, test_size, random_state)

    X_train, X_test = activation_data[train_indices], activation_data[test_indices]
    y_train, y_test = labels_repeated[train_indices], labels_repeated[test_indices]

    return X_train, X_test, y_train, y_test

# ---------------------------- Model Training and Evaluation ---------------------------- #

def train_svr(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
              kernel: str = 'linear', c: float = 2, use_sample_weights: bool = True, 
              weight_method: str = 'balanced', use_scaler: bool = True) -> (float, np.ndarray):
    """
    Trains an SVR model and evaluates its performance using R² score.
    
    Args:
    - X_train, X_test: Training and testing features.
    - y_train, y_test: Training and testing labels.

    Returns:
    - r2_svr: R² score of the model.
    - y_pred_svr: Predicted labels for the test set.
    """
    if use_scaler:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    svr_model = SVR(kernel=kernel, C=c)
    
    if use_sample_weights:
        sample_weights = compute_sample_weight(weight_method, y_train)
        svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    else:
        svr_model.fit(X_train_scaled, y_train)

    y_pred_svr = svr_model.predict(X_test_scaled)
    r2_svr = r2_score(y_test, y_pred_svr)

    return r2_svr, y_pred_svr

def train_svr_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, kernel: str = 'linear', 
                 c: float = 2, use_sample_weights: bool = True, weight_method: str = 'balanced', 
                 use_scaler: bool = True) -> (float, np.ndarray):
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

        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        svr_model = SVR(kernel=kernel, C=c)
        
        if use_sample_weights:
            sample_weights = compute_sample_weight(weight_method, y_train)
            svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        else:
            svr_model.fit(X_train_scaled, y_train)

        y_pred = svr_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

        y_pred_all[test_index] = y_pred

    avg_r2 = np.mean(r2_scores)
    return avg_r2, y_pred_all

def train_random_forest(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray,
                        n_estimators: int = 100, random_state: int = 42) -> (float, np.ndarray):
    """
    Trains a Random Forest model and evaluates its performance using R² score.
    
    Args:
    - X_train, X_test: Training and testing features.
    - y_train, y_test: Training and testing labels.

    Returns:
    - r2_rf: R² score of the model.
    - y_pred_rf: Predicted labels for the test set.
    """
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)

    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)

    return r2_rf, y_pred_rf

def train_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, 
                method: str = 'svr', training_config: Dict[str, Any] = None) -> (float, np.ndarray):
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
    if training_config is None:
        training_config = {}
    
    if method == 'svr':
        svr_config = training_config.get('svr', {})
        return train_svr(X_train, X_test, y_train, y_test, **svr_config)
    elif method == 'random_forest':
        rf_config = training_config.get('random_forest', {})
        return train_random_forest(X_train, X_test, y_train, y_test, **rf_config)
    elif method == 'svr_cv':
        svr_config = training_config.get('svr', {})
        cv_config = training_config.get('cross_validation', {})
        n_splits = cv_config.get('n_splits', 5)
        return train_svr_cv(X_train, y_train, n_splits=n_splits, **svr_config)  # Cross-validation doesn't need X_test/y_test
    else:
        raise ValueError(f"Unknown method: {method}")

# ---------------------------- Plotting Functions ---------------------------- #

def plot_r2_trends_across_models(r2_scores_dict: Dict[str, List[float]], models: List[ModelConfig], 
                                  label_column: str, plotting_config: Dict[str, Any], output_config: Dict[str, Any]):
    """
    Plots R² score trends across normalized layer depth for multiple models and highlights the best layer.

    Args:
    - r2_scores_dict: Dictionary with model names as keys and their respective R² scores.
    - models: List of ModelConfig instances.
    - label_column: The label used for regression (e.g., 'Atomic Number').
    """
    output_dir = output_config.get('results_dir', 'Results/r2_trends_basic')
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    # Set plot style
    figsize = plotting_config.get('figure_sizes', {}).get('comparison', [5, 3])
    plt.figure(figsize=figsize)
    style = plotting_config.get('style', 'whitegrid')
    sns.set(style=style)

    palette = plotting_config.get('color_palette', 'husl')
    colors = sns.color_palette(palette, len(models))

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

    # Set y-axis limit
    ylim = plotting_config.get('r2_ylim', [0, 1])
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.ylim(ylim[0], ylim[1])
    
    # Customize labels and title with label_column
    font_sizes = plotting_config.get('font_sizes', {})
    plt.xlabel('Layer Depth Proportion', fontsize=font_sizes.get('xlabel', 14))
    plt.ylabel('R² Score', fontsize=font_sizes.get('ylabel', 14))
    plt.title(f'{label_column}', fontsize=font_sizes.get('title', 13))

    # Set grid and tick parameters
    if plotting_config.get('show_grid', True):
        grid_style = plotting_config.get('grid_style', '--')
        grid_linewidth = plotting_config.get('grid_linewidth', 0.7)
        plt.grid(True, which='major', linestyle=grid_style, linewidth=grid_linewidth)
    plt.xticks(fontsize=font_sizes.get('ticks', 12))
    plt.yticks(fontsize=font_sizes.get('ticks', 12))

    # Save the main plot without the legend
    plt.tight_layout()
    dpi = plotting_config.get('dpi', 300)
    if output_config.get('save_comparison_plots', True):
        plt.savefig(f'{output_dir}/r2_trends_comparison_normalized_{label_column}.png', dpi=dpi, bbox_inches='tight')
    plt.close()

    # Create a separate figure for the legend
    if output_config.get('save_legend_separately', True):
        legend_figsize = plotting_config.get('figure_sizes', {}).get('legend', [12, 1])
        fig_legend = plt.figure(figsize=legend_figsize)
        legend_fontsize = font_sizes.get('legend', 10)
        plt.figlegend(handles=legend_handles, loc='center', fontsize=legend_fontsize, ncol=len(legend_handles)//2)
        fig_legend.savefig(f'{output_dir}/r2_trends_legend_{label_column}.png', dpi=dpi, bbox_inches='tight')
        plt.close(fig_legend)






def plot_results(r2_scores: List[float], predictions_dict: Dict[int, Dict[str, np.ndarray]], best_layer: int, 
                 label_column: str, activation_filename: str, model_name: str, model_num_layers: int,
                 plotting_config: Dict[str, Any], output_config: Dict[str, Any]):
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
    output_dir = output_config.get('results_dir', 'Results/r2_trends_basic')
    os.makedirs(output_dir, exist_ok=True)

    # Define base file name with model name
    base_filename = f'{label_column}_{activation_filename}_layer_{best_layer}_model_{model_name}'

    # Plot R² score trends across layers
    figsize = plotting_config.get('figure_sizes', {}).get('r2_trend', [5, 3])
    plt.figure(figsize=figsize)
    plt.plot(range(model_num_layers), r2_scores, marker='o', linestyle='-', color='b', label='R² Score')
    if plotting_config.get('show_best_layer_line', True):
        plt.axvline(best_layer, color='r', linestyle='--', label=f'Max R² at Layer {best_layer}')
    ylim = plotting_config.get('r2_ylim', [0, 1])
    plt.ylim(ylim[0], ylim[1])
    font_sizes = plotting_config.get('font_sizes', {})
    plt.xlabel('Layer Index', fontsize=font_sizes.get('xlabel', 12))
    plt.ylabel('R² Score', fontsize=font_sizes.get('ylabel', 12))
    plt.title(f'R² Score Trend Across Layers - {model_name}', fontsize=font_sizes.get('title', 14))
    if plotting_config.get('show_grid', True):
        plt.grid(True)
    plt.legend()
    if output_config.get('save_individual_plots', True):
        dpi = plotting_config.get('dpi', 300)
        plt.savefig(f'{output_dir}/{base_filename}_r2_trend.png', dpi=dpi)
    plt.show()

    # Plot prediction results
    figsize = plotting_config.get('figure_sizes', {}).get('predictions', [18, 6])
    plt.figure(figsize=figsize)

    # 1. Scatter plot of true vs predicted values
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_best, y_pred_best, color='blue', label='SVR Predictions', alpha=0.7)
    plt.plot([y_test_best.min(), y_test_best.max()], [y_test_best.min(), y_test_best.max()], 'k--', lw=2)
    font_sizes = plotting_config.get('font_sizes', {})
    plt.xlabel('True Values', fontsize=font_sizes.get('xlabel', 12))
    plt.ylabel('Predicted Values', fontsize=font_sizes.get('ylabel', 12))
    plt.title(f'SVR (Layer {best_layer}): True vs Predicted - {model_name}', fontsize=font_sizes.get('title', 14))
    plt.legend()

    # 2. Residual plot
    plt.subplot(1, 3, 2)
    residuals_best = y_test_best - y_pred_best
    sns.histplot(residuals_best, kde=True, color='orange', bins=12)
    plt.axvline(0, color='k', linestyle='--', lw=2)
    plt.xlabel('Residuals', fontsize=font_sizes.get('xlabel', 12))
    plt.ylabel('Frequency', fontsize=font_sizes.get('ylabel', 12))
    plt.title(f'Residual Distribution (Layer {best_layer}) - {model_name}', fontsize=font_sizes.get('title', 14))

    # 3. True vs predicted values with error visualization
    plt.subplot(1, 3, 3)
    plt.scatter(np.arange(len(y_test_best)), y_test_best, label='True Values', color='green', alpha=0.6)
    plt.scatter(np.arange(len(y_test_best)), y_pred_best, label='Predicted Values', color='blue', alpha=0.6)
    plt.fill_between(np.arange(len(y_test_best)), y_test_best, y_pred_best, color='gray', alpha=0.3)
    plt.xlabel('Sample Index', fontsize=font_sizes.get('xlabel', 12))
    plt.ylabel('Values', fontsize=font_sizes.get('ylabel', 12))
    plt.title(f'True vs Predicted Values (Layer {best_layer}) - {model_name}', fontsize=font_sizes.get('title', 14))
    plt.legend()

    # Save the plot
    plt.tight_layout()
    if output_config.get('save_individual_plots', True):
        dpi = plotting_config.get('dpi', 300)
        plt.savefig(f'{output_dir}/{base_filename}_predictions.png', dpi=dpi)
        print(f"Saved results for model: {model_name}, layer: {best_layer}")
    plt.show()

def plot_r2_difference_between_models(r2_scores_dict: Dict[str, List[float]], models: List[ModelConfig], 
                                       model_pairs: List[Tuple[str, str]], label_column: str,
                                       plotting_config: Dict[str, Any], output_config: Dict[str, Any],
                                       comparison_config: Dict[str, Any]):
    """
    Plots the R² difference (delta) between pairs of models across normalized layer depth proportion.

    Args:
    - r2_scores_dict: Dictionary with model names as keys and their respective R² scores.
    - models: List of ModelConfig instances.
    - model_pairs: List of tuples containing pairs of model names to compare.
    - label_column: The label used for regression (e.g., 'Atomic Number').
    """
    output_dir = output_config.get('results_dir', 'Results/r2_trends_basic')
    os.makedirs(output_dir, exist_ok=True)

    figsize = plotting_config.get('figure_sizes', {}).get('comparison', [5, 3])
    plt.figure(figsize=figsize)
    style = plotting_config.get('style', 'whitegrid')
    sns.set(style=style)

    palette = plotting_config.get('color_palette', 'husl')
    colors = sns.color_palette(palette, len(model_pairs))

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
        interpolation_points = comparison_config.get('interpolation_points', 100)
        common_normalized_layers = np.linspace(0, 1, interpolation_points)

        interp_r2_scores1 = np.interp(common_normalized_layers, normalized_layers1, r2_scores1)
        interp_r2_scores2 = np.interp(common_normalized_layers, normalized_layers2, r2_scores2)

        # Compute delta R²
        delta_r2 = interp_r2_scores1 - interp_r2_scores2

        plt.plot(common_normalized_layers, delta_r2, marker='o', linestyle='-', color=colors[i],
                 label=f'{model_name1} - {model_name2}', markersize=4, linewidth=1.5)

    font_sizes = plotting_config.get('font_sizes', {})
    plt.xlabel('Layer Depth Proportion', fontsize=font_sizes.get('xlabel', 14))
    plt.ylabel('Δ R² Score', fontsize=font_sizes.get('ylabel', 14))
    plt.title('R² Difference Between Models Across Normalized Layer Depth', fontsize=font_sizes.get('title', 13))
    if plotting_config.get('show_grid', True):
        grid_style = plotting_config.get('grid_style', '--')
        grid_linewidth = plotting_config.get('grid_linewidth', 0.7)
        plt.grid(True, linestyle=grid_style, linewidth=grid_linewidth)
    plt.legend(fontsize=font_sizes.get('legend', 10))
    plt.tight_layout()
    if output_config.get('save_comparison_plots', True):
        dpi = plotting_config.get('dpi', 300)
        plt.savefig(f'{output_dir}/r2_difference_between_models_{label_column}.png', dpi=dpi)
    plt.show()

# ---------------------------- Main Orchestration Function ---------------------------- #

def main(config: Config):
    """
    Main function to orchestrate the regression and R² score analysis using configuration.

    Args:
    - config: Configuration object containing all settings.
    """
    global prompt_template_number
    
    # Extract configuration values
    models = config.models
    methods = config.experiment['regression_methods']
    split_method = config.experiment['split_method']
    label_column = config.experiment['label_column']
    dataset_file = config.experiment['dataset_file']
    prompt_template_number = config.experiment['prompt_template_number']
    
    # Load data with configured parameters
    missing_fill = config.data_processing.get('missing_value_fill', -np.inf)
    if missing_fill == 'inf':
        missing_fill = np.inf
    elif missing_fill == '-inf':
        missing_fill = -np.inf
    
    labels_repeated = load_data(dataset_file, label_column, missing_fill)

    for model in models:
        print(f"\nProcessing Model: {model.name}")
        r2_scores_dict[model.name] = []
        predictions_dict[model.name] = {}
        activation_filename = config.experiment.get('activation_filename', 'last.11_templates')

        for layer in range(model.num_layers):
            try:
                # Get cross-validation config for data splitting
                cv_config = config.training.get('cross_validation', {})
                test_size = cv_config.get('test_size', 0.2)
                random_state = cv_config.get('random_state', 100)
                check_consistency = config.data_processing.get('check_sample_consistency', True)
                
                X_train, X_test, y_train, y_test = load_activation_data(
                    layer=layer,
                    activation_path_template=model.activation_path_template,
                    labels_repeated=labels_repeated,
                    split_method=split_method,
                    test_size=test_size,
                    random_state=random_state,
                    check_consistency=check_consistency
                )
            except FileNotFoundError as e:
                if config.logging.get('print_file_not_found_errors', True):
                    print(e)
                continue
            except ValueError as e:
                if config.logging.get('print_value_errors', True):
                    print(e)
                continue

            for method in methods:
                if config.logging.get('print_layer_progress', True):
                    print(f"Training with method: {method} for model: {model.name}, layer: {layer}")
                
                # Check for 'svr_cv' in the methods list
                if method == 'svr_cv':
                    r2, y_pred = train_model(X_train, X_test, y_train, y_test, method=method, training_config=config.training)
                    y_test_to_store = y_train  # Since we're using CV, the "y_test" will be from the full training set
                else:
                    r2, y_pred = train_model(X_train, X_test, y_train, y_test, method=method, training_config=config.training)
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
            if config.logging.get('print_best_layer_info', True):
                print(f"Model: {model.name}, Best R² score at layer {best_layer}, Score: {best_r2:.4f}")

            # Plot results for the best layer
            plot_results(
                r2_scores=r2_scores_dict[model.name],
                predictions_dict=predictions_dict[model.name],
                best_layer=best_layer,
                label_column=label_column,
                activation_filename=activation_filename,
                model_name=model.name,
                model_num_layers=model.num_layers,
                plotting_config=config.plotting,
                output_config=config.output
            )
        else:
            print(f"No valid R² scores computed for model: {model.name}")

    # Plot R² trends across all models
    plot_r2_trends_across_models(r2_scores_dict, models, label_column, config.plotting, config.output)

    # Get model pairs from config
    model_pairs_to_compare = [tuple(pair) for pair in config.comparison.get('model_pairs', [])]
    
    if model_pairs_to_compare:
        plot_r2_difference_between_models(r2_scores_dict, models, model_pairs_to_compare, 
                                         label_column, config.plotting, config.output, config.comparison)


# ---------------------------- Execute the Main Function ---------------------------- #

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run basic linear regression analysis with configurable parameters')
    parser.add_argument('--config', '-c', type=str, default='config_linear_regression.yaml',
                       help='Path to configuration YAML file (default: config_linear_regression.yaml)')
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
        print(f"Enabled models: {[model.name for model in config.models]}")
        print(f"Target label: {config.experiment['label_column']}")
        print(f"Split method: {config.experiment['split_method']}")
        print(f"Regression methods: {config.experiment['regression_methods']}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Please create a configuration file or specify a valid path using --config")
        exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)
    
    # Run the main function
    main(config)
