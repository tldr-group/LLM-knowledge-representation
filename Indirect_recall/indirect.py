
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
import argparse
import yaml

# ---------------------------- Model Configuration ---------------------------- #

@dataclass
class ModelConfig:
    name: str
    activation_path_template: str  # Use {layer} as placeholder for layer index
    num_layers: int

# Configuration will be loaded from YAML file

# ---------------------------- Configuration Loading ---------------------------- #

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_enabled_models(config: Dict[str, Any]) -> List[ModelConfig]:
    """Get list of enabled models from configuration."""
    models = []
    for model_config in config['models']:
        if model_config.get('enabled', True):  # Default to enabled if not specified
            models.append(ModelConfig(
                name=model_config['name'],
                activation_path_template=model_config['activation_path_template'],
                num_layers=model_config['num_layers']
            ))
    return models

# ---------------------------- Global Variables ---------------------------- #

predictions_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
r2_scores_dict: Dict[str, List[float]] = {}

# ---------------------------- Data Loading and Splitting ---------------------------- #

def load_data(file_path: str, label_column: str = 'Group', prompt_template_number: int = 11) -> np.ndarray:
    periodic_table = pd.read_csv(file_path)
    print(f"Loaded dataset with columns: {periodic_table.columns.tolist()}")
    labels = periodic_table[label_column].fillna(-np.inf).astype(float).values
    labels_repeated = np.repeat(labels, prompt_template_number)
    return labels_repeated

def split_data_middle_group(labels_repeated: np.ndarray, prompt_template_number: int = 11) -> (List[int], List[int]):
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

def split_data_first_group(labels_repeated: np.ndarray, prompt_template_number: int = 11) -> (List[int], List[int]):
    train_indices = []
    test_indices = []
    for label in np.unique(labels_repeated):
        label_indices = np.where(labels_repeated == label)[0]
        test_indices.extend(label_indices[:prompt_template_number])
        train_indices.extend(label_indices[prompt_template_number:])
    return train_indices, test_indices

def split_data_group_shuffle(labels_repeated: np.ndarray, prompt_template_number: int = 11, random_state: int = 100) -> (List[int], List[int]):
    valid_indices = np.isfinite(labels_repeated)
    valid_labels = labels_repeated[valid_indices]
    groups = np.repeat(np.arange(len(valid_labels) // prompt_template_number), prompt_template_number)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, test_idx = next(gss.split(np.arange(len(valid_labels)), groups=groups))
    train_indices = np.where(valid_indices)[0][train_idx]
    test_indices = np.where(valid_indices)[0][test_idx]
    return train_indices.tolist(), test_indices.tolist()

def split_data(labels_repeated: np.ndarray, method: str, prompt_template_number: int = 11, random_state: int = 100) -> (List[int], List[int]):
    if method == 'middle':
        return split_data_middle_group(labels_repeated, prompt_template_number)
    elif method == 'first':
        return split_data_first_group(labels_repeated, prompt_template_number)
    elif method == 'group_shuffle':
        return split_data_group_shuffle(labels_repeated, prompt_template_number, random_state)
    else:
        raise ValueError(f"Unknown split method: {method}")

def load_activation_data(layer: int, activation_path_template: str, labels_repeated: np.ndarray, split_method: str = 'middle', prompt_template_number: int = 11, random_state: int = 100) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    file_path = activation_path_template.format(layer=layer)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    activation_data = torch.load(file_path, weights_only=True).cpu().numpy()
    if activation_data.shape[0] != len(labels_repeated):
        raise ValueError(f"Inconsistent number of samples: {activation_data.shape[0]} features, {len(labels_repeated)} labels.")
    train_indices, test_indices = split_data(labels_repeated, split_method, prompt_template_number, random_state)
    X_train, X_test = activation_data[train_indices], activation_data[test_indices]
    y_train, y_test = labels_repeated[train_indices], labels_repeated[test_indices]
    return X_train, X_test, y_train, y_test

# ---------------------------- Model Training and Evaluation ---------------------------- #

def train_svr(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, kernel: str = 'linear', C: float = 2) -> (float, np.ndarray):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    sample_weights = compute_sample_weight('balanced', y_train)
    svr_model = SVR(kernel=kernel, C=C)
    svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    y_pred_svr = svr_model.predict(X_test_scaled)
    r2_svr = r2_score(y_test, y_pred_svr)
    return r2_svr, y_pred_svr

def train_svr_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, prompt_template_number: int = 11, kernel: str = 'linear', C: float = 2) -> (float, np.ndarray):
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
        svr_model = SVR(kernel=kernel, C=C)
        svr_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        y_pred = svr_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        y_pred_all[test_index] = y_pred
    avg_r2 = np.mean(r2_scores)
    return avg_r2, y_pred_all

def train_random_forest(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, n_estimators: int = 100, random_state: int = 42) -> (float, np.ndarray):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    return r2_rf, y_pred_rf

def train_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, method: str = 'svr', config: Dict[str, Any] = None) -> (float, np.ndarray):
    if config is None:
        config = {}
    
    training_config = config.get('training', {})
    
    if method == 'svr':
        return train_svr(X_train, X_test, y_train, y_test, 
                        kernel=training_config.get('svr_kernel', 'linear'),
                        C=training_config.get('svr_c', 2))
    elif method == 'random_forest':
        return train_random_forest(X_train, X_test, y_train, y_test,
                                 n_estimators=training_config.get('rf_n_estimators', 100),
                                 random_state=training_config.get('rf_random_state', 42))
    elif method == 'svr_cv':
        prompt_template_number = config.get('experiment', {}).get('prompt_template_number', 11)
        return train_svr_cv(X_train, y_train, 
                           n_splits=training_config.get('cv_splits', 5),
                           prompt_template_number=prompt_template_number,
                           kernel=training_config.get('svr_kernel', 'linear'),
                           C=training_config.get('svr_c', 2))
    else:
        raise ValueError(f"Unknown method: {method}")

# ---------------------------- Plotting Functions ---------------------------- #

def plot_r2_trends_across_models(r2_scores_dict: Dict[str, List[float]], models: List[ModelConfig], label_column: str, output_dir: str = 'Results/non_matching', config: Dict[str, Any] = None):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 3))
    sns.set(style="whitegrid")

    # Get plotting configuration
    plotting_config = config.get('plotting', {}) if config else {}
    color_map = plotting_config.get('color_map', {})
    style_map = plotting_config.get('style_map', {})
    default_color = plotting_config.get('default_color', 'gray')
    default_style = plotting_config.get('default_style', '-')

    legend_handles = []

    for model in models:
        r2_scores = r2_scores_dict[model.name]
        num_layers = model.num_layers
        if not r2_scores:
            print(f"Warning: No R² scores found for model {model.name}. Skipping.")
            continue
        normalized_layers = [layer / num_layers for layer in range(num_layers)]
        if len(normalized_layers) != len(r2_scores):
            print(f"Error: mismatch for {model.name}. Skipping.")
            continue

        color = color_map.get(model.name, default_color)
        line_style = style_map.get(model.name, default_style)
        line, = plt.plot(
            normalized_layers, r2_scores,
            marker='o', linestyle=line_style, color=color,
            label=model.name, markersize=3, linewidth=1.5
        )
        legend_handles.append(line)

        best_layer_index = r2_scores.index(max(r2_scores))
        best_layer_normalized = best_layer_index / num_layers
        best_layer_line = plt.axvline(
            x=best_layer_normalized, color=color,
            linestyle=':', linewidth=1.5, label=f"{model.name} Best Layer"
        )
        legend_handles.append(best_layer_line)

    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.ylim(0, 1)
    plt.xlabel('Layer Depth Proportion', fontsize=10)
    plt.ylabel('R² Score', fontsize=10)
    # plt.title(f'{label_column}', fontsize=10)
    plt.grid(True, which='major', linestyle='--', linewidth=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/r2_trends_comparison_normalized_{label_column}.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig_legend = plt.figure(figsize=(12, 1))
    plt.figlegend(handles=legend_handles, loc='center', fontsize=12, ncol=len(legend_handles)//6)
    fig_legend.savefig(f'{output_dir}/r2_trends_legend_{label_column}.png', dpi=300, bbox_inches='tight')
    plt.close(fig_legend)

# ---------------------------- Main Orchestration Function ---------------------------- #

def main(config: Dict[str, Any], models: List[ModelConfig] = None, methods: List[str] = None, split_method: str = None, label_column: str = None):
    # Use config values or defaults
    experiment_config = config.get('experiment', {})
    output_config = config.get('output', {})
    
    if models is None:
        models = get_enabled_models(config)
    if methods is None:
        methods = experiment_config.get('regression_methods', ['svr_cv'])
    if split_method is None:
        split_method = experiment_config.get('split_method', 'middle')
    if label_column is None:
        label_column = experiment_config.get('label_column', 'Group')
    
    dataset_file = experiment_config.get('dataset_file', 'periodic_table_dataset.csv')
    prompt_template_number = experiment_config.get('prompt_template_number', 11)
    output_dir = output_config.get('results_dir', 'Results/non_matching')
    random_state = config.get('training', {}).get('cv_random_state', 100)
    
    labels_repeated = load_data(dataset_file, label_column, prompt_template_number)
    for model in models:
        print(f"\nProcessing Model: {model.name}")
        r2_scores_dict[model.name] = []
        predictions_dict[model.name] = {}
        for layer in range(model.num_layers):
            try:
                X_train, X_test, y_train, y_test = load_activation_data(
                    layer=layer,
                    activation_path_template=model.activation_path_template,
                    labels_repeated=labels_repeated,
                    split_method=split_method,
                    prompt_template_number=prompt_template_number,
                    random_state=random_state
                )
            except FileNotFoundError as e:
                print(e)
                continue
            except ValueError as e:
                print(e)
                continue

            for method in methods:
                print(f"Training with method: {method} for model: {model.name}, layer: {layer}")
                if method == 'svr_cv':
                    r2, y_pred = train_model(X_train, X_test, y_train, y_train, method=method, config=config)
                    y_test_to_store = y_train
                else:
                    r2, y_pred = train_model(X_train, X_test, y_train, y_test, method=method, config=config)
                    y_test_to_store = y_test
                r2_scores_dict[model.name].append(r2)
                predictions_dict[model.name][layer] = {
                    'y_test': y_test_to_store,
                    'y_pred': y_pred
                }
        if r2_scores_dict[model.name]:
            best_layer = np.argmax(r2_scores_dict[model.name])
            best_r2 = np.max(r2_scores_dict[model.name])
            print(f"Model: {model.name}, Best R² at layer {best_layer}, Score: {best_r2:.4f}")
        else:
            print(f"No valid R² for model: {model.name}")

    plot_r2_trends_across_models(r2_scores_dict, models, label_column, output_dir, config)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run indirect recall experiments with configurable parameters')
    parser.add_argument('--config', '-c', type=str, default='config_indirect.yaml',
                       help='Path to configuration file (default: config_indirect.yaml)')
    parser.add_argument('--models', '-m', type=str, nargs='*',
                       help='Specific model names to run (if not specified, runs all enabled models from config)')
    parser.add_argument('--methods', type=str, nargs='*',
                       help='Regression methods to use (overrides config)')
    parser.add_argument('--split-method', type=str,
                       help='Data splitting method (overrides config)')
    parser.add_argument('--label-column', type=str,
                       help='Label column name (overrides config)')
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset file (overrides config)')
    parser.add_argument('--output-dir', type=str,
                       help='Output directory for results (overrides config)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load configuration
    try:
        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Please create a configuration file or specify a valid path.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        exit(1)
    
    # Override config with command line arguments if provided
    if args.dataset:
        config.setdefault('experiment', {})['dataset_file'] = args.dataset
    if args.output_dir:
        config.setdefault('output', {})['results_dir'] = args.output_dir
    
    # Get models to run
    if args.models:
        # Filter models by specified names
        all_models = get_enabled_models(config)
        models_to_run = [model for model in all_models if model.name in args.models]
        if not models_to_run:
            print(f"No models found matching: {args.models}")
            print(f"Available models: {[model.name for model in all_models]}")
            exit(1)
    else:
        models_to_run = get_enabled_models(config)
    
    if not models_to_run:
        print("No models enabled in configuration. Please enable at least one model.")
        exit(1)
    
    print(f"Running experiments with {len(models_to_run)} model(s):")
    for model in models_to_run:
        print(f"  - {model.name}")
    
    # Run main experiment
    main(
        config=config,
        models=models_to_run,
        methods=args.methods,
        split_method=args.split_method,
        label_column=args.label_column
    )