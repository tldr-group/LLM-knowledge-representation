
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

MODEL_CONFIGS: List[ModelConfig] = [
    # ModelConfig(
    #     name='Llama-2-7b-hf (continuation prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/group/group.last.11_templates.{layer}.pt',
    #     num_layers=32
    # ),
    # ModelConfig(
    #     name='Llama-2-7b-hf (question prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/group question/group question.last.11_templates_questions.{layer}.pt',
    #     num_layers=32
    # ),
    # ModelConfig(
    #     name='Llama-3.1-8B (continuation prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/group/group.last.11_templates.{layer}.pt',
    #     num_layers=32
    # ),
    # ModelConfig(
    #     name='Llama-3.1-8B (question prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/group question/group question.last.11_templates_questions.{layer}.pt',
    #     num_layers=32
    # ),
    # ModelConfig(
    #     name='Meta-Llama-3.1-70B (continuation prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group/group.last.11_templates.{layer}.pt',
    #     num_layers=80
    # ),
    # ModelConfig(
    #     name='Meta-Llama-3.1-70B (question prompt)',
    #     activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group question/group question.last.11_templates_questions.{layer}.pt',
    #     num_layers=80
    # ),
    ModelConfig(
        name='Llama-2-7b-hf',
        activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/group/group.last.11_templates.{layer}.pt',
        num_layers=32
    ),
    ModelConfig(
        name='Llama-2-7b-hf (Non-matching prompt)',
        activation_path_template='activation_datasets/meta-llama-Llama-2-7b-hf/atomic number/atomic number.last.11_templates.{layer}.pt',
        num_layers=32
    ),
    ModelConfig(
        name='Llama-3.1-8B',
        activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/group/group.last.11_templates.{layer}.pt',
        num_layers=32
    ),
    ModelConfig(
        name='Llama-3.1-8B (Non-matching prompt)',
        activation_path_template='activation_datasets/meta-llama-Llama-3.1-8B/atomic number/atomic number.last.11_templates.{layer}.pt',
        num_layers=32
    ),
    ModelConfig(
        name='Meta-Llama-3.1-70B',
        activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/group/group.last.11_templates.{layer}.pt',
        num_layers=80
    ),
    ModelConfig(
        name='Meta-Llama-3.1-70B (Non-matching prompt)',
        activation_path_template='activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number/atomic number.last.11_templates.{layer}.pt',
        num_layers=80
    ),
]

# ---------------------------- Global Variables ---------------------------- #

prompt_template_number = 11
predictions_dict: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
r2_scores_dict: Dict[str, List[float]] = {}

# ---------------------------- Data Loading and Splitting ---------------------------- #

def load_data(file_path: str, label_column: str = 'Group') -> np.ndarray:
    periodic_table = pd.read_csv(file_path)
    print(f"Loaded dataset with columns: {periodic_table.columns.tolist()}")
    labels = periodic_table[label_column].fillna(-np.inf).astype(float).values
    labels_repeated = np.repeat(labels, prompt_template_number)
    return labels_repeated

def split_data_middle_group(labels_repeated: np.ndarray) -> (List[int], List[int]):
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
    train_indices = []
    test_indices = []
    for label in np.unique(labels_repeated):
        label_indices = np.where(labels_repeated == label)[0]
        test_indices.extend(label_indices[:prompt_template_number])
        train_indices.extend(label_indices[prompt_template_number:])
    return train_indices, test_indices

def split_data_group_shuffle(labels_repeated: np.ndarray) -> (List[int], List[int]):
    valid_indices = np.isfinite(labels_repeated)
    valid_labels = labels_repeated[valid_indices]
    groups = np.repeat(np.arange(len(valid_labels) // prompt_template_number), prompt_template_number)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=100)
    train_idx, test_idx = next(gss.split(np.arange(len(valid_labels)), groups=groups))
    train_indices = np.where(valid_indices)[0][train_idx]
    test_indices = np.where(valid_indices)[0][test_idx]
    return train_indices.tolist(), test_indices.tolist()

def split_data(labels_repeated: np.ndarray, method: str) -> (List[int], List[int]):
    if method == 'middle':
        return split_data_middle_group(labels_repeated)
    elif method == 'first':
        return split_data_first_group(labels_repeated)
    elif method == 'group_shuffle':
        return split_data_group_shuffle(labels_repeated)
    else:
        raise ValueError(f"Unknown split method: {method}")

def load_activation_data(layer: int, activation_path_template: str, labels_repeated: np.ndarray, split_method: str = 'middle') -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
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

def train_svr(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> (float, np.ndarray):
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
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)
    return r2_rf, y_pred_rf

def train_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, method: str = 'svr') -> (float, np.ndarray):
    if method == 'svr':
        return train_svr(X_train, X_test, y_train, y_test)
    elif method == 'random_forest':
        return train_random_forest(X_train, X_test, y_train, y_test)
    elif method == 'svr_cv':
        return train_svr_cv(X_train, y_train)
    else:
        raise ValueError(f"Unknown method: {method}")

# ---------------------------- Plotting Functions ---------------------------- #

# COLOR_MAP = {
#     'Llama-2-7b-hf (question prompt)': '#e377c2',         
#     'Llama-2-7b-hf (continuation prompt)': '#f7c9e2',     
#     'Llama-3.1-8B (question prompt)': '#2ca02c',           
#     'Llama-3.1-8B (continuation prompt)': '#98df8a',       
#     'Meta-Llama-3.1-70B (question prompt)': '#1f77b4',     
#     'Meta-Llama-3.1-70B (continuation prompt)': '#aec7e8'  
# }

# STYLE_MAP = {
#     'Llama-2-7b-hf': '-',
#     'Llama-2-7b-hf (question prompt)': '-',
#     'Llama-3.1-8B (continuation prompt)': '-',
#     'Llama-3.1-8B (question prompt)': '-',
#     'Meta-Llama-3.1-70B (continuation prompt)': '-',
#     'Meta-Llama-3.1-70B (question prompt)': '-'
# }

COLOR_MAP = {
    'Llama-2-7b-hf (Non-matching prompt)': '#e377c2',     
    'Llama-2-7b-hf': '#f7c9e2',    
    'Llama-3.1-8B (Non-matching prompt)': '#2ca02c',         
    'Llama-3.1-8B': '#98df8a',      
    'Meta-Llama-3.1-70B (Non-matching prompt)': '#1f77b4',     
    'Meta-Llama-3.1-70B': '#aec7e8'  
}

STYLE_MAP = {
    'Llama-2-7b-hf': '-',
    'Llama-2-7b-hf (Non-matching prompt)': '-',
    'Llama-3.1-8B': '-',
    'Llama-3.1-8B (Non-matching prompt)': '-',
    'Meta-Llama-3.1-70B ': '-',
    'Meta-Llama-3.1-70B (Non-matching prompt)': '-'
}

def plot_r2_trends_across_models(r2_scores_dict: Dict[str, List[float]], models: List[ModelConfig], label_column: str):
    output_dir = 'Results/non_matching'
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 3))
    sns.set(style="whitegrid")

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

        color = COLOR_MAP.get(model.name, 'gray')
        line_style = STYLE_MAP.get(model.name, '-')
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

def main(models: List[ModelConfig], methods: List[str], split_method: str = 'middle', label_column: str = 'Group'):
    labels_repeated = load_data('periodic_table_dataset.csv', label_column)
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
                if method == 'svr_cv':
                    r2, y_pred = train_svr_cv(X_train, y_train)
                    y_test_to_store = y_train
                else:
                    r2, y_pred = train_model(X_train, X_test, y_train, y_test, method=method)
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

    plot_r2_trends_across_models(r2_scores_dict, models, label_column)

if __name__ == "__main__":
    models_to_compare = MODEL_CONFIGS
    regression_methods = ["svr_cv"]
    main(
        models=models_to_compare,
        methods=regression_methods,
        split_method='group_shuffle',
        label_column='Group'
    )