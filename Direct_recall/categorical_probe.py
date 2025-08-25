import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

# ---------------------------- Global Variables ---------------------------- #

prompt_template_number = 11
predictions_dict = {}  # Structure: model name -> layer -> {'y_test': ..., 'y_pred': ...}
accuracy_scores_dict = {}    # Structure: model name -> [accuracy per layer]

# Define model configuration using a dictionary, not a class
models = [
    {
        'name': 'Llama-2-7b-hf',
        'activation_path_template': 'activation_datasets/meta-llama-Llama-2-7b-hf/group/group.last.11_templates.{layer}.pt',
        'num_layers': 32
    },
    {
        'name': 'Llama-3.1-8B',
        'activation_path_template': 'activation_datasets/meta-llama-Llama-3.1-8B/group/group.last.11_templates.{layer}.pt',
        'num_layers': 32
    },
    {
        'name': 'Meta-Llama-3.1-70B',
        'activation_path_template': 'activation_datasets/meta-llama-Meta-Llama-3.1-70B/group/group.last.11_templates.{layer}.pt',
        'num_layers': 80
    }
]




# ---------------------------- Data Loading and Preprocessing ---------------------------- #

def load_data(file_path: str, label_column: str = 'Group') -> np.ndarray:
    """
    Load data and repeat labels using prompt_template_number.
    If the label column is numeric, keep it as is; otherwise, convert it to strings.
    """
    df = pd.read_csv(file_path)
    print(f"Loaded dataset with columns: {df.columns.tolist()}")
    
    # Check if the label column is numeric
    if pd.api.types.is_numeric_dtype(df[label_column]):
        # If numeric, fill missing values with a specific number (e.g., -9999, adjust as needed)
        labels = df[label_column].fillna(-9999).values
    else:
        df[label_column] = df[label_column].fillna('Unknown').astype(str)
        labels = df[label_column].values
        
    # Repeat labels (assuming each sample corresponds to multiple templates)
    labels = np.repeat(labels, prompt_template_number)
    return labels

def load_activation_data(layer: int, activation_path_template: str, labels_repeated: np.ndarray):
    """
    Load activation data for the specified layer.
    This function only loads data; train/test splitting will be done later in cross-validation.
    """
    file_path = activation_path_template.format(layer=layer)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Activation file not found: {file_path}")
    
    # weights_only=True assumes loading only weights, move to CPU and convert to numpy array
    activation_data = torch.load(file_path, weights_only=True).cpu().numpy()
    if activation_data.shape[0] != len(labels_repeated):
        raise ValueError(f"Inconsistent samples: {activation_data.shape[0]} features vs {len(labels_repeated)} labels.")
    
    return activation_data

# ---------------------------- Classification Probe Cross-Validation ---------------------------- #

def categorical_probe_cv(X, y, n_splits=5):
    """
    Perform cross-validation using a classification probe and return the average accuracy 
    and predictions for all test samples.
    Use GroupKFold to ensure the same group (based on template repetition) does not appear 
    in both training and testing.
    """
    groups = np.repeat(np.arange(len(y) // prompt_template_number), prompt_template_number)
    gkf = GroupKFold(n_splits=n_splits)
    accuracies = []
    y_pred_all = np.empty_like(y, dtype=int)  # Ensure predictions are integers
    
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        
        # Compute sample weights to address class imbalance
        sample_weights = compute_sample_weight('balanced', y[train_idx])
        
        # Use SVC for classification probe (linear kernel, C=2, class_weight set to balanced)
        model = SVC(kernel='linear', C=2, class_weight='balanced')
        model.fit(X_train, y[train_idx], sample_weight=sample_weights)
        y_pred = model.predict(X_test)
        
        accuracies.append(accuracy_score(y[test_idx], y_pred))
        y_pred_all[test_idx] = y_pred
        
    return np.mean(accuracies), y_pred_all

# ---------------------------- Plotting Functions ---------------------------- #

def plot_accuracy_trends_across_models(accuracy_scores_dict, models, label_column):
    output_dir = 'Results/element_token_accuracy_trends_basic'
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(5, 3))
    sns.set(style="whitegrid")
    colors = sns.color_palette("husl", len(models))
    for i, m in enumerate(models):
        scores = accuracy_scores_dict[m['name']]
        num_layers = m['num_layers']
        if not scores:
            print(f"Warning: No scores for {m['name']}.")
            continue
        normalized_layers = [layer / num_layers for layer in range(num_layers)]
        plt.plot(normalized_layers, scores, marker='o', linestyle='-', color=colors[i],
                 label=m['name'], markersize=3, linewidth=1)
        best_layer_norm = scores.index(max(scores)) / num_layers
        plt.axvline(x=best_layer_norm, color=colors[i], linestyle='--', linewidth=1.5,
                    label=f"{m['name']} Best Layer")
    plt.xlabel('Layer Depth Proportion', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'{label_column}', fontsize=13)
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', linewidth=0.7)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_trends_comparison_normalized_{label_column}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_results(accuracy_scores, predictions, best_layer, label_column, activation_filename, model_name, model_num_layers, le):
    """
    Plot accuracy trends for the model across layers, and confusion matrix and classification report for the best layer.
    """
    y_true = predictions[best_layer]['y_test']
    y_pred = predictions[best_layer]['y_pred']
    output_dir = 'Results/accuracy_r2_trends_basic'
    os.makedirs(output_dir, exist_ok=True)
    base_filename = f'{label_column}_{activation_filename}_layer_{best_layer}_model_{model_name}'
    
    # Plot accuracy trends
    plt.figure(figsize=(5,3))
    plt.plot(range(model_num_layers), accuracy_scores, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.axvline(best_layer, color='r', linestyle='--', label=f'Max at Layer {best_layer}')
    plt.ylim(0, 1)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Accuracy Trend - {model_name}', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/{base_filename}_accuracy_trend.png')
    plt.show()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    # Determine class names based on whether le exists
    if le is not None:
        class_names = le.classes_
    else:
        # If labels are numeric, convert sorted unique values to strings
        class_names = [str(x) for x in sorted(np.unique(y_true))]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name} Layer {best_layer}')
    plt.savefig(f'{output_dir}/{base_filename}_confusion_matrix.png')
    plt.show()
    
    # Output classification report
    if le is not None:
        target_names = le.classes_
    else:
        target_names = [str(x) for x in sorted(np.unique(y_true))]
    print(f"Classification Report for {model_name}, Layer {best_layer}:")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f"Saved results for {model_name}, layer {best_layer}")

# ---------------------------- Main Process ---------------------------- #

def main():
    # You can switch between label_column='Category' or label_column='Group' as needed
    # If Group is numeric, keep it as is; otherwise, keep it as a string
    label_col = 'Period'  # Or 'Category'
    
    # 1. Load label data
    labels_repeated = load_data('periodic_table_dataset.csv', label_column=label_col)
    
    # 2. If labels are numeric, do not encode; otherwise, encode for classifier training
    if np.issubdtype(labels_repeated.dtype, np.number):
        labels_repeated_encoded = labels_repeated
        le = None
    else:
        le = LabelEncoder()
        labels_repeated_encoded = le.fit_transform(labels_repeated)
    
    # Initialize dictionaries
    for m in models:
        model_name = m['name']
        print(f"\nProcessing Model: {model_name}")
        accuracy_scores_dict[model_name] = []
        predictions_dict[model_name] = {}
        activation_filename = 'last.11_templates'
        
        # Train probe for each layer
        for layer in range(m['num_layers']):
            try:
                # Load activation data for the layer, perform cross-validation later
                X = load_activation_data(
                    layer=layer,
                    activation_path_template=m['activation_path_template'],
                    labels_repeated=labels_repeated  # Use original labels to match sample count
                )
            except (FileNotFoundError, ValueError) as e:
                print(e)
                continue

            # Perform cross-validation using encoded labels
            accuracy, y_pred = categorical_probe_cv(X, labels_repeated_encoded, n_splits=5)
            accuracy_scores_dict[model_name].append(accuracy)
            predictions_dict[model_name][layer] = {
                'y_test': labels_repeated_encoded,  # Use encoded labels as ground truth
                'y_pred': y_pred
            }
            print(f"Model: {model_name}, Layer: {layer}, Accuracy: {accuracy:.4f}")
        
        # Find the best layer (highest accuracy)
        if accuracy_scores_dict[model_name]:
            best_layer = np.argmax(accuracy_scores_dict[model_name])
            best_acc = np.max(accuracy_scores_dict[model_name])
            print(f"{model_name} best layer: {best_layer} with Accuracy: {best_acc:.4f}")
            plot_results(
                accuracy_scores=accuracy_scores_dict[model_name],
                predictions=predictions_dict[model_name],
                best_layer=best_layer,
                label_column=label_col,  # Use actual column name
                activation_filename=activation_filename,
                model_name=model_name,
                model_num_layers=m['num_layers'],
                le=le  # If le is None, use original numeric labels
            )
            # Add mid-layer confusion matrix
            mid_layer = m['num_layers'] // 2
            if mid_layer in predictions_dict[model_name]:
                plot_results(
                    accuracy_scores=accuracy_scores_dict[model_name],
                    predictions=predictions_dict[model_name],
                    best_layer=mid_layer,
                    label_column=label_col,
                    activation_filename=activation_filename,
                    model_name=model_name,
                    model_num_layers=m['num_layers'],
                    le=le
                )
        else:
            print(f"No valid scores for {model_name}.")

    # Plot accuracy trends for all models
    plot_accuracy_trends_across_models(accuracy_scores_dict, models, label_column=label_col)

if __name__ == "__main__":
    main()