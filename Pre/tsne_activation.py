import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import numpy as np
import re

# ============================
# User Configuration Variables
# ============================

# Paths
ACTIVATIONS_DIR = 'activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number/'
CSV_PATH = 'periodic_table_dataset.csv'
OUTPUT_DIR = 'Results/results_tsne_plots'
GIF_OUTPUT = 'tsne_visualization.gif'

# Visualization Settings
NUM_SYMBOLS = 50  # Number of elements to load
ACTIVATIONS_PER_SYMBOL = 11  # Number of activations per element
FEATURES_TO_USE = ['Atomic Number', 'Group',"Period","Category"]  # Features to visualize
SELECTED_LAYERS = [0,49,79]  # Specify list of layer numbers to include, e.g., [33]. Set to None to include all layers.

# Plot Settings
POINT_SIZE = 22  # Reduced point size
ANNOTATE_FONT_SIZE = 22
COLOR_MAP = plt.cm.rainbow  # Colormap for continuous features
FPS_GIF = 2  # Frames per second for GIF

# ============================
# Function Definitions
# ============================

def load_symbols_and_features(csv_path, num_symbols=50, activations_per_symbol=11, features=[None]):
    """
    Load the first num_symbols elements and the specified features from the CSV file.
    Each element has multiple activations, so features are repeated accordingly.
    """
    df = pd.read_csv(csv_path)
    symbols = df['Symbol'].head(num_symbols).tolist()

    # Repeat each symbol according to activations_per_symbol
    symbols_repeated = []
    for symbol in symbols:
        symbols_repeated.extend([symbol] * activations_per_symbol)

    features_dict = {}
    for feature in features:
        if feature is not None:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' not found in CSV columns.")
            # Repeat each feature value according to activations_per_symbol
            feature_values = df[feature].head(num_symbols).tolist()
            features_dict[feature] = []
            for value in feature_values:
                features_dict[feature].extend([value] * activations_per_symbol)
        else:
            # No feature, fill with None
            features_dict[feature] = [None] * (num_symbols * activations_per_symbol)

    # Verify that all features have the correct length
    expected_length = num_symbols * activations_per_symbol
    for feature, values in features_dict.items():
        if len(values) != expected_length:
            raise ValueError(f"Feature '{feature}' has {len(values)} values, expected {expected_length}.")

    return symbols_repeated, features_dict

def get_layer_files(directory):
    """
    Get a sorted list of layer files from the specified directory.
    Sorting is based on the layer number extracted from the filename.
    Expected filename format: 'atomic number.last.{number}_templates.{layer}.pt'
    """
    files = []
    pattern = re.compile(r'atomic number\.last\.\d+_templates\.(\d+)\.pt')
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            layer_num = int(match.group(1))
            files.append((layer_num, filename))
    # Sort files based on layer number
    sorted_files = sorted(files, key=lambda x: x[0])
    return [filename for _, filename in sorted_files]

def load_activations(file_path, num_symbols=50, activations_per_symbol=11):
    """
    Load activations from a .pt file and convert to numpy array.
    The tensor is moved to the CPU before converting.
    Assumes that activations are ordered by element, with activations_per_symbol per element.
    """
    tensor = torch.load(file_path)
    if isinstance(tensor, torch.Tensor):
        activations = tensor.cpu().numpy()
    elif isinstance(tensor, dict):
        if 'activations' in tensor:
            activations = tensor['activations'].cpu().numpy()
        else:
            raise KeyError("Key 'activations' not found in the tensor dictionary.")
    else:
        raise ValueError(f"Unsupported tensor type in file {file_path}")

    expected_shape = num_symbols * activations_per_symbol
    if activations.shape[0] != expected_shape:
        raise ValueError(f"Expected {expected_shape} activations, but got {activations.shape[0]} in {file_path}")

    return activations

def perform_pca(data, n_components=50):
    """
    Perform PCA on the data.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_result = pca.fit_transform(data)
    return pca_result, pca

def perform_tsne(data, n_components=2):
    """
    Perform t-SNE on the data.
    """
    tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(data)
    return tsne_result

def assign_colors_to_categories(categories):
    """
    Assign a distinct color for each unique category using the 'rainbow' colormap.
    Ensures color consistency across plots.
    """
    unique_categories = sorted(list(set(categories)))  # Sort for consistency
    num_categories = len(unique_categories)

    # Generate a discrete colormap for the categories using 'rainbow'
    colors = plt.cm.rainbow(np.linspace(0, 1, num_categories))
    category_to_color = {category: colors[i] for i, category in enumerate(unique_categories)}

    # Assign a color to each category based on the mapping
    category_colors = [category_to_color.get(cat, (0.5, 0.5, 0.5, 1.0)) for cat in categories]  # Default to gray if category not found
    return category_colors, unique_categories

def plot_tsne_with_metrics(tsne_data, pca_data, labels, features_dict, title, output_path, annotate=False):
    num_features = len(features_dict)
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 6), squeeze=False)

    for idx, (feature, values) in enumerate(features_dict.items()):
        ax = axes[0, idx]
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        is_categorical = isinstance(values[0], str) or isinstance(values[0], bool)
        metric_text = ""

        if is_categorical:
            category_colors, unique_categories = assign_colors_to_categories(values)
            scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                                 s=POINT_SIZE, c=category_colors, marker='o', alpha=0.7)

            if annotate:
                symbols = list(set(labels))
                for symbol in symbols:
                    indices = [i for i, lbl in enumerate(labels) if lbl == symbol]
                    centroid = tsne_data[indices].mean(axis=0)

            # Remove legend title
            handles = [
                plt.Line2D([0], [0], marker='o', color='w', label=cat,
                           markerfacecolor=c, markersize=10)
                for cat, c in zip(unique_categories, plt.cm.rainbow(np.linspace(0, 1, len(unique_categories))))
            ]
            ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12, title="")

            try:
                labels_numeric = pd.factorize(values)[0]
                if len(set(labels_numeric)) > 1:
                    silhouette_avg = silhouette_score(tsne_data, labels_numeric)
                    metric_text = f"Silhouette Score: {silhouette_avg:.2f}"
                else:
                    metric_text = "Silhouette Score: N/A"
            except:
                metric_text = "Silhouette Score: N/A"

        elif feature is not None:
            values_array = np.array(values)
            mask = ~pd.isnull(values_array)
            num_missing = np.sum(~mask)
            num_present = np.sum(mask)

            if num_present < 2:
                scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                                     s=POINT_SIZE, c='gray', marker='o', alpha=0.7)
            else:
                scatter_present = ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                                             s=POINT_SIZE, c=values_array[mask], cmap=COLOR_MAP,
                                             norm=plt.Normalize(np.nanmin(values_array[mask]), np.nanmax(values_array[mask])),
                                             marker='o', alpha=0.7)
                ax.scatter(tsne_data[~mask, 0], tsne_data[~mask, 1],
                           s=POINT_SIZE, c='gray', marker='x', alpha=0.7)

                if annotate:
                    symbols = list(set(labels))
                    for symbol in symbols:
                        indices = [i for i, lbl in enumerate(labels) if lbl == symbol]
                        centroid = tsne_data[indices].mean(axis=0)

                # Horizontal colorbar on top, integer ticks
                cbar = fig.colorbar(scatter_present, ax=ax, orientation='horizontal', pad=0.1)
                cbar.set_label(feature if feature else '', fontsize=14)
                cbar.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

        else:
            scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                                 s=POINT_SIZE, c='gray', marker='o', alpha=0.7)
            if annotate:
                symbols = list(set(labels))
                for symbol in symbols:
                    indices = [i for i, lbl in enumerate(labels) if lbl == symbol]
                    centroid = tsne_data[indices].mean(axis=0)

        if metric_text:
            ax.text(0.95, 0.05, metric_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

# ============================
# Main Function
# ============================

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load symbols and selected features
    try:
        symbols, features_dict = load_symbols_and_features(
            CSV_PATH, 
            num_symbols=NUM_SYMBOLS, 
            activations_per_symbol=ACTIVATIONS_PER_SYMBOL, 
            features=FEATURES_TO_USE
        )
    except ValueError as e:
        print(f"Error loading symbols and features: {e}")
        return

    expected_activations = NUM_SYMBOLS * ACTIVATIONS_PER_SYMBOL
    if len(symbols) < expected_activations:
        print(f"Warning: Less than {expected_activations} activations found based on the CSV file.")

    # Get sorted layer files
    layer_files = get_layer_files(ACTIVATIONS_DIR)
    if not layer_files:
        print("No layer files found in the specified directory.")
        return

    # Filter layers if SELECTED_LAYERS is specified
    if SELECTED_LAYERS is not None:
        layer_files = [f for f in layer_files if int(re.search(r'atomic number\.last\.\d+_templates\.(\d+)\.pt', f).group(1)) in SELECTED_LAYERS]
        if not layer_files:
            print("No matching layer files found for the selected layers.")
            return

    total_layers = len(layer_files)
    annotate_threshold = int(total_layers * 0.5)  # 50% of total layers

    # Iterate through each layer file with corrected layer numbering
    for idx, filename in enumerate(layer_files):
        # Extract layer number from filename and add 1
        match = re.search(r'atomic number\.last\.\d+_templates\.(\d+)\.pt', filename)
        if not match:
            print(f"Filename {filename} does not match the expected pattern.")
            continue
        original_layer_num = int(match.group(1))
        layer_num = original_layer_num + 1  # Corrected layer number
        file_path = os.path.join(ACTIVATIONS_DIR, filename)
        print(f"Processing {filename} (Layer {layer_num})")

        # Load activations
        try:
            activations = load_activations(file_path, num_symbols=NUM_SYMBOLS, activations_per_symbol=ACTIVATIONS_PER_SYMBOL)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

        # Perform PCA
        try:
            pca_result, pca = perform_pca(activations, n_components=min(50, activations.shape[1]))
            explained_variance = pca.explained_variance_ratio_.sum()
            print(f"PCA explained variance ratio for Layer {layer_num}: {explained_variance:.2f}")
        except Exception as e:
            print(f"Error performing PCA on Layer {layer_num}: {e}")
            continue

        # Perform t-SNE on PCA-transformed data for visualization
        try:
            tsne_result = perform_tsne(pca_result, n_components=2)
        except Exception as e:
            print(f"Error performing t-SNE on Layer {layer_num}: {e}")
            continue

        # Determine whether to annotate based on layer position
        annotate = idx >= annotate_threshold

        # Plot t-SNE for each feature with metrics
        title = f"t-SNE Visualization of Layer {layer_num} Activations"
        output_path = os.path.join(OUTPUT_DIR, f"layer_{layer_num}_tsne.png")
        try:
            plot_tsne_with_metrics(
                tsne_data=tsne_result, 
                pca_data=pca_result,  # Pass PCA data here
                labels=symbols, 
                features_dict=features_dict, 
                title=title, 
                output_path=output_path,
                annotate=annotate
            )
            print(f"Saved t-SNE plot to {output_path}")
        except Exception as e:
            print(f"Error plotting t-SNE for Layer {layer_num}: {e}")

if __name__ == "__main__":
    main()
