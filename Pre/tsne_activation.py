import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import numpy as np
import imageio
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
# FEATURES_TO_USE = [None, 'Atomic Number', 'Group', 'Electronegativity',"Atomic Mass","Period","Category2"]  # Features to visualize
FEATURES_TO_USE = [None, 'Atomic Number', 'Group',"Period","Category"]  # Features to visualize
# FEATURES_TO_USE = [None, 'weekday']  # Features to visualize
# SELECTED_LAYERS = [0,19,49,79]  # Specify list of layer numbers to include, e.g., [33]. Set to None to include all layers.
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
    """
    Plot the t-SNE visualization with labels, using distinct colors for categorical features
    and color gradients for continuous features. Computes and annotates Silhouette Score for
    categorical features.
    Generates subplots for each feature.

    Parameters:
    - tsne_data: np.ndarray, shape (n_samples, 2)
    - pca_data: np.ndarray, shape (n_samples, n_pca_components)
    - labels: list of str, length n_samples
    - features_dict: dict, keys are feature names, values are lists of feature values
    - title: str, title of the plot
    - output_path: str, path to save the plot
    - annotate: bool, whether to annotate element symbols
    """
    num_features = len(features_dict)
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 4.6), squeeze=False)

    for idx, (feature, values) in enumerate(features_dict.items()):
        ax = axes[0, idx]
        is_categorical = isinstance(values[0], str) or isinstance(values[0], bool)
        metric_text = ""

        if is_categorical:
            # Handle categorical features
            category_colors, unique_categories = assign_colors_to_categories(values)
            scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                                 s=POINT_SIZE, c=category_colors, marker='o', alpha=0.7)

            if annotate:
                # Annotate only once per element by calculating centroids
                symbols = list(set(labels))
                centroids = {}
                for symbol in symbols:
                    indices = [i for i, lbl in enumerate(labels) if lbl == symbol]
                    centroid = tsne_data[indices].mean(axis=0)
                    centroids[symbol] = centroid
                    # ax.annotate(symbol, (centroid[0], centroid[1]),
                    #             textcoords="offset points", xytext=(0, 5),
                    #             ha='center', fontsize=ANNOTATE_FONT_SIZE, fontweight='bold')

            # Create a legend for the categories
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=cat,
                                  markerfacecolor=category_to_color, markersize=10) 
                       for cat, category_to_color in zip(unique_categories, plt.cm.rainbow(np.linspace(0, 1, len(unique_categories))))]
            ax.legend(handles=handles, title=feature, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=22)

            # Compute Silhouette Score
            try:
                labels_numeric = pd.factorize(values)[0]
                if len(set(labels_numeric)) > 1:
                    silhouette_avg = silhouette_score(tsne_data, labels_numeric)
                    metric_text = f"Silhouette Score: {silhouette_avg:.2f}"
                else:
                    metric_text = "Silhouette Score: N/A"
            except Exception as e:
                metric_text = "Silhouette Score: N/A"

        elif feature is not None:
            # Handle continuous features
            values_array = np.array(values)
            # Identify non-missing values
            mask = ~pd.isnull(values_array)
            num_missing = np.sum(~mask)
            num_present = np.sum(mask)
            print(f"Feature '{feature}': {num_present} present, {num_missing} missing values.")

            if num_present < 2:
                print(f"Not enough data points with non-missing '{feature}' for metric computation.")
                scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                                     s=POINT_SIZE, c='gray', marker='o', alpha=0.7)
                metric_text = ""
            else:
                # Prepare color mapping, handling missing values
                # Assign a default color (e.g., gray) to missing values
                # Plot non-missing and missing separately
                scatter_present = ax.scatter(tsne_data[mask, 0], tsne_data[mask, 1],
                                             s=POINT_SIZE, c=values_array[mask], cmap=COLOR_MAP, 
                                             norm=plt.Normalize(np.nanmin(values_array[mask]), np.nanmax(values_array[mask])),
                                             marker='o', alpha=0.7, label='Present')
                scatter_missing = ax.scatter(tsne_data[~mask, 0], tsne_data[~mask, 1],
                                             s=POINT_SIZE, c='gray', marker='x', alpha=0.7, label='Missing')

                # To avoid clutter, annotate only once per element by calculating centroids
                if annotate:
                    symbols = list(set(labels))
                    centroids = {}
                    for symbol in symbols:
                        indices = [i for i, lbl in enumerate(labels) if lbl == symbol]
                        centroid = tsne_data[indices].mean(axis=0)
                        centroids[symbol] = centroid
                        # ax.annotate(symbol, (centroid[0], centroid[1]),
                        #             textcoords="offset points", xytext=(0, 5),
                        #             ha='center', fontsize=ANNOTATE_FONT_SIZE, fontweight='bold')

                # Create color bar for the present values
                # cbar = fig.colorbar(scatter_present, ax=ax, ticks=np.linspace(np.nanmin(values_array[mask]), np.nanmax(values_array[mask]), 6))
                
                # cbar.set_label(feature if feature else 'No Feature', fontsize=22)
                # cbar.ax.tick_params(labelsize=22)
                cbar = fig.colorbar(scatter_present, ax=ax, 
                    ticks=np.linspace(np.nanmin(values_array[mask]), np.nanmax(values_array[mask]), 6), 
                    pad=0.05)

                # 设置 color bar 的标签字体大小
                cbar.set_label(feature if feature else 'No Feature', fontsize=22)

                # 设置 color bar 的刻度数字字体大小
                cbar.ax.tick_params(labelsize=14)



        else:
            # Feature is None; plot without metrics
            scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1],
                                 s=POINT_SIZE, c='gray', marker='o', alpha=0.7)
            if annotate:
                # Annotate only once per element by calculating centroids
                symbols = list(set(labels))
                centroids = {}
                for symbol in symbols:
                    indices = [i for i, lbl in enumerate(labels) if lbl == symbol]
                    centroid = tsne_data[indices].mean(axis=0)
                    centroids[symbol] = centroid
                    # ax.annotate(symbol, (centroid[0], centroid[1]),
                    #             textcoords="offset points", xytext=(0, 5),
                    #             ha='center', fontsize=ANNOTATE_FONT_SIZE, fontweight='bold')
            metric_text = ""

        # Annotate the metric on the plot
        if metric_text:
            ax.text(0.95, 0.05, metric_text, transform=ax.transAxes,
        fontsize=18,  
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))


        # Title and labels
        # feature_title = feature if feature else 'No Attribute'
        # ax.set_title(f"Feature: {feature_title}", fontsize=14)
        ax.set_xlabel('t-SNE Dimension 1', fontsize=22)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=22)


    # Save and close the plot

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle(title, fontsize=22, y=1.0)
    plt.savefig(output_path, dpi=300)
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
