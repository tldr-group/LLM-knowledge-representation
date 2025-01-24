import numpy as np
import matplotlib.pyplot as plt
from math import acos, degrees, pi
import random
import os

def generate_random_vectors(n: int, num_vectors: int) -> np.ndarray:
    """
    Generates random vectors in n-dimensional space.
    """
    vectors = np.random.randn(num_vectors, n)
    vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]  # Normalize to unit length
    return vectors

def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): First unit vector.
        vec2 (np.ndarray): Second unit vector.

    Returns:
        float: Cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Clamp for numerical stability
    return dot_product

def sample_random_pairs(vectors: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Samples a fixed number of random pairs of vectors and computes the cosine similarities between them.

    Args:
        vectors (np.ndarray): Array of vectors.
        num_samples (int): Number of random pairs to sample.

    Returns:
        np.ndarray: Cosine similarities between sampled pairs.
    """
    num_vectors = vectors.shape[0]
    sampled_similarities = []

    for _ in range(num_samples):
        i, j = random.sample(range(num_vectors), 2)  # Randomly pick two distinct vectors
        similarity = calculate_cosine_similarity(vectors[i], vectors[j])
        sampled_similarities.append(similarity)

    return np.array(sampled_similarities)

def plot_multiple_cosine_similarity_distributions(n_values: list, num_vectors: int, num_samples: int, num_bins: int, output_dir: str):
    """
    Plots and saves multiple histograms of the cosine similarities between vectors for different dimensions (n values).

    Args:
        n_values (list): List of dimensionalities of the space.
        num_vectors (int): Number of random vectors to generate for each dimension.
        num_samples (int): Number of random pairs to sample for each dimension.
        num_bins (int): The number of bins for the histogram.
        output_dir (str): Directory to save the plot.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Set up the figure with subplots
    num_plots = len(n_values)
    plt.figure(figsize=(5 * num_plots, 5))  # Adjust size based on the number of plots

    # Loop through each dimensionality value in n_values
    for idx, n in enumerate(n_values):
        # Generate random vectors
        vectors = generate_random_vectors(n, num_vectors)
        
        # Sample random pairs and compute the cosine similarities
        similarities = sample_random_pairs(vectors, num_samples)
        
        # Calculate theoretical 99.9% confidence interval
        z_score = 3.291  # For 99.9% confidence
        sigma = 1 / np.sqrt(n)  # Standard deviation of the cosine similarity
        delta_similarity = z_score * sigma
        lower_bound = -delta_similarity
        upper_bound = delta_similarity

        # Create a subplot for each n value
        plt.subplot(1, num_plots, idx + 1)
        
        # Plot the histogram with a more appealing color palette
        hist_values, bins, _ = plt.hist(similarities, bins=num_bins, color='#3498db', edgecolor='white', alpha=0.9)
        
        # Add labels and title for each subplot
        plt.xlabel('Cosine Similarity', fontsize=16)
        plt.ylabel('Frequency', fontsize=16)
        plt.title(f'n = {n}', fontsize=18)
        plt.xlim(-0.1, 0.1)
        
        # Draw vertical lines for the theoretical confidence interval
        plt.axvline(x=lower_bound, color='#e74c3c', linestyle='--', linewidth=2, label=f'99.9% CI Lower ({lower_bound:.4f})')
        plt.axvline(x=upper_bound, color='#2ecc71', linestyle='--', linewidth=2, label=f'99.9% CI Upper ({upper_bound:.4f})')
        
        # Add legend in each subplot
        plt.legend(fontsize=12)
        
    # Add a global title for the entire figure
    plt.suptitle('Distribution of Cosine Similarities Between Randomly Sampled Vectors in n-Dimensional Space', fontsize=16)

    # Adjust layout and save the final plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout to make space for the suptitle
    file_name = 'multiple_cosine_similarity_distributions.png'
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path, dpi=300)

    # Show the final plot with all subplots
    plt.show()

    print(f"Plot saved at: {file_path}")

# ------------------ Parameters ------------------ #
# n_values = [4096, 8192]  # List of different dimensionalities
n_values = [8192]
num_vectors = 1000       # Number of random vectors to generate
num_samples = 100000     # Number of random pairs to sample
num_bins = 50            # Number of bins for the histogram

# Output directory for saving the plot
output_dir = 'Results_cosine_similarity/random'

# ------------------ Main Logic ------------------ #
# Plot and save cosine similarity distributions for different dimensionalities (n_values)
plot_multiple_cosine_similarity_distributions(n_values, num_vectors, num_samples, num_bins, output_dir)