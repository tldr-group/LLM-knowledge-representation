# python

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --------------------------
# 1. Configuration
# --------------------------
NUM_LAYERS = 80
START_LAYER = 1
NUM_FOLDS = 5
RESULT_CSV = "model_results.csv"
OUTPUT_PLOT = "Results/relationship/model_results_plot.png"
PCA_COMPONENTS_X = 20  

DATA_PAIRS = {
    "atomic number - period": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/period_relationship/period_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number_relationship/atomic number_relationship.last.1_templates.layer_{}.pt"
    },
    "atomic number - group": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/group_relationship/group_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number_relationship/atomic number_relationship.last.1_templates.layer_{}.pt"
    },
    "period - group": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/group_relationship/group_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/period_relationship/period_relationship.last.1_templates.layer_{}.pt"
    },
    "atomic mass - group": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/group_relationship/group_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic mass_relationship/atomic mass_relationship.last.1_templates.layer_{}.pt"
    },
    "atomic mass - period": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/period_relationship/period_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic mass_relationship/atomic mass_relationship.last.1_templates.layer_{}.pt"
    },
    "atomic number - electronegativity": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/electronegativity_relationship/electronegativity_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number_relationship/atomic number_relationship.last.1_templates.layer_{}.pt"
    },
    "atomic mass - electronegativity": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/electronegativity_relationship/electronegativity_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic mass_relationship/atomic mass_relationship.last.1_templates.layer_{}.pt"
    },
    "atomic mass - atomic number": {
        "Y": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number_relationship/atomic number_relationship.last.1_templates.layer_{}.pt",
        "X": "activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic mass_relationship/atomic mass_relationship.last.1_templates.layer_{}.pt"
    },

        

}

def load_activation(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    tensor = torch.load(file_path, map_location='cpu')
    return tensor.numpy()

# --------------------------
# 2. calculate R²
# --------------------------
results = []

for pair_name, templates in DATA_PAIRS.items():
    for layer in range(START_LAYER, NUM_LAYERS + 1):
        x_file = templates["X"].format(layer)
        y_file = templates["Y"].format(layer)
        try:
            X = load_activation(x_file)
            Y = load_activation(y_file)

            if X.shape[0] != Y.shape[0]:
                raise ValueError(f"Layer {layer}: Sample size mismatch.")

            # X: PCA
            pipeline_x = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=PCA_COMPONENTS_X))
            ])
            X_pca = pipeline_x.fit_transform(X)

            # Y:PCA
            pipeline_y = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=PCA_COMPONENTS_X))
            ])
            Y_pca = pipeline_y.fit_transform(Y)

            # Linear regression
            model_linear = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
            kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
            scores = cross_val_score(model_linear, X_pca, Y_pca, cv=kf, scoring='r2')
            r2_avg = scores.mean()

            results.append([pair_name, layer, r2_avg])
            print(f"{pair_name}, Layer {layer}, R²={r2_avg:.4f}")

        except Exception as e:
            print(f"Error in {pair_name}, layer {layer}: {e}")


df_results = pd.DataFrame(results, columns=["pair", "layer", "r2"])
df_results.to_csv(RESULT_CSV, index=False)
print(f"Results saved to {RESULT_CSV}")

# --------------------------
# 3. Plot R²
# --------------------------
df_plot = pd.read_csv(RESULT_CSV)
pairs_in_data = df_plot["pair"].unique()

plt.figure(figsize=(8, 5))
for p in pairs_in_data:
    sub_df = df_plot[df_plot["pair"] == p]
    plt.plot(sub_df["layer"], sub_df["r2"], label=p, marker='o')
plt.xlabel("Layer")
plt.ylabel("R² Score")
plt.title("Linear Model R² with PCA for Each Pair")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.show()

print(f"Plot saved as {OUTPUT_PLOT}")