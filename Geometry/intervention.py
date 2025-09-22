import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_sample_weight
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch.nn as nn
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ---------------------------- Configuration ---------------------------- #

CONFIG = {
    # Transformation Configuration
    'transformation_type': 'polar_3d',  # Choose from: polar_3d, polar_2d, polar_with_period, 
                                        # unit_circle_3d, unit_circle_period, atomic_only, 
                                        # cartesian_3d, random_control, random_polar, 
                                        # scaled_polar, mixed_coordinates
    
    # PCA Configuration
    'USE_PCA': True,                # Toggle PCA usage
    'PCA_COMPONENTS': 35,           # Number of PCA components

    # Data Configuration
    'label_columns': ['Group', 'Atomic Number', 'Period','Random','Random_group'],
    'file_path': 'periodic_table_dataset.csv',

    # Activation Files Configuration
    'activation_file_template': 'activation_datasets/meta-llama-Meta-Llama-3.1-70B/atomic number_single/atomic number_single.last.1_templates.layer_{layer}.pt',

    # 'activation_file_template': 'activation_datasets/meta-llama-Meta-Llama-3.1-70B/numebr_test/numebr_test.last.1_templates.layer_{layer}.pt',

    # Output Paths
    'csv_save_path': "results_linear/number_difference_per_layer.csv",
    'csv_save_path_first_num': "results_linear/first_number_per_layer.csv",
    'csv_save_path_similarity': "results_linear/cosine_similarity_per_layer.csv",  # New CSV for cosine similarities

    # Model Configuration
    'model_name': "meta-llama/Meta-Llama-3.1-70B",

    'use_quantization': True,       # Toggle quantization
    'quantization_config': {        # Quantization parameters (used if 'use_quantization' is True)
        'load_in_4bit': True,
        'bnb_4bit_use_double_quant': False,
        'bnb_4bit_quant_type': "nf4",
        'bnb_4bit_compute_dtype': torch.float16
    },

    # Activation Replacement Configuration
    'replace_activation_dtype': torch.float16,  # Data type for replacing activations (torch.float16 or torch.float32)

    # Processing Configuration
    'num_layers': 80,                # Total number of layers to process
}

# ---------------------------- Transformation Manager ---------------------------- #

class TransformationManager:
    """
    Manages different coordinate transformations for periodic table data.
    Makes it easy to switch between different transformation strategies.
    """
    
    def __init__(self, transformation_type="polar_3d"):
        self.transformation_type = transformation_type
        
    def get_available_transformations(self):
        """Return list of available transformation types."""
        return [
            "polar_3d",           # r*cos(θ), r*sin(θ), r
            "polar_2d",           # r*cos(θ), r*sin(θ)
            "polar_with_period",  # r*cos(θ), r*sin(θ), period
            "unit_circle_3d",     # cos(θ), sin(θ), r
            "unit_circle_period", # cos(θ), sin(θ), period
            "atomic_only",        # atomic_number only
            "cartesian_3d",       # atomic_number, group, period
            "random_control",     # random values for control
            "random_polar",       # random polar coordinates
            "scaled_polar",       # scaled version with different radius scaling
            "mixed_coordinates"   # custom mixed coordinate system
        ]
    
    def transform_labels(self, groups, atomic_numbers, periods, random, random_group):
        """
        Apply the selected transformation to the input data.
        
        Args:
            groups: Group numbers from periodic table
            atomic_numbers: Atomic numbers
            periods: Period numbers
            random: Random values for control
            random_group: Random group values
            
        Returns:
            labels_transformed: Transformed coordinates as numpy array
        """
        
        # Common calculations
        theta = groups * (2 * np.pi / 18)  # Group-based angle
        theta_atomic = atomic_numbers * (2 * np.pi / 118)  # Atomic number based angle
        r = atomic_numbers  # Radius based on atomic number
        
        # Handle NaN values
        cos_theta = np.where(np.isfinite(theta), np.cos(theta), np.nan)
        sin_theta = np.where(np.isfinite(theta), np.sin(theta), np.nan)
        cos_theta_atomic = np.where(np.isfinite(theta_atomic), np.cos(theta_atomic), np.nan)
        sin_theta_atomic = np.where(np.isfinite(theta_atomic), np.sin(theta_atomic), np.nan)
        r = np.where(np.isfinite(r), r, np.nan)
        
        if self.transformation_type == "polar_3d":
            # Default: r*cos(θ), r*sin(θ), r
            return np.vstack((r * cos_theta, r * sin_theta, r)).T
            
        elif self.transformation_type == "polar_2d":
            # 2D polar: r*cos(θ), r*sin(θ)
            return np.vstack((r * cos_theta, r * sin_theta)).T
            
        elif self.transformation_type == "polar_with_period":
            # Polar with period: r*cos(θ), r*sin(θ), period
            return np.vstack((r * cos_theta, r * sin_theta, periods)).T
            
        elif self.transformation_type == "unit_circle_3d":
            # Unit circle with radius: cos(θ), sin(θ), r
            return np.vstack((cos_theta, sin_theta, r)).T
            
        elif self.transformation_type == "unit_circle_period":
            # Unit circle with period: cos(θ), sin(θ), period
            return np.vstack((cos_theta, sin_theta, periods)).T
            
        elif self.transformation_type == "atomic_only":
            # Just atomic number
            return atomic_numbers.reshape(-1, 1)
            
        elif self.transformation_type == "cartesian_3d":
            # Direct cartesian: atomic_number, group, period
            return np.vstack((r, groups, periods)).T
            
        elif self.transformation_type == "random_control":
            # Random values for control experiments
            return random.reshape(-1, 1)
            
        elif self.transformation_type == "random_polar":
            # Random polar coordinates
            random_theta = random_group * (2 * np.pi / 18)
            return np.vstack((np.cos(random_theta), np.sin(random_theta), r)).T
            
        elif self.transformation_type == "scaled_polar":
            # Scaled polar with different radius scaling
            alpha = atomic_numbers * (2 * np.pi / 50)
            return np.vstack((np.cos(alpha), np.sin(theta), np.sqrt(periods))).T
            
        elif self.transformation_type == "mixed_coordinates":
            # Custom mixed coordinate system
            return np.vstack((groups/np.cos(theta), np.sin(theta), r)).T
            
        else:
            raise ValueError(f"Unknown transformation type: {self.transformation_type}")
    
    def get_target_coordinates(self, target_group, target_r, r_period, random_target, random_group_target):
        """
        Generate target coordinates for intervention based on the transformation type.
        
        Args:
            target_group: Target group number
            target_r: Target atomic number (radius)
            r_period: Target period
            random_target: Random target value
            random_group_target: Random group target
            
        Returns:
            linear_target: Target coordinates as numpy array
        """
        
        # Common calculations
        theta_target = target_group * (2 * np.pi / 18)
        theta_atomic_target = target_r * (2 * np.pi / 118)
        
        if self.transformation_type == "polar_3d":
            return np.array([target_r * np.cos(theta_target), target_r * np.sin(theta_target), target_r])
            
        elif self.transformation_type == "polar_2d":
            return np.array([target_r * np.cos(theta_target), target_r * np.sin(theta_target)])
            
        elif self.transformation_type == "polar_with_period":
            return np.array([target_r * np.cos(theta_target), target_r * np.sin(theta_target), r_period])
            
        elif self.transformation_type == "unit_circle_3d":
            return np.array([np.cos(theta_target), np.sin(theta_target), target_r])
            
        elif self.transformation_type == "unit_circle_period":
            return np.array([np.cos(theta_target), np.sin(theta_target), r_period])
            
        elif self.transformation_type == "atomic_only":
            return np.array([target_r])
            
        elif self.transformation_type == "cartesian_3d":
            return np.array([target_r, target_group, r_period])
            
        elif self.transformation_type == "random_control":
            return np.array([random_target])
            
        elif self.transformation_type == "random_polar":
            random_theta_target = random_group_target * (2 * np.pi / 18)
            return np.array([np.cos(random_theta_target), np.sin(random_theta_target), target_r])
            
        elif self.transformation_type == "scaled_polar":
            alpha_target = target_r * (2 * np.pi / 50)
            return np.array([np.cos(alpha_target), np.sin(theta_target), np.sqrt(r_period)])
            
        elif self.transformation_type == "mixed_coordinates":
            return np.array([target_group/np.cos(theta_target), np.sin(theta_target), target_r])
            
        else:
            raise ValueError(f"Unknown transformation type: {self.transformation_type}")
    
    def get_description(self):
        """Return a description of the current transformation."""
        descriptions = {
            "polar_3d": "3D polar coordinates: [r*cos(θ), r*sin(θ), r] where θ=group*(2π/18), r=atomic_number",
            "polar_2d": "2D polar coordinates: [r*cos(θ), r*sin(θ)] where θ=group*(2π/18), r=atomic_number",
            "polar_with_period": "Polar with period: [r*cos(θ), r*sin(θ), period] where θ=group*(2π/18), r=atomic_number",
            "unit_circle_3d": "Unit circle with radius: [cos(θ), sin(θ), r] where θ=group*(2π/18), r=atomic_number",
            "unit_circle_period": "Unit circle with period: [cos(θ), sin(θ), period] where θ=group*(2π/18)",
            "atomic_only": "Atomic number only: [atomic_number]",
            "cartesian_3d": "Direct cartesian: [atomic_number, group, period]",
            "random_control": "Random control: [random_value]",
            "random_polar": "Random polar: [cos(random_θ), sin(random_θ), r] where random_θ=random_group*(2π/18)",
            "scaled_polar": "Scaled polar: [cos(α), sin(θ), √period] where α=atomic_number*(2π/50), θ=group*(2π/18)",
            "mixed_coordinates": "Mixed coordinates: [group/cos(θ), sin(θ), r] where θ=group*(2π/18)"
        }
        return descriptions.get(self.transformation_type, "Unknown transformation")

# ---------------------------- Data Loading and Label Transformation ---------------------------- #

def load_data(config):
    """
    Load and preprocess the periodic table data using the transformation manager.
    """
    periodic_table = pd.read_csv(config['file_path'])
    print(f"Loaded dataset columns: {periodic_table.columns.tolist()}")

    labels_original = periodic_table[config['label_columns']].replace(-np.inf, np.nan).astype(float).values
    symbols = periodic_table['Symbol'].values
    groups = labels_original[:, 0]
    atomic_numbers = labels_original[:, 1]
    periods = labels_original[:, 2]
    random = labels_original[:, 3]
    random_group = labels_original[:, 4]

    # Initialize transformation manager
    transform_manager = TransformationManager(config['transformation_type'])
    print(f"Using transformation: {config['transformation_type']}")
    print(f"Description: {transform_manager.get_description()}")

    # Apply the selected transformation
    labels_transformed = transform_manager.transform_labels(
        groups, atomic_numbers, periods, random, random_group
    )
    
    print(f"Transformed labels shape: {labels_transformed.shape}")

    return periodic_table, labels_transformed, symbols, groups, atomic_numbers, periods, transform_manager

# ---------------------------- Model Loading ---------------------------- #

def load_tokenizer(model_name, hf_token):
    """
    Load the tokenizer for the specified model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
    return tokenizer

def load_model(model_name, hf_token, config):
    """
    Load the transformer model with optional quantization.
    """
    if config['use_quantization']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config['quantization_config']['load_in_4bit'],
            bnb_4bit_use_double_quant=config['quantization_config']['bnb_4bit_use_double_quant'],
            bnb_4bit_quant_type=config['quantization_config']['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=config['quantization_config']['bnb_4bit_compute_dtype']
        )
    else:
        bnb_config = None

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config if config['use_quantization'] else None,
        use_auth_token=hf_token,
    )
    return model

# ---------------------------- Hook Functions ---------------------------- #

def get_batch_mask(prompts, tokenizer):
    """
    Tokenize prompts and create attention masks.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return inputs["input_ids"], inputs["attention_mask"]

def detach_tensor(tensor):
    """
    Detach tensors from the computation graph.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(detach_tensor(x) for x in tensor)
    else:
        return tensor

def get_activation_hook(name, activations):
    """
    Create a hook to capture activations.
    """
    def hook(model, input, output):
        activations[name] = detach_tensor(output)
    return hook

def register_hooks(model, activations):
    """
    Register hooks for all layers in the model.
    """
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(get_activation_hook(f'layer_{i}', activations))
        hooks.append(hook)
    return hooks

def get_activations(model, input_ids, batch_mask):
    """
    Forward pass to collect activations.
    """
    activations = {}
    hooks = register_hooks(model, activations)

    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=batch_mask, output_hidden_states=True)

    for hook in hooks:
        hook.remove()
    return activations

# ---------------------------- Intervention Function ---------------------------- #

def perform_intervention_and_generate(
    random_group_target, random_target, r_period, r_target, layer, symbol, target_group, target_r,
    model, tokenizer, input_ids, batch_mask, device,
    periodic_table, labels_transformed, activation_file_template,
    scaler, y_scaler, pca, config, transform_manager
):
    """
    Perform intervention on activations and generate text.
    """
    activation_file_path = activation_file_template.format(layer=layer)
    if not os.path.exists(activation_file_path):
        print(f"Layer {layer}: Activation file {activation_file_path} does not exist.")
        return None, None, None
    activation_data = torch.load(activation_file_path, map_location='cpu').numpy()
    print(f"Loaded activation data from {activation_file_path}, shape: {activation_data.shape}")

    if activation_data.shape[0] != len(labels_transformed):
        print(f"Layer {layer}: Activation count {activation_data.shape[0]} does not match label count {len(labels_transformed)}. Skipping layer.")
        return None, None, None

    valid_indices = ~np.isnan(labels_transformed).any(axis=1)
    X = activation_data[valid_indices]
    y = labels_transformed[valid_indices]
    symbols_valid = periodic_table['Symbol'].values[valid_indices]
    print(f"Layer {layer}: Number of valid samples: {X.shape[0]}")

    target_indices = np.where(symbols_valid == symbol)[0]
    if len(target_indices) == 0:
        print(f"Layer {layer}: Target symbol {symbol} not found in valid symbols.")
        return None, None, None
    target_index = target_indices[0]

    target_index_num = np.where(periodic_table['Atomic Number'] == r_target)[0]
    if len(target_index_num) > 0:
        target_index_num = target_index_num[0]
        X_train = np.delete(X, target_index_num, axis=0)
        y_train = np.delete(y, target_index_num, axis=0)
    else:
        print(f"Warning: r_target {r_target} not found in y.")
    print(f"Layer {layer}: Excluded target symbol {symbol} from training data.")

    # Standardize X_train and y_train
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)

    if pca is not None:
        # print("PCA Components Statistics:")
        # print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        # print(f"Singular Values: {pca.singular_values_}")
        X_train_scaled = pca.fit_transform(X_train_scaled)
        print(f"Layer {layer}: Applied PCA to the training dataset.")

    lr = LinearRegression()
    try:
        lr.fit(X_train_scaled, y_train_scaled)
        print(f"Layer {layer}: Linear Regression model training completed.")
    except Exception as e:
        print(f"Layer {layer}: Error during model training: {e}. Skipping layer.")
        return None, None, None

    try:
        W = lr.coef_
        b = lr.intercept_
        print(f"Layer {layer}: Extracted coefficients and intercepts.")
    except Exception as e:
        print(f"Layer {layer}: Error extracting weights or intercepts: {e}. Skipping layer.")
        return None, None, None

    # Automatically generate linear_target using the transformation manager
    linear_target = transform_manager.get_target_coordinates(
        target_group, r_target, r_period, random_target, random_group_target
    )
    print(f"Layer {layer}, Symbol {symbol}: Generated linear_target: {linear_target}")


    x_average = np.mean(X, axis=0, keepdims=True)
    x_average_scaled = scaler.transform(x_average)
    if pca is not None:
        x_average_scaled = pca.transform(x_average_scaled)

    W_x_average = W.dot(x_average_scaled.T).flatten()
    W_x_average_with_b = W_x_average + b
    print(f"W_x_average_with_b: {W_x_average_with_b}")

    # Scale linear_target using y_scaler
    linear_target_scaled = y_scaler.transform(linear_target.reshape(1, -1)).flatten()
    # print(f"Linear target scaled: {linear_target_scaled}")

    delta = linear_target_scaled - W_x_average_with_b
    # print(f"Layer {layer}, Symbol {symbol}: Delta: {delta}")

    W_pseudo_inverse = np.linalg.pinv(W)
    Delta_x = W_pseudo_inverse.dot(delta)

    Delta_x = Delta_x.reshape(1, -1)
    x_new_scaled = x_average_scaled + Delta_x

    if pca is not None:
        x_new_scaled = pca.inverse_transform(x_new_scaled)
    x_new = scaler.inverse_transform(x_new_scaled)


    # print("Activation Data Statistics Before Scaling:")
    # print(f"Mean: {X.mean()}, Std: {X.std()}, Max: {X.max()}, Min: {X.min()}")
    # print(f"Any NaN in X: {np.isnan(X).any()}, Any Inf in X: {np.isinf(X).any()}")

    # print(f"Layer {layer}: Standardized the training dataset.")
    # print("Activation Data Statistics After Scaling:")
    # print(f"Mean: {X_train_scaled.mean()}, Std: {X_train_scaled.std()}, Max: {X_train_scaled.max()}, Min: {X_train_scaled.min()}")

    # print("Labels Before Scaling:")
    # print(f"Mean: {y.mean()}, Std: {y.std()}, Max: {y.max()}, Min: {y.min()}")
    # print(f"Any NaN in y: {np.isnan(y).any()}, Any Inf in y: {np.isinf(y).any()}")

    # print("Labels After Scaling:")
    # print(f"Mean: {y_train_scaled.mean()}, Std: {y_train_scaled.std()}, Max: {y_train_scaled.max()}, Min: {y_train_scaled.min()}")

    # print("x_average Statistics Before Scaling:")
    # print(f"Mean: {x_average.mean()}, Std: {x_average.std()}, Max: {x_average.max()}, Min: {x_average.min()}")

    # print("x_average_scaled Statistics:")
    # print(f"Mean: {x_average_scaled.mean()}, Std: {x_average_scaled.std()}, Max: {x_average_scaled.max()}, Min: {x_average_scaled.min()}")


    # print(f"W shape: {W.shape}, W statistics: Mean: {W.mean()}, Std: {W.std()}, Max: {W.max()}, Min: {W.min()}")
    # print(f"b: {b}")


    # print("Activation Data from File Statistics:")
    activation_data = torch.load(activation_file_path, map_location='cpu').numpy()
    # print(f"Mean: {activation_data.mean()}, Std: {activation_data.std()}, Max: {activation_data.max()}, Min: {activation_data.min()}")

    activation_replaced = False

    def replace_activation(module, input, output):
        nonlocal activation_replaced
        if activation_replaced:
            return output

        if isinstance(output, tuple):
            output_tensor = output[0]
        else:
            output_tensor = output

        batch_size, seq_length, hidden_size = output_tensor.shape

        current_positions = torch.full((batch_size,), seq_length - 1, dtype=torch.long, device=output_tensor.device)

        # Ensure x_new is in the same dtype as output_tensor
        x_new_tensor = torch.tensor(x_new, dtype=config['replace_activation_dtype']).to(output_tensor.device)

        # Expand x_new_tensor to match the batch size
        x_new_tensor = x_new_tensor.expand(batch_size, -1)

        output_tensor[torch.arange(batch_size), current_positions, :] = x_new_tensor

        activation_replaced = True

        if isinstance(output, tuple):
            return (output_tensor,) + output[1:]
        else:
            return output_tensor

    handle = model.model.layers[layer].register_forward_hook(replace_activation)

    input_ids = input_ids.to(device)
    batch_mask = batch_mask.to(device)

    # Perform generation with intervention
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=batch_mask,
            max_length=50,
            do_sample=False,
            num_return_sequences=1,
        )
    handle.remove()

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"Layer {layer}, Symbol {symbol}: Generated text: {generated_text}")

    match = re.search(r"(\d+)", generated_text)
    if match:
        first_number = int(match.group())
        number_difference = first_number - r_target
        print(f"Layer {layer}, Symbol {symbol}: First number: {first_number}, Target: {r_target}, Difference: {number_difference}")
    else:
        print(f"Layer {layer}, Symbol {symbol}: No number found in the generated text.")
        number_difference = None
        first_number = None

    return number_difference, first_number

# ---------------------------- Collect Number Differences ---------------------------- #

def collect_number_differences(
    periodic_table, labels_transformed, symbols, groups, atomic_numbers, periods,
    config, model, tokenizer, scaler, y_scaler, pca, transform_manager
):
    """
    Iterate through layers and symbols to collect number differences.
    """
    number_diff_df = pd.DataFrame(columns=symbols)
    first_num_df = pd.DataFrame(columns=symbols)

    # Define prompts_dict (Adjust the prompt as needed)
    prompts_dict = {symbol: f"In the periodic table, the atomic number of element" for symbol in symbols}
    # prompts_dict = {symbol: f"In numbers, the Arabic numeral for number" for symbol in symbols}

    device = next(model.parameters()).device

    # Load existing CSVs if they exist
    if os.path.exists(config['csv_save_path']):
        number_diff_df = pd.read_csv(config['csv_save_path'], index_col=0)
    else:
        print("No existing CSV found for number differences. Starting fresh.")

    if os.path.exists(config['csv_save_path_first_num']):
        first_num_df = pd.read_csv(config['csv_save_path_first_num'], index_col=0)
    else:
        print("No existing CSV found for first numbers. Starting fresh.")



    # Create output directory if it doesn't exist
    if not os.path.exists("results_linear"):
        os.makedirs("results_linear")



    # for layer in range(0, config['num_layers']):
    for layer in range(20,21):  # Adjusted to process layer 20 only for demonstration
        print(f"Processing Layer {layer}...")

        # Initialize rows if they don't exist
        if layer not in number_diff_df.index:
            number_diff_df.loc[layer] = [np.nan] * len(symbols)
        if layer not in first_num_df.index:
            first_num_df.loc[layer] = [np.nan] * len(symbols)

        for idx, symbol in enumerate(symbols):
            if not pd.isna(number_diff_df.at[layer, symbol]):
                print(f"Layer {layer}, Symbol {symbol} already processed. Skipping.")
                continue

            element = periodic_table[periodic_table['Symbol'] == symbol]
            if element.empty:
                print(f"Symbol {symbol} not found in the dataset.")
                continue
            target_group = element['Group'].values[0]
            target_r = element['Atomic Number'].values[0]
            r_period = element['Period'].values[0]
            random_target = element['Random'].values[0]
            random_group = element['Random_group'].values[0]

            prompt = prompts_dict[symbol]
            input_ids, batch_mask = get_batch_mask(prompt, tokenizer)

            number_diff, first_num = perform_intervention_and_generate(
                random_group_target=random_group,
                random_target=random_target,
                r_period=r_period,
                r_target=target_r,
                layer=layer,
                symbol=symbol,
                target_group=target_group,
                target_r=target_r,
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                batch_mask=batch_mask,
                device=device,
                periodic_table=periodic_table,
                labels_transformed=labels_transformed,
                activation_file_template=config['activation_file_template'],
                scaler=scaler,
                y_scaler=y_scaler,
                pca=pca,
                config=config,
                transform_manager=transform_manager
            )

            # Update DataFrames with results
            number_diff_df.at[layer, symbol] = number_diff if number_diff is not None else np.nan
            first_num_df.at[layer, symbol] = first_num if first_num is not None else np.nan

        # Save intermediate results after each layer
        number_diff_df.to_csv(config['csv_save_path'])
        print(f"Saved number differences for Layer {layer} to {config['csv_save_path']}.")

        first_num_df.to_csv(config['csv_save_path_first_num'])
        print(f"Saved first numbers for Layer {layer} to {config['csv_save_path_first_num']}.")

    return number_diff_df, first_num_df

# ---------------------------- Visualization ---------------------------- #

def visualize_results(config, number_diff_df, similarity_df):
    """
    Generate heatmap and line plots for the collected number differences and cosine similarities.
    """
    if os.path.exists(config['csv_save_path']):
        number_diff_df = pd.read_csv(config['csv_save_path'], index_col=0)
        print(f"Loaded number differences from {config['csv_save_path']}.")
    else:
        print(f"CSV file {config['csv_save_path']} not found. Exiting visualization.")
        return

    if os.path.exists(config['csv_save_path_similarity']):
        similarity_df = pd.read_csv(config['csv_save_path_similarity'], index_col=0)
        print(f"Loaded cosine similarities from {config['csv_save_path_similarity']}.")
    else:
        print(f"CSV file {config['csv_save_path_similarity']} not found. Skipping cosine similarity visualization.")

    number_diff_df = number_diff_df.astype(float)

    # Heatmap for Number Differences
    plt.figure(figsize=(20, 15))
    sns.heatmap(number_diff_df, annot=True, fmt=".1f", cmap='Blues', cbar_kws={'label': 'Number Difference'})
    plt.title('Number Difference Heatmap per Layer and Element')
    plt.xlabel('Element Symbol')
    plt.ylabel('Layer')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results_linear/intervention_number_difference_heatmap.png")
    plt.show()

    # Line Plot for Average Number Difference
    average_diff_per_layer = number_diff_df.mean(axis=1)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x=average_diff_per_layer.index, y=average_diff_per_layer.values, marker='o')
    plt.title('Average Number Difference per Layer')
    plt.xlabel('Layer')
    plt.ylabel('Average Number Difference')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results_linear/average_number_difference_per_layer.png")
    plt.show()

    # Heatmap for Cosine Similarities (if available)
    if 'similarity_df' in locals() and not similarity_df.empty:
        plt.figure(figsize=(20, 15))
        sns.heatmap(similarity_df, annot=True, fmt=".4f", cmap='viridis', cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Cosine Similarity of Last Token Logits per Layer and Element')
        plt.xlabel('Element Symbol')
        plt.ylabel('Layer')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("results_linear/cosine_similarity_heatmap.png")
        plt.show()

        # Line Plot for Average Cosine Similarity
        average_similarity_per_layer = similarity_df.mean(axis=1)

        plt.figure(figsize=(12, 6))
        sns.lineplot(x=average_similarity_per_layer.index, y=average_similarity_per_layer.values, marker='o', color='green')
        plt.title('Average Cosine Similarity per Layer')
        plt.xlabel('Layer')
        plt.ylabel('Average Cosine Similarity')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("results_linear/average_cosine_similarity_per_layer.png")
        plt.show()

# ---------------------------- Helper Functions for Easy Experimentation ---------------------------- #

def list_available_transformations():
    """List all available transformation types with descriptions."""
    manager = TransformationManager()
    print("Available transformation types:")
    print("=" * 50)
    for transform_type in manager.get_available_transformations():
        temp_manager = TransformationManager(transform_type)
        print(f"{transform_type:20} - {temp_manager.get_description()}")
    print("=" * 50)

def quick_test_transformation(transformation_type, test_data=None):
    """
    Quickly test a transformation with sample data.
    
    Args:
        transformation_type: The transformation type to test
        test_data: Optional tuple of (groups, atomic_numbers, periods, random, random_group)
                  If None, uses sample data
    """
    if test_data is None:
        # Sample data for testing
        groups = np.array([1, 2, 13, 14, 15, 16, 17, 18])
        atomic_numbers = np.array([1, 3, 5, 6, 7, 8, 9, 10])
        periods = np.array([1, 2, 2, 2, 2, 2, 2, 2])
        random = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        random_group = np.array([5, 7, 2, 8, 12, 4, 9, 1])
    else:
        groups, atomic_numbers, periods, random, random_group = test_data
    
    manager = TransformationManager(transformation_type)
    print(f"\nTesting transformation: {transformation_type}")
    print(f"Description: {manager.get_description()}")
    
    # Transform labels
    labels_transformed = manager.transform_labels(groups, atomic_numbers, periods, random, random_group)
    print(f"Input shape: groups({len(groups)}), atomic_numbers({len(atomic_numbers)}), periods({len(periods)})")
    print(f"Output shape: {labels_transformed.shape}")
    print(f"Sample transformed data (first 3 rows):")
    print(labels_transformed[:3])
    
    # Test target coordinate generation
    target_coords = manager.get_target_coordinates(
        target_group=1, target_r=1, r_period=1, 
        random_target=0.5, random_group_target=3
    )
    print(f"Sample target coordinates: {target_coords}")
    print("-" * 50)

def run_transformation_comparison(config_updates=None):
    """
    Run a comparison of different transformations on the same data.
    
    Args:
        config_updates: Dictionary of config updates to apply
    """
    if config_updates is None:
        config_updates = {}
    
    # Update config
    test_config = CONFIG.copy()
    test_config.update(config_updates)
    
    # Test a few key transformations
    test_transformations = ['polar_3d', 'polar_2d', 'unit_circle_3d', 'atomic_only', 'cartesian_3d']
    
    print("Transformation Comparison")
    print("=" * 60)
    
    for transform_type in test_transformations:
        test_config['transformation_type'] = transform_type
        try:
            periodic_table, labels_transformed, symbols, groups, atomic_numbers, periods, transform_manager = load_data(test_config)
            print(f"{transform_type:20} - Shape: {labels_transformed.shape}, Valid samples: {(~np.isnan(labels_transformed).any(axis=1)).sum()}")
        except Exception as e:
            print(f"{transform_type:20} - ERROR: {e}")
    
    print("=" * 60)

# ---------------------------- Main Function ---------------------------- #

def main():
    # Load data
    periodic_table, labels_transformed, symbols, groups, atomic_numbers, periods, transform_manager = load_data(CONFIG)

    # Load model and tokenizer
    hf_token = CONFIG.get("hf_token", "")  # Ensure your Hugging Face token is set if required
    model = load_model(CONFIG['model_name'], hf_token, CONFIG)
    tokenizer = load_tokenizer(CONFIG['model_name'], hf_token)

    model.eval()

    # Initialize scalers and PCA
    scaler = StandardScaler()
    y_scaler = StandardScaler()
    pca = PCA(n_components=CONFIG['PCA_COMPONENTS']) if CONFIG['USE_PCA'] else None

    # Collect number differences
    number_diff_df, first_num_df = collect_number_differences(
        periodic_table, labels_transformed, symbols, groups, atomic_numbers, periods,
        CONFIG, model, tokenizer, scaler, y_scaler, pca, transform_manager
    )

    # Visualize results
    visualize_results(CONFIG, number_diff_df, None)

if __name__ == "__main__":
    main()
