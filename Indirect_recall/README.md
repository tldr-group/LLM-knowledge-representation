# Indirect Recall Experiments

This script runs indirect recall experiments on various language models using configurable parameters through command line arguments and configuration files.

## Configuration

The script uses a YAML configuration file (`config_indirect.yaml`) to specify:
- Model configurations (name, activation paths, number of layers)
- Experimental parameters (dataset, methods, splitting strategy)
- Training parameters (SVR settings, Random Forest settings)
- Output settings

### Configuration File Structure

```yaml
models:
  - name: "Model Name"
    activation_path_template: "path/to/activations.{layer}.pt"
    num_layers: 32
    enabled: true  # Set to false to disable this model

experiment:
  dataset_file: "periodic_table_dataset.csv"
  label_column: "Group"
  split_method: "group_shuffle"  # Options: middle, first, group_shuffle
  regression_methods: ["svr_cv"]  # Options: svr, random_forest, svr_cv
  prompt_template_number: 11

output:
  results_dir: "../Results/non_matching"

training:
  svr_kernel: "linear"
  svr_c: 2
  rf_n_estimators: 100
  rf_random_state: 42
  cv_splits: 5
  cv_random_state: 100
```

## Usage

### Basic Usage

Run with default configuration:
```bash
python Indirect_recall/indirect.py
```

### Specify Configuration File

```bash
python Indirect_recall/indirect.py --config my_config_indirect.yaml
```

### Run Specific Models

```bash
python Indirect_recall/indirect.py --models "Llama-2-7b-hf" "Llama-3.1-8B"
```

### Override Configuration Parameters

```bash
python Indirect_recall/indirect.py --methods svr random_forest --split-method middle --label-column Group
```

### Full Example

```bash
python Indirect_recall/indirect.py \
    --config config_indirect.yaml \
    --models "Llama-2-7b-hf (Non-matching prompt)" "Llama-2-7b-hf (Matching prompt)" \
    --methods svr_cv \
    --split-method group_shuffle \
    --dataset periodic_table_dataset.csv \
    --output-dir Results/my_experiment
```

## Command Line Arguments

- `--config, -c`: Path to configuration file (default: config_indirect.yaml)
- `--models, -m`: Specific model names to run (runs all enabled models if not specified)
- `--methods`: Regression methods to use (overrides config)
- `--split-method`: Data splitting method (overrides config)
- `--label-column`: Label column name (overrides config)
- `--dataset`: Path to dataset file (overrides config)
- `--output-dir`: Output directory for results (overrides config)

## Available Options

### Regression Methods
- `svr`: Support Vector Regression
- `random_forest`: Random Forest Regression
- `svr_cv`: Cross-validated SVR (recommended)

### Split Methods
- `middle`: Use middle group for testing
- `first`: Use first group for testing
- `group_shuffle`: Random group-based splitting

## Output

The script generates:
- R² score plots comparing models across layers
- Legend files for the plots
- Console output showing best performing layers for each model

Results are saved to the directory specified in the configuration or command line arguments.
