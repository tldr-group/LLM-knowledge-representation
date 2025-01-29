import json
import os
import torch
import einops
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd

# Load configuration and model parameters
def load_config(config_file="config.json"):
    """
    Load the configuration file.

    Args:
        config_file (str): Path to the configuration JSON file.

    Returns:
        dict: Configuration data.
    """
    with open(config_file) as f:
        return json.load(f)

def load_tokenizer(model_name, hf_token):
    """
    Load the tokenizer for the specified model.

    Args:
        model_name (str): Name of the pre-trained model.
        hf_token (str): Hugging Face token.

    Returns:
        AutoTokenizer: Loaded tokenizer with pad_token set to eos_token.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, hf_token):
    """
    Load the pre-trained model with quantization configuration.

    Args:
        model_name (str): Name of the pre-trained model.
        hf_token (str): Hugging Face token.

    Returns:
        AutoModelForCausalLM: Loaded model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        use_auth_token=hf_token,
    )
    return model

# Prepare input data
def get_batch_mask(prompts, tokenizer):
    """
    Tokenize the input prompts.

    Args:
        prompts (list of str): Input prompts.
        tokenizer (AutoTokenizer): Tokenizer.

    Returns:
        tuple: input_ids and attention_mask tensors.
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    
    # Print number of valid tokens per input
    # print("Number of valid tokens per input:")
    # print(inputs["attention_mask"].sum(dim=1))
    return inputs["input_ids"], inputs["attention_mask"]

# Register activation hooks
def get_activation_hook(name, activations):
    """
    Create a forward hook to save activations.

    Args:
        name (str): Name of the layer.
        activations (dict): Dictionary to store activations.

    Returns:
        function: Hook function.
    """
    def hook(model, input, output):
        activations[name] = detach_tensor(output)
    return hook

def detach_tensor(tensor):
    """
    Detach tensor from computation graph.

    Args:
        tensor (torch.Tensor or tuple/list): Tensor to detach.

    Returns:
        torch.Tensor or tuple/list: Detached tensor.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach()
    elif isinstance(tensor, (tuple, list)):
        return type(tensor)(detach_tensor(x) for x in tensor)
    else:
        return tensor

def register_hooks(model, activations):
    """
    Register forward hooks for each layer in the model.

    Args:
        model (AutoModelForCausalLM): The model.
        activations (dict): Dictionary to store activations.

    Returns:
        list: List of hook handles.
    """
    hooks = []
    for i, layer in enumerate(model.model.layers):
        hook = layer.register_forward_hook(get_activation_hook(f'layer_{i}', activations))
        hooks.append(hook)
    return hooks

# Retrieve activations
def get_activations(model, input_ids, batch_mask):
    """
    Get activations from the model for given inputs.

    Args:
        model (AutoModelForCausalLM): The model.
        input_ids (torch.Tensor): Input IDs tensor.
        batch_mask (torch.Tensor): Attention mask tensor.

    Returns:
        dict: Activations per layer.
    """
    activations = {}
    hooks = register_hooks(model, activations)
    
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=batch_mask, output_hidden_states=True)
    
    for hook in hooks:
        hook.remove()
    return activations

# Process activations
def process_activation_batch(activations, batch_mask, aggregation='last'):
    """
    Process activations based on the aggregation method.

    Args:
        activations (torch.Tensor): Activations tensor.
        batch_mask (torch.Tensor): Attention mask tensor.
        aggregation (str): Aggregation method ('last', 'mean', 'max', 'none').

    Returns:
        torch.Tensor: Processed activations.
    """
    if isinstance(activations, tuple):
        activations = activations[0]
    
    batch_mask = batch_mask.to(activations.device)

    if aggregation == 'last':
        # Get the activation of the last valid token
        last_ix = batch_mask.flip(dims=[1]).argmax(dim=1)
        processed_activations = activations[torch.arange(activations.size(0)), activations.size(1) - 1 - last_ix]
    
    elif aggregation == 'mean':
        # Mean activation of all valid tokens
        masked_activations = activations * batch_mask.unsqueeze(-1)
        valid_token_count = batch_mask.sum(dim=1, keepdim=True)
        processed_activations = masked_activations.sum(dim=1) / valid_token_count
    
    elif aggregation == 'max':
        # Max activation among all tokens
        masked_activations = activations * batch_mask.unsqueeze(-1)
        masked_activations[batch_mask == 0] = float('-inf')
        processed_activations = masked_activations.max(dim=1)[0]
    
    elif aggregation == 'none':
        # No aggregation, return activations for all valid tokens
        processed_activations = einops.rearrange(activations, 'b s d -> (b s) d')
        processed_activations = processed_activations[batch_mask.view(-1) == 1]
    
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")
    
    return processed_activations

# Get and process activations
def get_and_process_activations(model, tokenizer, prompts, aggregation='last'):
    """
    Get and process activations for a set of prompts.

    Args:
        model (AutoModelForCausalLM): The model.
        tokenizer (AutoTokenizer): The tokenizer.
        prompts (list of str): Input prompts.
        aggregation (str): Aggregation method.

    Returns:
        dict: Processed activations per layer.
    """
    input_ids, batch_mask = get_batch_mask(prompts, tokenizer)
    activations = get_activations(model, input_ids, batch_mask)

    processed_activations = {}
    for layer_name, layer_activations in activations.items():
        processed_activations[layer_name] = process_activation_batch(layer_activations, batch_mask, aggregation)
    
    return processed_activations

# Save activations

def save_activations(model_name, activations, entity_type, prompt_name, layer_ix, aggregation='last', save_dir='activation_datasets'):
    """
    Save activations to a specified directory as a .pt file.

    Args:
        model_name (str): Name of the model.
        activations (torch.Tensor): The activations to save.
        entity_type (str): Type of the entity (e.g., 'person', 'element').
        prompt_name (str): Name of the prompt.
        layer_ix (int): Index of the layer.
        aggregation (str): Type of aggregation used for activations.
        save_dir (str): Base directory where activations will be saved.
    """
    # Define the model-specific directory
    model_dir = os.path.join(save_dir, model_name.replace('/', '-'))
    
    # Define the save path
    activation_save_path = os.path.join(model_dir, entity_type)
    os.makedirs(activation_save_path, exist_ok=True)
    
    # Define the save file name
    save_name = f'{entity_type}.{aggregation}.{prompt_name}.layer_{layer_ix}.pt'
    save_path = os.path.join(activation_save_path, save_name)
    
    # Save the activations as a .pt file
    torch.save(activations, save_path)
    print(f"Activations saved at: {save_path}")


# Process and save activations
# Process and save activations
def process_and_save_activations(model, tokenizer, prompts, layer_ix, entity_type, prompt_name, model_name, aggregation='last', save_dir='activation_datasets', batch_size=550):
    """
    Process activations and save them for a given model, tokenizer, and prompts, and store the activations from the same layer in a single file.

    Args:
        model (AutoModelForCausalLM): The model to use.
        tokenizer (AutoTokenizer): The tokenizer for the model.
        prompts (list of str): List of input prompts.
        layer_ix (int): Layer index to save activations.
        entity_type (str): Type of the entity.
        prompt_name (str): Name of the prompt.
        model_name (str): Name of the model (for saving in the correct folder).
        aggregation (str): Aggregation method ('last', 'mean', 'max').
        save_dir (str): Base directory to save activations.
    """
    all_activations = []  # To store all batch activations for the layer

    # Iterate over batches of prompts
    for start_ix in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start_ix:start_ix + batch_size]
        
        # Process the activations for the current batch
        processed_activations = get_and_process_activations(model, tokenizer, batch_prompts, aggregation)
        layer_key = f'layer_{layer_ix}'
        
        if layer_key in processed_activations:
            all_activations.append(processed_activations[layer_key])  # Collect activations
        else:
            print(f"Layer {layer_ix} not found in activations.")
    
    # Concatenate all the activations for the layer
    if all_activations:
        concatenated_activations = torch.cat(all_activations, dim=0)
        save_activations(
            model_name=model_name,
            activations=concatenated_activations,  # Now saving all activations together
            entity_type=entity_type,
            prompt_name=prompt_name,
            layer_ix=layer_ix,
            aggregation=aggregation,
            save_dir=save_dir
        )
    else:
        print(f"No activations found for Layer {layer_ix}.")



# Generate prompts
def generate_prompts(df, templates):
    """
    Generate prompts based on templates and DataFrame rows.

    Args:
        df (pandas.DataFrame): DataFrame containing entity data.
        templates (list of str): List of prompt templates.

    Returns:
        list of str: Generated prompts.
    """
    prompts = []
    for _, row in df.iterrows():
        for template in templates:
            try:
                prompt = template.format(**row.to_dict())
                prompts.append(prompt)
            except KeyError as e:
                print(f"Missing key in data for template: {e}")
    return prompts

# Main processing function
def main():
    # Load configuration
    config_data = load_config()
    HF_TOKEN = config_data.get("HF_TOKEN")
    
    # Define model name
    # model_name = config_data.get("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B")
    model_name = "meta-llama/Meta-Llama-3.1-70B"
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "meta-llama/Llama-2-7b-hf"
    # model_name = "meta-llama/Llama-3.1-8B"

    
    # Load tokenizer and model
    tokenizer = load_tokenizer(model_name, HF_TOKEN)
    model = load_model(model_name, HF_TOKEN)
    
    # Define entity types with their data files and templates
    entities = [
        {
            "entity_type": "atomic number",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "{Element Name} ({Symbol}) has an atomic number of ",
                "The atomic number of {Element Name} ({Symbol}) is ",
                "For {Element Name} ({Symbol}), the atomic number is ",
                "{Element Name} ({Symbol}) has the atomic number ",
                "{Element Name} ({Symbol}) holds an atomic number of ",
                "{Element Name} ({Symbol}) is assigned the atomic number ",
                "For {Element Name} ({Symbol}), it is known that the atomic number is ",
                "In terms of atomic numbers, {Element Name} ({Symbol}) is assigned ",
                "{Element Name} ({Symbol}) has an assigned atomic number of ",
                "The atomic number for {Element Name} ({Symbol}) is known to be ",
                "When it comes to atomic numbers, {Element Name} ({Symbol}) has "
            ],
            "prompt_name": "11_templates"
        },
        {
            "entity_type": "atomic mass",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "{Element Name} ({Symbol}) has an atomic mass of ",
                "The atomic mass of {Element Name} ({Symbol}) is ",
                "For {Element Name} ({Symbol}), the atomic mass is ",
                "{Element Name} ({Symbol}) has the atomic mass ",
                "{Element Name} ({Symbol}) holds an atomic mass of ",
                "{Element Name} ({Symbol}) is assigned the atomic mass ",
                "For {Element Name} ({Symbol}), it is known that the atomic mass is ",
                "In terms of atomic masses, {Element Name} ({Symbol}) is assigned ",
                "{Element Name} ({Symbol}) has an assigned atomic mass of ",
                "The atomic mass for {Element Name} ({Symbol}) is known to be ",
                "When it comes to atomic masses, {Element Name} ({Symbol}) has "
            ],
            "prompt_name": "11_templates"
        },
        {
            "entity_type": "group",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "{Element Name} ({Symbol}) belongs to group ",
                "The element {Element Name} ({Symbol}) is part of group ",
                "In the periodic table, {Element Name} ({Symbol}) is found in group ",
                "{Element Name} ({Symbol}) is assigned to group ",
                "For {Element Name} ({Symbol}), the group number is ",
                "{Element Name} ({Symbol}) can be classified under group ",
                "When categorized, {Element Name} ({Symbol}) is placed in group ",
                "In terms of groups, {Element Name} ({Symbol}) falls under group ",
                "You will find {Element Name} ({Symbol}) in group ",
                "The group for {Element Name} ({Symbol}) in the periodic table is ",
                "According to the periodic table, {Element Name} ({Symbol}) is in group "
            ],
            "prompt_name": "11_templates"
        },
        {
            "entity_type": "period",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "{Element Name} ({Symbol}) is located in period ",
                "The element {Element Name} ({Symbol}) belongs to period ",
                "In the periodic table, {Element Name} ({Symbol}) is found in period ",
                "{Element Name} ({Symbol}) falls under period ",
                "{Element Name} ({Symbol}) is assigned to period ",
                "For {Element Name} ({Symbol}), the period number is ",
                "The element {Element Name} ({Symbol}) is categorized under period ",
                "{Element Name} ({Symbol}) can be found in period ",
                "In terms of periods, {Element Name} ({Symbol}) is placed in period ",
                "According to the periodic table, {Element Name} ({Symbol}) is in period ",
                "You will find {Element Name} ({Symbol}) in period "
            ],
            "prompt_name": "11_templates"
        },
        {
            "entity_type": "electronegativity",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "{Element Name} ({Symbol}) has an electronegativity of ",
                "The element {Element Name} ({Symbol}) has an electronegativity of ",
                "In the periodic table, {Element Name} ({Symbol}) is assigned an electronegativity of ",
                "{Element Name} ({Symbol}) possesses an electronegativity of ",
                "{Element Name} ({Symbol}) is assigned an electronegativity of ",
                "For {Element Name} ({Symbol}), the electronegativity value is ",
                "The element {Element Name} ({Symbol}) has been assigned an electronegativity of ",
                "{Element Name} ({Symbol}) can be found with an electronegativity of ",
                "In terms of electronegativity, {Element Name} ({Symbol}) has a value of ",
                "According to the periodic table, {Element Name} ({Symbol}) has an electronegativity of ",
                "You will find {Element Name} ({Symbol}) with an electronegativity of "
            ],
            "prompt_name": "11_templates"
        },
        {
            "entity_type": "atomic number question",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "What is the atomic number of {Element Name} ({Symbol})?",
                "Do you know the atomic number of {Element Name} ({Symbol})?",
                "Can you tell me the atomic number of {Element Name} ({Symbol})?",
                "What number is {Element Name} ({Symbol}) on the periodic table?",
                "How high is the atomic number of {Element Name} ({Symbol})?",
                "What position does {Element Name} ({Symbol}) hold in atomic number?",
                "Where does {Element Name} ({Symbol}) rank in atomic number?",
                "Do you happen to know {Element Name} ({Symbol})'s atomic number?",
                "Can you guess the atomic number of {Element Name} ({Symbol})?",
                "What's the atomic number of {Element Name} ({Symbol})?",
                "Which atomic number is assigned to {Element Name} ({Symbol})?"
            ],
            "prompt_name": "11_templates_questions"
        },
        {
            "entity_type": "period question",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "Which period is {Element Name} ({Symbol}) located in?",
                "Do you know which period {Element Name} ({Symbol}) belongs to?",
                "In the periodic table, what period is {Element Name} ({Symbol}) found in?",
                "Can you tell me which period {Element Name} ({Symbol}) falls under?",
                "What period is {Element Name} ({Symbol}) assigned to?",
                "For {Element Name} ({Symbol}), what is the period number?",
                "Which period is the element {Element Name} ({Symbol}) categorized under?",
                "Where can {Element Name} ({Symbol}) be found in terms of periods?",
                "What period is {Element Name} ({Symbol}) placed in?",
                "According to the periodic table, which period is {Element Name} ({Symbol}) in?",
                "Can you tell me in which period {Element Name} ({Symbol}) is located?"
            ],
            "prompt_name": "11_templates_questions"
        },
        
        {
            "entity_type": "group question",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "Which group does {Element Name} ({Symbol}) belong to?",
                "Do you know which group {Element Name} ({Symbol}) is part of?",
                "In the periodic table, what group is {Element Name} ({Symbol}) found in?",
                "Can you tell me which group {Element Name} ({Symbol}) is assigned to?",
                "What is the group number for {Element Name} ({Symbol})?",
                "Under which group can {Element Name} ({Symbol}) be classified?",
                "When categorized, in which group is {Element Name} ({Symbol}) placed?",
                "In terms of groups, which group does {Element Name} ({Symbol}) fall under?",
                "Where can you find {Element Name} ({Symbol}) in terms of groups?",
                "What is the group for {Element Name} ({Symbol}) in the periodic table?",
                "According to the periodic table, what group is {Element Name} ({Symbol}) in?"
            ],
            "prompt_name": "11_templates_questions"
        },
        {
            "entity_type": "atomic mass question",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "What is the atomic mass of {Element Name} ({Symbol})?",
                "Do you know the atomic mass of {Element Name} ({Symbol})?",
                "What is the atomic mass for {Element Name} ({Symbol})?",
                "Can you tell me the atomic mass of {Element Name} ({Symbol})?",
                "What atomic mass is assigned to {Element Name} ({Symbol})?",
                "What is the atomic mass value for {Element Name} ({Symbol})?",
                "What is known about the atomic mass of {Element Name} ({Symbol})?",
                "In terms of atomic mass, what is {Element Name} ({Symbol}) assigned?",
                "What atomic mass has been assigned to {Element Name} ({Symbol})?",
                "What is the known atomic mass for {Element Name} ({Symbol})?",
                "When it comes to atomic mass, what value does {Element Name} ({Symbol}) have?"
            ],
            "prompt_name": "11_templates_questions"
        },
        {
            "entity_type": "electronegativity question",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "What is the electronegativity of {Element Name} ({Symbol})?",
                "Do you know the electronegativity of {Element Name} ({Symbol})?",
                "What is the electronegativity assigned to {Element Name} ({Symbol}) in the periodic table?",
                "Can you tell me the electronegativity of {Element Name} ({Symbol})?",
                "What electronegativity value is assigned to {Element Name} ({Symbol})?",
                "What is the electronegativity value for {Element Name} ({Symbol})?",
                "What electronegativity has been assigned to {Element Name} ({Symbol})?",
                "Where can you find {Element Name} ({Symbol}) in terms of electronegativity?",
                "In terms of electronegativity, what value does {Element Name} ({Symbol}) have?",
                "According to the periodic table, what is the electronegativity of {Element Name} ({Symbol})?",
                "Can you find the electronegativity of {Element Name} ({Symbol})?"
            ],
            "prompt_name": "11_templates_questions"
        },
        {
            "entity_type": "following atomic number",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "In the periodic table, the element directly after {Element Name} ({Symbol}) has an atomic number of ",
                "The atomic number of the element that comes right after {Element Name} ({Symbol}) is ",
                "The atomic number of the element following {Element Name} ({Symbol}) is ",
                "{Element Name} ({Symbol}) is followed by an element with an atomic number of ",
                "In the periodic sequence, the element after {Element Name} ({Symbol}) has an atomic number of ",
                "Just after {Element Name} ({Symbol}), there is an element with an atomic number of ",
                "The element that comes after {Element Name} ({Symbol}) in the periodic table has an atomic number of ",
                "Right after {Element Name} ({Symbol}), the element’s atomic number is ...",
                "After {Element Name} ({Symbol}) in the periodic table, there is an element with an atomic number of ",
                "The element directly following {Element Name} ({Symbol}) has an atomic number of ",
                "The atomic number of the element found just after {Element Name} ({Symbol}) is "
            ],
            "prompt_name": "11_templates_following"
        },
        {
            "entity_type": "atomic mass difference",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "The atomic mass difference between {Element Name} ({Symbol}) and the previous element is ",
                "The atomic mass of {Element Name} ({Symbol}) differs from that of the preceding element by ",
                "The difference in atomic mass between {Element Name} ({Symbol}) and the element before it is ",
                "The atomic mass difference between {Element Name} ({Symbol}) and the element that precedes it is ",
                "The atomic mass difference between {Element Name} ({Symbol}) and the element just before it is",
                "The atomic mass of {Element Name} ({Symbol}) is different from the element before it by ",
                "The element before {Element Name} ({Symbol}) has an atomic mass difference of ",
                "In the periodic sequence, the atomic mass difference between {Element Name} ({Symbol}) and the element before it is ",
                "Right before {Element Name} ({Symbol}), the atomic mass difference is ...",
                "Before {Element Name} ({Symbol}) in the periodic table, the atomic mass difference between them is ",
                "The difference in atomic mass between {Element Name} ({Symbol}) and the element found just before it is "
            ],
            "prompt_name": "11_templates_atomic_mass_difference"
        },
        {
            "entity_type": "double atomic mass",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "The atomic mass of {Element Name} ({Symbol}) doubled is  ",
                "Twice the atomic mass of {Element Name} ({Symbol}) is  ",
                "Double the atomic mass of {Element Name} ({Symbol}) gives  ",
                "The atomic mass of {Element Name} ({Symbol}), when multiplied by 2, is  ",
                "Doubling the atomic mass of {Element Name} ({Symbol}) results in  ",
                "The result of doubling the atomic mass of {Element Name} ({Symbol}) is  ",
                "{Element Name} ({Symbol})'s atomic mass, when doubled, equals  ",
                "Multiplying the atomic mass of {Element Name} ({Symbol}) by 2 gives  ",
                "{Element Name} ({Symbol})’s atomic mass times two equals  ",
                "Two times the atomic mass of {Element Name} ({Symbol}) is  ",
                "The value of {Element Name} ({Symbol})'s atomic mass doubled is  "
            ],
            "prompt_name": "11_templates_double_atomic_mass"
        },
        {
            "entity_type": "square atomic mass",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
                "The square of the atomic mass of {Element Name} ({Symbol}) is ",
                "The atomic mass of {Element Name} ({Symbol}) squared is ",
                "Squaring the atomic mass of {Element Name} ({Symbol}) gives ",
                "The atomic mass of {Element Name} ({Symbol}), when squared, is ",
                "The result of squaring the atomic mass of {Element Name} ({Symbol}) is ",
                "{Element Name} ({Symbol})'s atomic mass, when squared, equals ",
                "Multiplying the atomic mass of {Element Name} ({Symbol}) by itself gives ",
                "{Element Name} ({Symbol})’s atomic mass times itself equals ",
                "The value of {Element Name} ({Symbol})'s atomic mass squared is ",
                "The atomic mass of {Element Name} ({Symbol}) to the power of two is ",
                "When you square the atomic mass of {Element Name} ({Symbol}), you get "
            ],
            "prompt_name": "11_templates_square_atomic_mass"
        },
        

        {
            "entity_type": "numebr_test",
            "data_file": "periodic_table_dataset.csv",
            "templates": [
              "In numbers, the Arabic numeral for {Number}",
            ],
            "prompt_name": "1_templates"
        },
        
        {
            "entity_type": "atomic number_single",
            "data_file":"periodic_table_dataset.csv",
            "templates": [
              "In the periodic table, the atomic number of {Symbol}",
            ],
            "prompt_name": "1_templates"
        },

        {
            "entity_type": "atomic number_relationship",
            "data_file":"periodic_table_dataset.csv",
            "templates": [
              "In the periodic table, the atomic number of {Symbol} is ",
            ],
            "prompt_name": "1_templates"
        },


        {
            "entity_type": "atomic mass_relationship",
            "data_file":"periodic_table_dataset.csv",
            "templates": [
              "In the periodic table, the atomic mass of {Symbol} is ",
            ],
            "prompt_name": "1_templates"
        },



        {
            "entity_type": "group_relationship",
            "data_file":"periodic_table_dataset.csv",
            "templates": [
              "In the periodic table, the group of {Symbol} is ",
            ],
            "prompt_name": "1_templates"
        },

        {
            "entity_type": "period_relationship",
            "data_file":"periodic_table_dataset.csv",
            "templates": [
              "In the periodic table, the period of {Symbol} is ",
            ],
            "prompt_name": "1_templates"
        },

        {
            "entity_type": "electronegativity_relationship",
            "data_file":"periodic_table_dataset.csv",
            "templates": [
              "In the periodic table, the electronegativity of {Symbol}",
            ],
            "prompt_name": "1_templates"
        }
    ]
    
    # Define save directory
    base_save_dir = "activation_datasets"
    
    # Iterate over each entity type
    for entity in entities:
        entity_type = entity["entity_type"]
        data_file = entity["data_file"]
        templates = entity["templates"]
        prompt_name = entity["prompt_name"]
        
        print(f"Processing entity type: {entity_type}")
        
        # Load data
        if not os.path.exists(data_file):
            print(f"Data file {data_file} not found. Skipping entity {entity_type}.")
            continue
        df = pd.read_csv(data_file)
        
        # Generate prompts
        prompts = generate_prompts(df, templates)
        print(f"Generated {len(prompts)} prompts for entity type '{entity_type}'.")
        
        # Define batch size
        batch_size = 550 # Adjust based on GPU memory
        
        # Define aggregation method
        aggregation = 'last'  # Can be 'last', 'mean', 'max', 'none'
        
        # Define number of layers (assuming model has 'n_layers' layers)
        # Alternatively, determine from the model
        num_layers = len(model.model.layers)
        print(f"Model has {num_layers} layers.")
        
        # Iterate over each layer
        for layer_ix in range(num_layers):
        # for layer_ix in range(20,21):
            print(f"Processing Layer {layer_ix}")
            # Iterate over batches
            for start_ix in range(0, len(prompts), batch_size):
                batch_prompts = prompts[start_ix:start_ix + batch_size]
                
                try:
                    # Process and save activations for the current batch and layer
                    process_and_save_activations(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=batch_prompts,
                        layer_ix=layer_ix,
                        entity_type=entity_type,
                        prompt_name=prompt_name,
                        model_name=model_name,
                        aggregation=aggregation,
                        save_dir=base_save_dir,
                        batch_size=batch_size
                    )
                    print(f"Processed batch {start_ix // batch_size + 1} for Layer {layer_ix}")
                
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA out of memory: {e}. Reducing batch size or freeing up memory.")
                    torch.cuda.empty_cache()
                    # Optionally, implement a smaller batch size retry mechanism
                    # For simplicity, we skip the batch if OOM occurs
                    continue
            
            print(f"Layer {layer_ix} activations processed and saved.")
    
    print("All activations processed and saved.")

if __name__ == "__main__":
    main()