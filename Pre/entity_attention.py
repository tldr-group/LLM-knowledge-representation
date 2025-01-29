
import torch
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Configuration settings
CONFIG = {
    "model_name": "meta-llama/Llama-3.1-8B",
    "prompt_template": "In the periodic table of elements, the {Attribute} of {Element} is",
    "attributes": ["group", "period"],
    "csv_file": "periodic_table_dataset.csv",
    "quantize": False,
}

def load_config(config_file="config.json"):
    with open(config_file) as f:
        return json.load(f)

def load_tokenizer(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model(model_name, hf_token, quantize=False):
    if quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config if quantize else None,
        use_auth_token=hf_token,
        output_attentions=True
    )
    return model

def get_input_ids_and_attention_mask(prompts, tokenizer):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return inputs["input_ids"], inputs["attention_mask"]

def find_token_positions(tokenizer, tokens, target_words):
    def tokenize_word(word):
        return tokenizer.tokenize(' ' + word) if not word.startswith(' ') else tokenizer.tokenize(word)

    positions = {}
    for word in target_words:
        word_tokens = tokenize_word(word)
        word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
        for i in range(len(tokens) - len(word_ids) + 1):
            if tokens[i:i+len(word_ids)] == word_ids:
                positions[word] = list(range(i, i+len(word_ids)))
                break
        else:
            positions[word] = []
    return positions

import torch.nn.functional as F

def analyze_attention(model, input_ids, attention_mask, tokenizer, source_tokens, target_tokens, eos_token_id):
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

    all_attentions = outputs.attentions  # list of [batch_size, num_heads, seq_len, seq_len]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    token_ids = input_ids[0].tolist()

    # drop the first token
    tokens = tokens[1:]
    token_ids = token_ids[1:]
    attention_mask = attention_mask[:, 1:]

    # drop the last token if it is EOS
    if token_ids and token_ids[-1] == eos_token_id:
        tokens = tokens[:-1]
        token_ids = token_ids[:-1]
        attention_mask = attention_mask[:, :-1]

    # crop the attention matrices
    cropped_attentions = []
    for layer_attn in all_attentions:
        # layer_attn: [1, num_heads, seq_len, seq_len]
        layer_attn = layer_attn[:, :, 1:, 1:]  # drop the first token
        if token_ids and token_ids[-1] == eos_token_id:  # drop the last token if it is EOS
            layer_attn = layer_attn[:, :, :-1, :-1]
        cropped_attentions.append(layer_attn)

    positions = find_token_positions(tokenizer, token_ids, source_tokens + target_tokens)

    source_positions, target_positions = [], []
    for w in source_tokens:
        source_positions.extend(positions.get(w, []))
    for w in target_tokens:
        target_positions.extend(positions.get(w, []))

    num_layers = len(cropped_attentions)
    num_heads = cropped_attentions[0].size(1)
    seq_len = cropped_attentions[0].size(2)

    attention_source_to_target = []
    attention_source_to_others = []
    attention_source_to_all_tokens = []
    attention_entropies = []

    for layer in range(num_layers):
        attn = cropped_attentions[layer][0]        # shape: [num_heads, seq_len, seq_len]
        attn_avg = attn.mean(dim=0)                # shape: [seq_len, seq_len]
        attn_source = attn_avg[source_positions, :] if source_positions else torch.zeros((1, seq_len))
        attn_source_mean = attn_source.mean(dim=0) if source_positions else torch.zeros(seq_len)

        if target_positions:
            attn_to_target = attn_source_mean[target_positions].mean().item()
        else:
            attn_to_target = 0.0

        other_indices = [i for i in range(seq_len) if i not in source_positions + target_positions]
        attn_to_others = torch.mean(attn_source[:, other_indices]).item() if source_positions and other_indices else 0.0

        attention_source_to_target.append(attn_to_target)
        attention_source_to_others.append(attn_to_others)
        attention_source_to_all_tokens.append(attn_source_mean.cpu().numpy())

        attn_probs = F.softmax(attn_source_mean, dim=0)
        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-10)).item()
        attention_entropies.append(entropy)

    return attention_source_to_target, attention_source_to_others, attention_source_to_all_tokens, num_layers, tokens, attention_entropies

def plot_attention_comparison(layers, attention_to_target, attention_to_others, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(layers, attention_to_target, label='Attention to Target')
    plt.plot(layers, attention_to_others, label='Attention to Other Tokens')
    plt.xlabel('Layer')
    plt.ylabel('Average Attention Weight')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_heatmap(attention_source_to_all_tokens, tokens, title, save_path):
    # Convert list of lists to a 2D NumPy array (layers x tokens)
    attention_matrix = np.stack(attention_source_to_all_tokens)

    tokens = ['In', 'Ġthe', 'Ġperiodic', 'Ġtable', 'Ġof', 'Ġelement', ',', 'Ġthe', '{Attribute}', 'Ġof', '{Element}', 'Ġis']
    attention_matrix = attention_matrix[:, :-1]

    print("Final Tokens:", tokens)
    print("Final Attention Matrix Shape:", attention_matrix.shape)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        attention_matrix,
        cmap='Blues',
        xticklabels=tokens,
        yticklabels=range(1, attention_matrix.shape[0] + 1)
    )
    plt.xlabel('Tokens')
    plt.ylabel('Layer')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()




def plot_attention_entropy(layers, attention_entropies, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(layers, attention_entropies, label='Attention Entropy', color='green')
    plt.xlabel('Layer')
    plt.ylabel('Attention Entropy')
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_figure(
    layers,
    attention_heatmap,
    heatmap_tokens,
    avg_attention_to_attributes,
    avg_attention_to_others_attributes,
    avg_attention_to_elements,
    avg_attention_to_others_elements,
    avg_attention_entropies,
    save_path
):
    import matplotlib as mpl
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    attention_heatmap_array = (
        np.stack(attention_heatmap) if isinstance(attention_heatmap, list) else attention_heatmap
    )

    mpl.rcParams['font.size'] = 14
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12

    # Adjusted figsize to make the figure flatter
    fig = plt.figure(figsize=(20, 8))  # Reduced height from 10 to 8
    gridspec_main = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[3, 2], wspace=0.2)

    # Left side: Attention Heatmap
    ax_heatmap = fig.add_subplot(gridspec_main[0, 0])
    attention_heatmap_array_new = attention_heatmap_array[:, :-1]
    heatmap_tokens_new = ['In', 'Ġthe', 'Ġperiodic', 'Ġtable', 'Ġof', 'Ġelement', ',', 'Ġthe', '{Attribute}', 'Ġof', '{Element}', 'Ġis']
    sns.heatmap(
        attention_heatmap_array_new,
        cmap='Blues',
        xticklabels=heatmap_tokens_new,
        yticklabels=range(1, attention_heatmap_array_new.shape[0] + 1),
        ax=ax_heatmap
    )
    ax_heatmap.set_xlabel('Tokens')
    
    ax_heatmap.set_ylabel('Layer')
    ax_heatmap.set_title('Attention Heatmap: "is" to All Tokens')
    ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha='right')


    # Right side: Three subplots
    gridspec_right = gridspec_main[0, 1].subgridspec(3, 1, hspace=0.6)  # Increased hspace from 0.4 to 0.6

    # Top subplot
    ax_top = fig.add_subplot(gridspec_right[0, 0])
    ax_top.plot(layers, avg_attention_to_attributes, label='Attention to Attributes', marker='o')
    ax_top.plot(layers, avg_attention_to_others_attributes, label='Attention to Other Tokens', marker='o')
    ax_top.set_title('Attention from "is" to Attributes vs. Others')
    ax_top.set_xlabel('Layer')
    ax_top.set_ylabel('Attention')
    ax_top.legend()
    ax_top.grid(True)

    # Middle subplot
    ax_mid = fig.add_subplot(gridspec_right[1, 0])
    ax_mid.plot(layers, avg_attention_to_elements, label='Attention to Elements', color='purple', marker='o')
    ax_mid.plot(layers, avg_attention_to_others_elements, label='Attention to Other Tokens', color='grey', marker='o')
    ax_mid.set_title('Attention from "is" to Elements vs. Others')
    ax_mid.set_xlabel('Layer')
    ax_mid.set_ylabel('Attention')
    ax_mid.legend()
    ax_mid.grid(True)

    # Bottom subplot
    ax_bot = fig.add_subplot(gridspec_right[2, 0])
    ax_bot.plot(layers, avg_attention_entropies, label='Attention Entropy', color='green', marker='o')
    ax_bot.set_title('Attention Entropy by Layer')
    ax_bot.set_xlabel('Layer')
    ax_bot.set_ylabel('Entropy')
    ax_bot.legend()
    ax_bot.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    model_name = CONFIG["model_name"]
    prompt_template = CONFIG["prompt_template"]
    attributes = CONFIG["attributes"]
    csv_file = CONFIG["csv_file"]
    quantize = CONFIG["quantize"]
    config = load_config()
    hf_token = config['HF_TOKEN']

    tokenizer = load_tokenizer(model_name, hf_token)
    model = load_model(model_name, hf_token, quantize=quantize)

    df = pd.read_csv(csv_file)
    elements = df['Symbol'].tolist()

    prompts = []
    attribute_last_tokens = {}
    element_last_tokens = {}
    for attribute in attributes:
        last_token = attribute.split()[-1]
        attribute_last_tokens[attribute] = last_token

    for element in elements:
        element_last_tokens[element] = element  

    for attribute in attributes:
        for element in elements:
            prompt = prompt_template.format(Attribute=attribute, Element=element)
            prompts.append(prompt)

    input_prompts = prompts
    input_ids, attention_mask = get_input_ids_and_attention_mask(input_prompts, tokenizer)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    avg_attention_to_attributes = []
    avg_attention_to_elements = []
    avg_attention_to_others_attributes = []
    avg_attention_to_others_elements = []
    avg_attention_to_all_tokens = []
    avg_attention_entropies = []
    total_layers = None
    tokens_reference = None
    num_prompts = len(prompts)

    eos_token_id = tokenizer.eos_token_id

    for idx in range(num_prompts):
        input_id = input_ids[idx].unsqueeze(0)
        att_mask = attention_mask[idx].unsqueeze(0)

        # Attention from "is" to Attributes
        source_tokens_attr = ["is"]
        target_tokens_attr = list(attribute_last_tokens.values())

        attn_attr, attn_others_attr, attn_all_attr, num_layers, tokens, attn_entropy_attr = analyze_attention(
            model,
            input_id,
            att_mask,
            tokenizer,
            source_tokens_attr,
            target_tokens_attr,
            eos_token_id
        )

        # Attention from "is" to Elements
        source_tokens_elem = ["is"]
        target_tokens_elem = list(element_last_tokens.values())

        attn_elem, attn_others_elem, attn_all_elem, _, _, attn_entropy_elem = analyze_attention(
            model,
            input_id,
            att_mask,
            tokenizer,
            source_tokens_elem,
            target_tokens_elem,
            eos_token_id
        )

        if idx == 0:
            total_layers = num_layers
            tokens_reference = tokens.copy()

            # Verify that eos token is excluded
            if tokenizer.eos_token in tokens_reference:
                tokens_reference.remove(tokenizer.eos_token)

            avg_attention_to_attributes = attn_attr
            avg_attention_to_elements = attn_elem
            avg_attention_to_others_attributes = attn_others_attr
            avg_attention_to_others_elements = attn_others_elem
            avg_attention_to_all_tokens = attn_all_attr  
            avg_attention_entropies = attn_entropy_attr  
        else:
            avg_attention_to_attributes = [x + y for x, y in zip(avg_attention_to_attributes, attn_attr)]
            avg_attention_to_elements = [x + y for x, y in zip(avg_attention_to_elements, attn_elem)]
            avg_attention_to_others_attributes = [x + y for x, y in zip(avg_attention_to_others_attributes, attn_others_attr)]
            avg_attention_to_others_elements = [x + y for x, y in zip(avg_attention_to_others_elements, attn_others_elem)]
            avg_attention_to_all_tokens = [x + y for x, y in zip(avg_attention_to_all_tokens, attn_all_attr)]
            avg_attention_entropies = [x + y for x, y in zip(avg_attention_entropies, attn_entropy_attr)]

    avg_attention_to_attributes = [x / num_prompts for x in avg_attention_to_attributes]
    avg_attention_to_elements = [x / num_prompts for x in avg_attention_to_elements]
    avg_attention_to_others_attributes = [x / num_prompts for x in avg_attention_to_others_attributes]
    avg_attention_to_others_elements = [x / num_prompts for x in avg_attention_to_others_elements]
    avg_attention_to_all_tokens = [x / num_prompts for x in avg_attention_to_all_tokens]
    avg_attention_entropies = [x / num_prompts for x in avg_attention_entropies]

    layers = list(range(1, total_layers + 1))
    save_dir = 'Results/entity_attention'
    os.makedirs(save_dir, exist_ok=True)

    # plot: Average Attention from "is" to Attributes vs. Other Tokens Across Layers
    title_attr = 'Average Attention from "is" to Attributes vs. Other Tokens Across Layers'
    save_path_attr = os.path.join(save_dir, 'attention_is_to_attributes_comparison_avg.png')
    plot_attention_comparison(layers, avg_attention_to_attributes, avg_attention_to_others_attributes, title_attr, save_path_attr)

    # plot: Average Attention from "is" to Elements vs. Other Tokens Across Layers
    title_elem = 'Average Attention from "is" to Elements vs. Other Tokens Across Layers'
    save_path_elem = os.path.join(save_dir, 'attention_is_to_elements_comparison_avg.png')
    plt.figure(figsize=(10, 6))
    plt.plot(layers, avg_attention_to_elements, label='Attention to Elements', color='purple')
    plt.plot(layers, avg_attention_to_others_elements, label='Attention to Other Tokens', color='grey')
    plt.xlabel('Layer')
    plt.ylabel('Average Attention Weight')
    plt.title(title_elem)
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path_elem, dpi=300, bbox_inches='tight')
    plt.close()

    # plot: Average Attention Heatmap from "is" to All Tokens Across Layers
    title_heatmap = 'Average Attention from "is" to All Tokens Across Layers'
    save_path_heatmap = os.path.join(save_dir, 'attention_is_to_all_tokens_heatmap_avg.png')
    plot_attention_heatmap(avg_attention_to_all_tokens, tokens_reference, title_heatmap, save_path_heatmap)

    # plot: Average Attention Entropy Across Layers
    entropy_title = 'Average Attention Distribution Entropy Across Layers'
    save_path_entropy = os.path.join(save_dir, 'attention_entropy_avg.png')
    plot_attention_entropy(layers, avg_attention_entropies, entropy_title, save_path_entropy)

    print(f"Attention analysis complete. Averaged plots saved to {save_dir}")

    # plot: Combined Figure
    save_path_combined = os.path.join(save_dir, 'combined_figure.png')
    plot_combined_figure(
        layers,
        avg_attention_to_all_tokens,
        tokens_reference,
        avg_attention_to_attributes,
        avg_attention_to_others_attributes,
        avg_attention_to_elements,
        avg_attention_to_others_elements,
        avg_attention_entropies,
        save_path_combined
    )
    
    

    

if __name__ == "__main__":
    main()