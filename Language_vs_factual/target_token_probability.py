import torch
import json
import os
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from torch.nn import LayerNorm
import numpy as np

def load_model(model_name, hf_token):
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

def load_tokenizer(model_name, hf_token):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_config(config_file="config.json"):
    with open(config_file) as f:
        return json.load(f)

def top_k_top_p_filtering(
    logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), min_tokens_to_keep=1
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[..., indices_to_remove] = filter_value

    return logits

def generate_response(
    model, tokenizer, prompt, n=3, max_length=100, temperature=0.7, top_k=50, top_p=0.9
):
    model.config.output_hidden_states = True
    model.config.return_dict = True

    layer_norm = LayerNorm(model.config.hidden_size).to(model.device).to(torch.float16)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated_tokens = input_ids.clone()

    layer_probs_per_token = []
    generated_token_strings = []
    top50_flags_per_token = []

    for _ in range(n):
        with torch.no_grad():
            outputs = model(
                input_ids=generated_tokens,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states

        filtered_logits = logits / temperature
        filtered_logits = top_k_top_p_filtering(filtered_logits, top_k=top_k, top_p=top_p)
        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

        generated_token_string = tokenizer.decode(next_token_id[0].item(), skip_special_tokens=True)
        generated_token_strings.append(generated_token_string)

        target_token_id = next_token_id[0].item()

        layer_probs = []
        layer_top50_flags = []
        for hidden_state in hidden_states:
            state_half = hidden_state[:, -1, :].half()
            # layer_logits = model.lm_head(state_half)
            normalized_hidden = layer_norm(state_half)
            layer_logits = model.lm_head(normalized_hidden)
            layer_prob_vector = torch.softmax(layer_logits, dim=-1)
            layer_prob = layer_prob_vector[0, target_token_id].item()
            layer_probs.append(layer_prob)

            top_values, top_indices = torch.topk(layer_prob_vector, 50)
            in_top50 = target_token_id in top_indices[0]
            layer_top50_flags.append(in_top50)

        layer_probs_per_token.append(layer_probs)
        top50_flags_per_token.append(layer_top50_flags)

        if generated_tokens.shape[-1] >= max_length:
            break

    new_tokens = generated_tokens[:, input_ids.shape[-1]:]
    response = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    os.makedirs('Results/token_prob', exist_ok=True)

    plt.figure(figsize=(6, 4))
    layer_probs_per_token = np.array(layer_probs_per_token)
    num_layers = layer_probs_per_token.shape[1]

    for token_idx in range(layer_probs_per_token.shape[0]):
        token_label = f'Token {token_idx + 1}: "{generated_token_strings[token_idx]}"'
        line, = plt.plot(
            range(1, num_layers + 1),
            layer_probs_per_token[token_idx],
            label=token_label
        )
        color = line.get_color()
        x_in_top50, y_in_top50 = [], []
        for layer_idx in range(num_layers):
            if top50_flags_per_token[token_idx][layer_idx]:
                x_in_top50.append(layer_idx + 1)
                y_in_top50.append(layer_probs_per_token[token_idx][layer_idx])
        plt.scatter(x_in_top50, y_in_top50, color=color, marker='o', s=30, zorder=3)

    plt.yscale('log')
    plt.xlabel('Layer')
    plt.ylabel('Probability')
    plt.title('Probability of Each Generated Token Across Layers')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend(loc='best', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig('Results/token_prob/probabilities_generated_tokens_across_layers.png', dpi=300)
    plt.close()

    return response

def main(model_name, prompt, n=3, max_length=100, temperature=0.7, top_k=50, top_p=0.9):
    # Load Hugging Face token from config file
    config = load_config()
    hf_token = config.get("hf_token", "")

    # Load model and tokenizer
    model = load_model(model_name, hf_token)
    tokenizer = load_tokenizer(model_name, hf_token)
    
    # Generate and output the response
    response = generate_response(
        model, tokenizer, prompt, n=n, max_length=max_length,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    print("Prompt:", prompt)
    print("Response:", response)

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    # model_name = "meta-llama/Meta-Llama-3.1-70B"
    # prompt = "What is the addition of the atomic number of Magnesium and Copper? "
    prompt = "The atomic number of Mg is "
    n = 10
    max_length = 150
    temperature = 0.5
    top_k = 40
    top_p = 0.95

    main(model_name, prompt, n=n, max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p)
