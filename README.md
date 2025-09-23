
# Do Llamas understand the periodic table?

This repository contains the official codebase for our paper:  
**"Do Llamas understand the periodic table?"**

We investigate how large language models (LLMs) encode structured scientific knowledge using chemical elements as a case study. Our key findings include:

- Discovery of a **3D spiral structure** in LLM activations, aligned with the periodic table.
- **Intermediate layers** encode continuous, overlapping attributes suitable for indirect recall.
- **Deeper layers** sharpen categorical boundaries and integrate linguistic context.
- LLMs organize facts as **geometry-aware manifolds**, not just isolated tokens.



## Repository Structure

Each folder corresponds to a section or concept in the paper:

- `Pre/` — Preprocessing scripts: prompt creation, activation extraction.
- `Geometry/` — Code for geometric analyses, such as spiral detection.
- `Direct_recall/` — Linear probing for direct factual recall.
- `Indirect_recall/` — Experiments on retrieving unmentioned or related facts.
- `Appendix/` — Extra analysis, visualizations, and ablation results.
- `Results/` — Saved figures, metrics, and outputs.
- `periodic_table_dataset.csv` — Structured dataset of 50 elements and attributes.

---

## Setup & Installation

1. Clone the repository and enter the project directory.

2. Set your HuggingFace API token in `config.json`:
   ```json
   {
     "HF_TOKEN": "your_huggingface_token"
   }
   ```

3. Install dependencies:
   ```bash
   conda create --name myenv python=3.10
   conda activate myenv
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements.txt
   ```

4. Datasets
This project uses **activation_datasets**.
- Location: `./activation_datasets/` (project root)

### Extracting Residual Stream Yourself
Edit the configuration file: config_extract_activation.yaml
Run the script:
  ```bash
  python Pre/extract_activations.py
  ```
### Download from Hugging Face
  ```bash
  huggingface-cli download leige1114/activation_datasets \
  --repo-type dataset \
  --local-dir activation_datasets \
  --local-dir-use-symlinks False
  ```

## Hardware Compatibility & Quantization

- **bitsandbytes 4-bit quantization** (`load_in_4bit`, `nf4`) is only supported on **Linux with NVIDIA GPUs**.  
  It does **not** work on **macOS (including Apple Silicon)** or **CPU-only** setups.

### If you don’t have an NVIDIA GPU:
- **Disable quantization in configs:**
  - `config_extract_activation.yaml`: set `extraction.quantization.load_in_4bit: false` (or remove the whole block).  
  - `config_indirect.yaml`: set `quantization.load_in_4bit: false` if used.
- **Disable quantization in scripts:**
  - `Geometry/intervention.py`: `'use_quantization': False`  
  - `Appendix/entity_attention.py`: `quantize=False`
- **For scripts without a toggle:**  
  Remove BitsAndBytes-related code, or pass `quantization_config=None`.  
  On CPU, you can also use `device_map="cpu"` and reduce batch size.

 **Note:** `requirements.txt` pins `bitsandbytes`.  
On macOS/CPU-only, installation may fail—remove the dependency and keep quantization disabled.
