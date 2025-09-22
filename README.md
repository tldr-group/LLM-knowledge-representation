
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


## Hardware compatibility and quantization

- bitsandbytes 4-bit quantization (e.g., `load_in_4bit`, `nf4`) is supported only on Linux with NVIDIA GPUs. It is not supported on macOS (including Apple Silicon) or CPU-only environments.
- If you are on macOS or on a machine without an NVIDIA GPU:
  - Disable quantization in configs:
    - In `config_extract_activation.yaml`, set `extraction.quantization.load_in_4bit: false` (or remove the entire `extraction.quantization` block).
    - In `config_indirect.yaml`, set `quantization.load_in_4bit: false` if that section is used by your workflow.
  - Where a script provides a toggle, turn quantization off:
    - `Geometry/intervention.py`: set `'use_quantization': False` in the config.
    - `Appendix/entity_attention.py`: set `quantize=False`.
  - For scripts without a toggle, remove BitsAndBytes-related code or ensure `quantization_config=None` is passed to `from_pretrained`. If running entirely on CPU, you may also set `device_map="cpu"` and reduce batch size.

Note: The repo pins `bitsandbytes` in `requirements.txt`. On macOS/CPU-only, installation or import may fail; you can remove this dependency and disable quantization as above.
