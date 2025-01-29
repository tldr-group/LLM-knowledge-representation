# LLM-Knowledge-Representation

This study investigates how Large Language Models (LLMs) represent and recall **Interwoven Structured Knowledge** across transformer layers.

## Project Structure
The project consists of three main parts, along with preliminary setup and data preparation in the `Pre` folder.

### **Pre** (Preliminary Work)
The `Pre` folder includes essential preparation steps:
- **Dataset generation**
- **Activation extraction**
- **Attention map analysis**
- **t-SNE visualization of activation distributions**

### **1. Intermediate Layers Encode Knowledge, Later Layers Shape Language** (`Language_vs_factual` folder)
This section explores how intermediate layers store factual knowledge, while later layers refine language outputs.
- **Linear probing**: Training Support Vector Regression (SVR) models for each layer.
- **Non-matching linear probing**
- **Probability of the target token across layers**: Probabilities are calculated by iteratively re-running the model with the next token appended to the prompt.

### **2. Recall Peaks at Intermediate Layers** (`Intervention` folder)
This section investigates whether related attributes are interconnected by analyzing the recall ability of LLMs—their capacity to retrieve attributes related to, but not explicitly mentioned in the prompt. Additionally, we explore the geometric mechanisms behind this recall process by intervention.

### **3. Relationship in Attribute Representation: From Superposition to Separation** (`relationship` folder)
This section examines whether one attribute’s representation can recall related attributes without being mentioned, as well as the relationships between attribute representations across different layers.

## **Findings**
We show that **intermediate layers encode factual knowledge** by superimposing related attributes in overlapping spaces, enabling effective recall even when attributes are not explicitly prompted. In contrast, **later layers refine linguistic patterns and progressively separate attribute representations**, optimizing task-specific outputs while narrowing attribute recall.

All study results can be found in the `Results` folder.

We identify diverse encoding patterns, including the first-time observation of **3D spiral structures** when analyzing information related to the periodic table of elements. Our findings reveal a dynamic transition in attribute representations across layers, contributing to mechanistic interpretability and providing insights into how LLMs process complex, interrelated knowledge.

## **Setup & Installation**
1. Before running the code, **set your HF_TOKEN** in `config.json`.
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the respective scripts from each folder as needed.