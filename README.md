# frank-radfact
Frankenstein repo to simplify the implementation of RadFact score assessment

## Clone Repository

To clone the repository with FActScore submodule:

```bash
git clone --recurse-submodules https://github.com/jagh/frank-radfact.git
```

## Prerequisites

1. Install Python 3.10.14

2. Create and activate conda environment:

```bash
conda create -n frank-radfact python=3.10.14
conda activate frank-radfact
```

3. Install requriements:

```bash
pip install -r requirements.txt
```

4. Setup HuggingFace access:

- Create an account at https://huggingface.co
- Visit https://huggingface.co/settings/tokens
- Create a new token with "read" access
- Login using CLI:

```bash
   huggingface-cli login   # Enter your token when prompted
   # Optional: Set up git credentials
   git config --global credential.helper store
   # Verify login
   huggingface-cli whoami
   # Request LLaMA-2 access:
   huggingface-cli login --token <your-token>
   huggingface-cli join meta-llama/Llama-2-7b
```

## Installation

### Installation of FActScore

FActScore is a submodule of this repository. To install it, run the following commands:

```bash
cd src/third-party/FActScore/
pip install --upgrade factscore
python -m spacy download en_core_web_sm
```

### Download the data
```bash
python -m factscore.download_data --llama_7B_HF_path "llama-7B"
```

### Using huggyllama/llama-7b:
```bash
python -m factscore.download_data --llama_7B_HF_path "huggyllama/llama-7b"
```