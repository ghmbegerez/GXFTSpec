# Genexus SPEC (GXFT)

## Overview
This repository contains the specification and implementation for the Genexus SPEC (Supervised Pre-trained Engine for Computing) system, a comprehensive solution for fine-tuning large language models.

## System Features

- REST API for fine-tuning job management
- Command Line Interface for interacting with the API
- SDK for programmatic integration
- Job queue system for managing concurrent training
- Support for different fine-tuning methods (SFT, LoRA, QLoRA)
- Model and adapter management in a centralized hub

## Documentation

### Spec File
- `spec.md`: Main specification document 
- Use markdown preview and mermaid preview for visualization

### Generating PDF Documentation
```bash
# Generate PDF from spec.md
pandoc spec.md --filter mermaid-filter -o spec.pdf
```

#### Prerequisites
1. Install LaTeX support:
```bash
brew install --cask basictex
eval "$(/usr/libexec/path_helper)"
```

2. Install Pandoc:
```bash
brew install pandoc
```

3. Install Mermaid filter:
```bash
npm install -g mermaid-filter
```

## Using the Fine-tuning System

### Using the CLI for fine-tuning

The CLI includes several commands for working with models, adapters, and fine-tuning jobs:

```bash
# View all available commands
python -m sdk.finetunercli -h

# Create a configuration file template
python -m sdk.finetunercli create-config my_config --model "gpt2" --dataset "fine_tuning_data/alpaca.jsonl" --method "lora"

# Run fine-tuning with the configuration file
python -m sdk.finetunercli train my_config.json

# Evaluate a fine-tuned model
python -m sdk.finetunercli evaluate ./lora_output/final_model --dataset "fine_tuning_data/alpaca.jsonl"
```

### Example Configuration File

To run fine-tuning directly, you can use a JSON or YAML configuration file:

```json
{
  "dataset": {
    "format": "alpaca",
    "path": "fine_tuning_data/alpaca.jsonl",
    "train_split": "train"
  },
  "model": {
    "_model_name_or_path": "gpt2",
    "max_length": 512,
    "padding_side": "right",
    "device_map": "auto"
  },
  "training": {
    "method": "lora",
    "output_dir": "./test_results",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-05,
    "weight_decay": 0.01,
    "lora": {
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.05,
      "target_modules": ["c_attn"]
    }
  }
}
```

## Project Structure

### API Directory
- **api.py**: REST API implementation
- **apiserver.py**: REST server
- **apiclient.py**: API client for remote access

### Jobs Directory
- **jobqueuesystem**: SQLite3-based job queue system

### SDK Directory
- **entities.py**: Primary entity definitions
- **managers.py**: Entity management classes
- **sdk.py**: Main SDK module
- **finetuner.py**: Fine-tuning implementation
- **finetunercli.py**: Command-line interface
- **externaltools.py**: Open source command-line library support
- **utils.py**: Utility functions

## Version History
### v0.2 
- Added direct fine-tuning capabilities to the CLI
- Improved dataset handling and model support
- Added logging system for fine-tuning
- Focused on causal language models for simplified fine-tuning workflow

### v0.1
- Initial SDK entity review
- Specification adjustment to align with SDK