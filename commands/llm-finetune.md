# /llm-finetune

Prepare data and configure fine-tuning jobs for LLMs.

## Usage

```
/llm-finetune <dataset> --base-model <model> [--method lora|full] [--epochs 3] [--learning-rate 2e-5]
```

- `dataset`: path to training data (JSONL, CSV, or directory of text files)
- `--base-model`: base model to fine-tune (e.g., `gpt-4o-mini`, `llama-3-8b`, `mistral-7b`)
- `--method`: fine-tuning method (default: lora)
- `--epochs`: training epochs (default: 3)
- `--learning-rate`: learning rate (default: 2e-5 for LoRA, 5e-6 for full)

## Workflow

### Stage 0: Environment Check

1. Check if `ml_utils.py` exists in `src/` — if missing, copy from core plugin (`~/.claude/plugins/*/templates/ml_utils.py`)
2. Check if `llm_utils.py` exists in `src/` — if missing, copy from this plugin's `templates/llm_utils.py`
3. Verify dataset file/directory exists
4. Check for required packages based on method:
   - API fine-tuning (OpenAI): `openai`
   - Local LoRA: `transformers`, `peft`, `bitsandbytes`, `datasets`
   - Local full: `transformers`, `deepspeed` or `accelerate`

### Stage 1: Data Loading and Validation

1. Load dataset:
   - **JSONL**: expect `{"prompt": ..., "completion": ...}` or `{"messages": [...]}` format
   - **CSV**: expect `prompt` and `completion` columns
   - **Directory**: load text files as completion-only training data
2. Validate format:
   - Check required fields present
   - Flag empty or malformed entries
   - Check for duplicates (exact and near-duplicate via hash)
3. Report: sample count, avg prompt/completion length, format validation results

### Stage 2: Data Preparation

1. **Format conversion** — convert to target format:
   - OpenAI: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`
   - HuggingFace: `{"text": "<prompt>\n<completion>"}` or chat template format
2. **Train/validation split** — 90/10 stratified split (if no val set provided)
3. **Token analysis**:
   - Count tokens per sample using model tokenizer
   - Flag samples exceeding context window
   - Truncate or warn for over-length samples
4. **Data quality checks**:
   - PII detection (email, phone, SSN patterns)
   - Instruction-following quality (does completion match prompt intent)
   - Label consistency (for classification fine-tuning)
5. Save prepared datasets:
   - `data/train.jsonl`
   - `data/val.jsonl`
   - `data/data_report.json`

### Stage 3: Training Configuration

1. Generate training config based on `--method`:

   **LoRA Configuration:**
   ```json
   {
     "method": "lora",
     "base_model": "<model>",
     "lora_r": 16,
     "lora_alpha": 32,
     "lora_dropout": 0.05,
     "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
     "learning_rate": 2e-5,
     "epochs": 3,
     "batch_size": 4,
     "gradient_accumulation_steps": 4,
     "warmup_ratio": 0.1,
     "max_seq_length": 2048
   }
   ```

   **Full Fine-Tuning Configuration:**
   ```json
   {
     "method": "full",
     "base_model": "<model>",
     "learning_rate": 5e-6,
     "epochs": 3,
     "batch_size": 2,
     "gradient_accumulation_steps": 8,
     "warmup_ratio": 0.1,
     "weight_decay": 0.01,
     "fp16": true,
     "deepspeed_config": "ds_config.json"
   }
   ```

2. Estimate resources:
   - VRAM requirement based on model size and method
   - Training time estimate
   - Estimated cost (if using API)
3. Save config to `config/finetune_config.json`

### Stage 4: Training Script Generation

1. Generate `src/finetune.py` with:
   - Data loading and preprocessing
   - Model and tokenizer initialization
   - LoRA/full training setup
   - Training loop with validation logging
   - Checkpoint saving
   - Final model merge (for LoRA)
2. Generate `src/evaluate_finetuned.py` for post-training evaluation
3. For API fine-tuning (OpenAI):
   - Generate `src/submit_finetune_job.py` with file upload and job creation
   - Generate `src/monitor_finetune.py` for job status polling

### Stage 5: Report

```python
from ml_utils import save_agent_report
save_agent_report("prompt-engineer", {
    "status": "completed",
    "base_model": base_model,
    "method": method,
    "dataset": {
        "train_samples": train_count,
        "val_samples": val_count,
        "avg_tokens": avg_tokens,
        "data_issues": data_issues
    },
    "config_file": "config/finetune_config.json",
    "scripts": generated_scripts,
    "resource_estimate": resource_estimate,
    "recommendations": recommendations
})
```

Print: dataset summary, config highlights, resource estimates, next steps.
