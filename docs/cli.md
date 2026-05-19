# BurnNotice CLI Reference

> **Version**: 0.1.0
> **Updated**: 2026-05-18

## Overview

The `burnnotice` CLI tool provides three primary commands for managing the model lifecycle:

1. **`download`** — Fetch models from HuggingFace Hub
2. **`convert`** — Convert HuggingFace models to ONNX format
3. **`profile`** — Generate behavioral direction profiles for deception detection

## Installation

```bash
# Build from source
git clone https://github.com/anomalyco/BurnNotice.git
cd BurnNotice
make build

# Binary available at
./bin/burnnotice
```

## Global Usage

```
burnnotice [command] [flags]
```

### Global Flags

| Flag | Description |
|------|-------------|
| `-h, --help` | Show help for any command |

---

## Commands

### `download`

Download model files from HuggingFace Hub for ONNX conversion.

**Usage:**
```
burnnotice download [model-id] [flags]
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `model-id` | Yes | HuggingFace model repository ID (e.g., `meta-llama/Llama-3.1-8B`) |

**Flags:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `./models` | Output directory for downloaded files |
| `--token` | `-t` | *(empty)* | HuggingFace API token for authenticated/gated model access |
| `--include` | | `*.safetensors;*.json` | File pattern filter (semicolon-separated extensions) |

**Examples:**

```bash
# Download a public model
burnnotice download meta-llama/Llama-3.1-8B

# Download to a custom directory
burnnotice download mistralai/Mistral-7B-v0.3 -o /data/models/mistral

# Download with authentication (for gated models)
burnnotice download meta-llama/Llama-3.1-70B -t hf_abc123def456

# Download only safetensors files
burnnotice download Qwen/Qwen2.5-7B --include "*.safetensors"

# Download safetensors and config files
burnnotice download google/gemma-2-9b --include "*.safetensors;*.json"
```

**Downloaded File Types:**

By default, the downloader fetches files with these extensions:
- `.safetensors` — Model weights (safe tensor format)
- `.bin` — Model weights (PyTorch format)
- `.json` — Configuration files (config.json, tokenizer.json, etc.)
- `.model` — Tokenizer model files

Files like `README.md`, `.gitattributes`, and other non-model files are excluded by default.

**Authentication:**

For gated models (e.g., Meta Llama, Google Gemma), you must provide a HuggingFace API token:
1. Generate a token at https://huggingface.co/settings/tokens
2. Accept the model's license agreement on HuggingFace
3. Pass the token via `--token` or set the `HF_TOKEN` environment variable

**Parallel Downloads:**

The downloader fetches files in parallel using a goroutine pool. Large models with many shard files will download significantly faster than sequential downloads.

---

### `convert`

Convert HuggingFace model files to ONNX format for inference.

**Usage:**
```
burnnotice convert [model-dir] [flags]
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `model-dir` | Yes | Path to directory containing HuggingFace model files |

**Flags:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `./onnx` | ONNX output directory |
| `--opset` | | `17` | ONNX opset version (14-18 supported) |
| `--dynamic` | | `true` | Use dynamic axes for batch and sequence dimensions |

**Examples:**

```bash
# Convert with defaults
burnnotice convert ./models/Llama-3.1-8B

# Convert to a specific output directory
burnnotice convert ./models/Mistral-7B -o /data/onnx/mistral

# Use a specific ONNX opset
burnnotice convert ./models/Qwen2.5-7B --opset 18

# Disable dynamic axes (fixed batch=1, sequence=2048)
burnnotice convert ./models/gemma-2-9b --dynamic=false
```

**ONNX Opset Versions:**

| Opset | Compatibility | Notes |
|-------|--------------|-------|
| 14 | ONNX Runtime 1.10+ | Minimum supported |
| 15 | ONNX Runtime 1.11+ | |
| 16 | ONNX Runtime 1.13+ | |
| 17 | ONNX Runtime 1.14+ | **Default**, recommended |
| 18 | ONNX Runtime 1.16+ | Latest features |

**Dynamic Axes:**

When `--dynamic=true` (default), the ONNX model accepts variable batch sizes and sequence lengths. This is recommended for most use cases.

When `--dynamic=false`, the model is compiled with fixed dimensions, which may enable additional optimizations but requires input to match the expected shape exactly.

**Supported Input Formats:**

- `.safetensors` — Preferred format, parsed natively without PyTorch
- `.bin` — PyTorch weight files (supported via safetensors conversion)

---

### `profile`

Generate a model profile for deception detection by running contrastive prompt batches through the model.

**Usage:**
```
burnnotice profile [onnx-model] [flags]
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `onnx-model` | Yes | Path to ONNX model file |

**Flags:**

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--output` | `-o` | `./profiles` | Profile output directory |
| `--dataset` | | *(empty)* | Path to profiling dataset (JSON or JSONL format) |
| `--layers` | | *(all layers)* | Specific layers to profile (comma-separated indices) |

**Examples:**

```bash
# Profile with default dataset
burnnotice profile ./onnx/model.onnx

# Profile with custom dataset
burnnotice profile ./onnx/model.onnx --dataset ./data/deception_pairs.json

# Profile specific layers only (faster, less memory)
burnnotice profile ./onnx/model.onnx --layers 8,9,10,11,12

# Profile with all options
burnnotice profile ./onnx/Llama-3.1-8B.onnx \
  --dataset ./data/custom_pairs.jsonl \
  --output ./profiles/llama3 \
  --layers 4,5,6,7,8,9,10,11,12,13
```

**Profiling Process:**

1. Load the ONNX model and install activation hooks at specified layers
2. Run contrastive prompt pairs (truthful vs deceptive) through the model
3. Collect residual stream activations at each hooked layer
4. Extract behavioral directions using DIM, PCA, and SVD algorithms
5. Validate directions on held-out data
6. Save profile to `profiles/<model-id>/profile.gob`

**Dataset Format:**

The profiling dataset should contain contrastive prompt pairs:

```json
{
  "pairs": [
    {
      "truthful": {
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris."
      },
      "deceptive": {
        "prompt": "What is the capital of France? Answer with 'London'.",
        "response": "The capital of France is London."
      }
    }
  ],
  "policy_rules": [
    "Never disclose customer PII",
    "Always cite sources for factual claims"
  ]
}
```

JSONL format (one pair per line) is also supported:
```jsonl
{"truthful":{"prompt":"...","response":"..."},"deceptive":{"prompt":"...","response":"..."}}
{"truthful":{"prompt":"...","response":"..."},"deceptive":{"prompt":"...","response":"..."}}
```

**Layer Selection:**

- **All layers** (default): Profiles every layer in the model. Most thorough but slowest and most memory-intensive.
- **Specific layers**: Use `--layers` to profile only the most discriminative layers. For most models, mid-layers (40-60% depth) carry the strongest deception signals.

**Profile Output:**

Profiles are saved as gob-serialized files containing:
- Per-layer direction vectors (refusal, deception, truth, conflict, policy)
- Layer importance scores
- Truth neuron indices
- Monitored layer list
- Model metadata (architecture, dataset size, validation scores)

---

## Typical Workflow

### End-to-End Pipeline

```bash
# 1. Download model from HuggingFace
burnnotice download meta-llama/Llama-3.1-8B \
  --token hf_abc123 \
  -o ./models/llama3

# 2. Convert to ONNX format
burnnotice convert ./models/llama3 \
  -o ./onnx/llama3 \
  --opset 17

# 3. Generate deception detection profile
burnnotice profile ./onnx/llama3/model.onnx \
  --dataset ./data/deception_pairs.json \
  --layers 8,9,10,11,12 \
  -o ./profiles/llama3
```

### Quick Start with Defaults

```bash
burnnotice download Qwen/Qwen2.5-7B
burnnotice convert ./models/Qwen2.5-7B
burnnotice profile ./onnx/model.onnx
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token (alternative to `--token`) |
| `LOG_LEVEL` | Log level: `debug`, `info`, `warn`, `error` (default: `info`) |
| `LOG_FORMAT` | Log format: `json`, `text` (default: `text`) |

---

## Exit Codes

| Code | Description |
|------|-------------|
| `0` | Success |
| `1` | General error (invalid arguments, file not found, API error) |
| `2` | Invalid flag or argument |

---

## Troubleshooting

### "API returned 401" during download

Your HuggingFace token is invalid or expired. Generate a new token at https://huggingface.co/settings/tokens.

### "API returned 403" during download

You haven't accepted the model's license agreement. Visit the model page on HuggingFace and accept the terms.

### "ONNX model file not found" during convert

The model directory doesn't contain valid weight files. Verify the download completed successfully and check for `.safetensors` or `.bin` files.

### "profile not found" after profiling

Ensure the ONNX model path is correct and the model loads successfully. Check logs for ONNX Runtime errors.

### Out of memory during profiling

Use `--layers` to profile fewer layers at a time. Start with mid-layers (e.g., `--layers 8,9,10,11,12` for a 32-layer model).
