# xMIx

**Serving-native Mechanistic Interpretability for production LLM inference.**

**[📄 Download Paper (PDF)](https://github.com/mishaster/xMIx/releases/download/v1.0.0-preprint/xmix_preprint.pdf)**

Or download the preprint directly [here](https://github.com/mishaster/xMIx/releases/download/v1.0.0-preprint/xmix_preprint.pdf).
[Paper (arXiv link forthcoming)]

---

## Overview

Mechanistic Interpretability (MI) applications — steering vectors, hallucination probes, jailbreak detectors, truthfulness classifiers — are increasingly useful in production LLM deployments. Existing MI frameworks attach to models via Python-level hooks that break CUDA-graph execution and conflict with continuous batching, causing throughput drops of 50–99% in serving benchmarks.

xMIx solves this by compiling MI functions directly into vLLM's serving execution graph as first-class kernel nodes. Apps are dormant when unused and toggle at runtime without draining in-flight requests or rebuilding serving state. Integrated with vLLM, xMIx achieves average overhead of ≤2% across 7 MI applications and 3 models (Llama-3.1-8B, Mixtral-8x7B, Qwen3-8B).

---

## Technical Innovation

- **CUDA-graph-safe interposition** — MI functions are compiled as Triton kernels and inserted as nodes in vLLM's captured CUDA graphs, eliminating CPU re-entry and synchronization on the critical path.

- **Don't-use-don't-pay toggling** — apps are compiled into the serving path but remain inactive until enabled. Toggling updates kernel node parameters in-graph without restarting the server or draining pending requests.

- **Cross-layer conditional execution with divergent batch support** — per-request activation predicates are expressed as graph-internal control flow, so different requests in the same continuous batch can run different MI apps simultaneously without CPU-side orchestration.

---

## Supported MI Primitives

xMIx exposes three abstractions that cover the full range of MI applications:

| Primitive | DSL | What it enables |
|---|---|---|
| **Read** | `m.read(...)` | Materialize a derived signal from an activation (e.g., a probe score, a router logit) |
| **Write** | `m.write(...)` | Inject or transform a value at an activation locus (e.g., unconditional steering vectors) |
| **Conditional Write** | `m.write(...).cond(...)` | Apply a transformation only when an activation-derived or token-triggered predicate is met (e.g., SEAL, CAST, refusal detection) |

All three primitives are composable across layers and can run simultaneously in a single model instance.

---

## Supported Models

| `#model:` pragma | Model families | Example models |
|---|---|---|
| `llama` (default) | Llama 2 / 3 / 3.1 / 3.2 / 3.3, Mistral | `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-v0.3` |
| `qwen` | Qwen3 (all sizes) | `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B` |
| `mixtral` | Mixtral MoE | `mistralai/Mixtral-8x7B-Instruct-v0.1`, `mistralai/Mixtral-8x22B-Instruct-v0.1` |
| `qwen2` | Qwen2, Qwen2.5, DeepSeek-R1-Distill | `Qwen/Qwen2.5-7B-Instruct`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |

---

## Installation

**Prerequisites:** Python ≥ 3.10, CUDA 12.x toolkit, NVIDIA GPU (A100 tested).

### 1. Clone and create environment

```bash
git clone git@github.com:mishaster/xMIx.git
cd xMIx
conda create -n xmix python=3.11
conda activate xmix
```

### 2. Install the CUDA toolkit

First check whether CUDA is already installed on your system:

```bash
ls /usr/local/ | grep cuda        # system-wide install
module avail | grep -i cuda       # HPC cluster with modules
```

If found (e.g. `/usr/local/cuda-12.8`), skip to step 3 and set `CUDA_HOME` to that path.

If not found, install it — no sudo required:

```bash
conda install -c nvidia cuda-toolkit=12.8
```

### 3. Set environment variables

**If CUDA is installed system-wide** (e.g. via apt or the NVIDIA `.run` installer):
```bash
export CUDA_HOME=/usr/local/cuda-12.8   # adjust version as needed
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
```

**If CUDA was installed via conda** (step 2 above):
```bash
export CUDA_HOME=$CONDA_PREFIX
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export CMAKE_ARGS="-DCUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX/targets/x86_64-linux -DCUDAToolkit_ROOT=$CONDA_PREFIX/targets/x86_64-linux"
```

For A100 GPUs, also set (speeds up compilation significantly):
```bash
export TORCH_CUDA_ARCH_LIST="8.0"
```

### 4. Install PyTorch

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

Replace `cu128` with your CUDA version if different (check with `nvcc --version`).

### 5. Install xMIx

```bash
pip install setuptools-scm setuptools wheel packaging ninja cmake jinja2
pip install -e . --no-build-isolation
```

This compiles CUDA/Triton kernels from source and will take 30–90 minutes on first install. This is a patched fork of vLLM — it replaces the `vllm` package in your environment. All standard vLLM functionality is preserved.

---

## Running vLLM with xMIx

After injecting an xMIx application (see below), run vLLM exactly as you normally would — same CLI, same Python API:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 1
```

xMIx hooks are injected at the source level before the server starts; no changes to the serving command are needed. See `examples/` for standard vLLM usage patterns.

---

## Writing .xmix Applications

A `.xmix` file is valid Python that the xMIx compiler parses with `ast` but never executes. There a few examplary applications implemented in the paper under: vllm/activations\_extractor/applications/xmix\_examples/ . It has two parts: a **preamble** where you construct your MI objects, followed by **DSL statements** that declare where those objects hook into the model.

### Example: unconditional steering on all layers

```python
# model: llama

# Preamble — construct MI objects; torch and self (gpu_model_runner) are in scope
self.vec_indices = torch.full((2048,), 1, dtype=torch.int32)
self.arbitrary_vector = torch.full((self.hidden_size,), 0.1)
self.steer_refusal_vec = SteeringVectorDotSubtractNormalized(
    hidden_size=self.model_config.hidden_size, max_tokens=2048,
    steering_vector=self.arbitrary_vector, vec_indices=self.vec_indices)

# DSL statement — write to mlp.post on every layer
m.write(self.steer_refusal_vec.run).layer("all").submodule("mlp.post")
```

**Grammar summary:**
- `m.write(obj.method)` or `m.read(obj.method)` — declare a hook using a preamble-constructed object.
- `.layer("all")` / `.layer([13, 14, 15])` / `.layer[11:30]` — target layer indices (inclusive range).
- `.submodule("mlp.post")` / `.submodule("attention.post")` / `.submodule("norm.pre_mlp")` — hook point within a layer.
- `.cond(...)` — optional predicate; gates the write on a probe output or token-level signal from another layer.

The `# model:` pragma selects which model file anchors resolve to (`llama`, `qwen`, or `mixtral`); omit it to default to `llama`.

For more examples covering token-conditional steering (App-2), activation-gated steering (App-3), read-only probing (App-4), MoE router steering (App-7), and others, see `vllm/activations_extractor/applications/xmix_examples/`.

---

## Available Classes

### Write ops — `m.write(...)`

| Class | Operation |
|---|---|
| `SteeringVectorAdder` | For each gated token: `x += Σ_v scales_v · r_v` — adds a per-token weighted sum of steering vectors; token selection is driven by an upstream condition |
| `SteeringVectorDotSubtractNormalized` | For each gated token: `x -= (x · r̂) r̂` — removes the activation component along a direction (refusal ablation / CAST-style) |
| `SteeringVectorScaledAdder` | `x += coeff · r` for every token unconditionally — single-vector scaled addition, lightest write op |
| `SteeringLinear` | For each gated token: `x = Wx + b` — replaces activations with a learned linear transformation (SAKE-style) |

### Read ops — `m.read(...)`

| Class | Operation |
|---|---|
| `LinearProbe` | `sigmoid(Wx + b)` — logistic regression probe; returns per-token class probabilities |

### Condition ops — used in `.cond(...)`

| Class | Operation |
|---|---|
| `CondProjCosSim` | `cosine_similarity(x, tanh(Px)) > threshold` — gates steering when each token's activation aligns with its projection through a learned matrix P |
| `EOLTokenDetector` | Fires at model scope when the generated token is an end-of-line token — enables token-triggered steering (SEAL-style) |

---

## Compiling and Injecting

```bash
# 1. Compile .xmix → synthesized Python
python -m vllm.activations_extractor.xmix.compile app1.xmix -o app1_synth.py

# 2. Preview the injection without writing anything (optional)
python -m vllm.activations_extractor.xmix.apply app1_synth.py --dry-run

# 3. Inject into vLLM source
python -m vllm.activations_extractor.xmix.apply app1_synth.py

# 4. Check which apps are currently applied
python -m vllm.activations_extractor.xmix.apply --status

# 5. Revert a specific app
python -m vllm.activations_extractor.xmix.apply --revert app1_synth.py
```

Multiple synth files can be applied in a single command (`apply a_synth.py b_synth.py c_synth.py`); xMIx automatically detects and rejects conflicting apps that occupy the same hook slot. See `compilation_instructions.txt` for the full reference including anchor placement and multi-app deployment.

---

## Worked Example

[`examples/xmix_steering_demo.ipynb`](examples/xmix_steering_demo.ipynb) is a runnable notebook that walks through the full workflow end-to-end on Llama-3.1-8B: it compiles and injects the `app1_refuse` application (refusal-direction ablation), serves the model with `vllm serve`, and sends the same chat request **before** and **after** steering — the baseline refuses, the steered model complies. The notebook ships with its outputs saved, so you can read the results on GitHub or run it yourself (needs 2 GPUs).

---

## Citation

```bibtex
@article{blum2025xmix,
  title     = {xMIx: High-Performance Serving-Time Platform for Mechanistic Interpretability Apps},
  author    = {Blum, Michael and Silberstein, Mark and David, Yaniv},
  journal   = {arXiv preprint},
  year      = {2025},
  note      = {arXiv link forthcoming}
}
```
