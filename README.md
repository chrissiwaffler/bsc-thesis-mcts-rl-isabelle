# Isabella: MCTS + LLM + Isabelle + RL for Automated Theorem Proving

This repository contains the implementation for a Bachelor thesis combining Monte Carlo Tree Search (MCTS), Large Language Models (LLMs), Isabelle theorem prover, and Reinforcement Learning (RL) for automated theorem proving.

## Project Structure

The repository consists of two main components:

### 1. Beam Search Implementation (`beamsearch/`)
- Core beam search algorithm for proof generation
- Client configuration and proof state management
- Data handling and few-shot examples
- Utility functions for I/O, parsing, and visualization

### 2. MCTS Training Loop (`mcts/`)
- Complete MCTS implementation with reinforcement learning
- Integration with Isabelle theorem prover
- Training pipeline with PPO (Proximal Policy Optimization)
- Comprehensive test suite for validation
- Logging and experiment tracking with Weights & Biases

### Submodules
- `miniF2F/`: OpenAI's miniF2F benchmark for formal mathematics
- `miniF2F-facebook/`: Facebook's miniF2F implementation with fixes
- `qisabelle/`: Python-Isabelle integration server (custom fork)

### Results
- `final_results.tar.gz`: Compressed experimental results (not publicly accessible)

## Installation

### Prerequisites
- Python >= 3.10
- [UV](https://docs.astral.sh/uv/) - Fast Python package manager
- Isabelle theorem prover (native or Docker)
- Docker (for qisabelle server)

### Setup

```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd bsc_thesis_mcts_rl_isabelle

# Install dependencies
uv sync --no-build-isolation

# Install qisabelle in editable mode
uv pip install -e ./qisabelle
```

### Isabelle Configuration

```bash
# Set Isabelle PATH (macOS example)
export PATH="/Applications/Isabelle2023.app/bin:$PATH"

# Verify installation
which isabelle
```

## Usage

### Running MCTS Training

```bash
# Configure Ray runtime (required for distributed training)
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

# Run MCTS training pipeline
uv run python mcts/launcher.py
```

### Running Tests

```bash
# Run all tests
uv run pytest mcts/tests/

# Run specific test categories
uv run pytest mcts/tests/test_mcts_mock.py
uv run pytest mcts/tests/test_isabelle_interface.py
```

### GPU Configuration

GPU allocation for MCTS rollouts can be configured in `mcts/launcher.py`:
- `policy_vllm_cuda_visible_devices`: Policy model GPU allocation
- `value_vllm_cuda_visible_devices`: Value model GPU allocation

For GRPO training GPU configuration, use HuggingFace Accelerate:
```bash
uv run accelerate config
```

## Development

### Code Quality

```bash
# Run all quality checks
uv run ruff format . && uv run ruff check . --fix && uv run mypy mcts/ beamsearch/

# Individual checks
uv run ruff format .
uv run ruff check . --fix
uv run mypy mcts/ beamsearch/
```

### Environment Variables

Create a `.env` file based on `.env.example`:
```bash
OPENAI_API_KEY=your-key
ANTHROPIC_API_KEY=your-key  # Optional
```

## Troubleshooting

### Restarting Services
```bash
# Stop Ray completely
uv run ray stop --force

# Restart Docker instances
docker-compose down && docker-compose up -d
```

### Nix Development Shell (Optional)

For isolated development environment:
```bash
# Enter isolated Nix shell
nix --extra-experimental-features nix-command --extra-experimental-features flakes develop --ignore-env

# Or with system packages
nix develop
```

## Architecture Overview

### MCTS Component
- Tree search with intelligent backtracking
- Policy and value networks for proof guidance
- Integration with Isabelle for proof verification
- Reinforcement learning with PPO for model improvement

### Beam Search Component
- Parallel proof exploration with configurable beam width
- Multiple temperature settings for exploration
- Probability calculation combining LLM confidence and Isabelle verification
- Visualization and tracking capabilities

### Integration Points
- LLM providers (OpenAI, Anthropic) via standard APIs
- Isabelle via qisabelle HTTP server
- Local models via Hugging Face transformers
- Results tracking and experiment management

## Citation

If you use this work in your research, please cite it appropriately according to your institution's guidelines.