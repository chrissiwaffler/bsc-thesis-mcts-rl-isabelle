# CLAUDE.md

IMPORTANT: when you want to run a python file, please do so with the package manager `uv` inorder to have access to all installed dependencies

Note to you, the agent implementing changes on this codebase: please make use of the fact that you can invoke the following subagents by calling them with `@` in your message.

Don't put icons or emojies in any files as decoration!
Start single line comments with a lower case letter
For MCTS: the value model is also a LLM but only generates the value which will be parsed and converted to a valid float.

Instead of creating test script to test things, create a proper pytest inside a tests folder, if existing and it makes sense for the current test.

- debugger:
- implementation-documenter
- python-type-review

call these agents after you've implemented a bigger section to review the implementatin, update the documentation of it or if you've encounter problems to debug them.
example: `@python-type-reviewer please review my changes`.

## Project Overview

Isabella is a Bachelor thesis project combining Monte Carlo Tree Search (MCTS), Large Language Models (LLMs), Isabelle theorem prover, and Reinforcement Learning (RL) for automated theorem proving.

## Common Development Commands

### Environment Setup

```bash
# Install dependencies using uv (10-100x faster than pip)
uv sync

# Install with development tools (linting, testing, etc.)
uv sync --extra dev

# Set up environment variables
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"  # Optional
export PATH="/Applications/Isabelle2023.app/bin:$PATH"  # For macOS
```

### Code Quality & Linting

The project uses **ruff** for linting and formatting (from the same company that makes UV):

```bash
# Run all quality checks with one command
uv run python scripts/check_quality.py

# Or run individual checks:
uv run ruff format .              # Format code
uv run ruff check . --fix         # Lint with auto-fix
uv run mypy step_by_step_proof/   # Type checking
uv run pytest                     # Run tests
```

### Running Tests

```bash
# Test simple theorem proving
uv run python test_simple.py --test simple

# Compare with baselines
uv run python test_simple.py --test baselines

# Test on miniF2F problem
uv run python test_simple.py --test minif2f

# Run all tests
uv run python test_simple.py --test all
```

### Running Benchmarks

```bash
# Run full benchmark (50 problems)
uv run python -m step_by_step_proof.benchmark --max-problems 50

# Compare different approaches
uv run python -m step_by_step_proof.benchmark \
    --approaches truncate_resume whole_proof \
    --models gpt-4o \
    --max-problems 20
```

### Isabelle Integration

```bash
# Start Isabelle server via Docker
docker-compose up

# Or start native Isabelle TCP server
start_isabelle_server  # from isabelle-client package
```

### MiniF2F Evaluation

```bash
# Generate proofs for miniF2F test set
uv run python miniF2F_openai/generate_proofs.py

# Evaluate generated proofs
uv run python miniF2F_openai/evaluate_proofs.py
```

### Beam Search Example

```bash
# Run the beam search implementation with Mirascope
uv run python step_by_step_proof/beamsearch.py
```

## Architecture Overview

### Core Components

1. **step_by_step_proof/**: Advanced proof generation system
   - Tree search with intelligent backtracking
   - Truncate-and-resume mechanism for handling long proofs
   - Auxiliary lemma generation capability
   - Multi-model support (OpenAI, Anthropic, local models)

2. **qisabelle/**: Python-Isabelle integration
   - HTTP server (Scala) managing Isabelle processes
   - Python client for API interaction
   - Docker-based deployment with pre-built AFP heaps
   - Sledgehammer integration for proof search

3. **miniF2F/**: Formal mathematics benchmark
   - Problems from mathematical olympiads (AMC, AIME, IMO)
   - Isabelle theorem statements in test/ and valid/ directories
   - Evaluation framework for measuring proof success rates

### Key Architectural Patterns

- **Tree Search**: The proof generation system explores multiple proof paths simultaneously, backtracking intelligently when stuck
- **Client-Server Architecture**: Isabelle runs as an HTTP server, allowing stateful interaction and theory management
- **State Management**: Proof states are tracked and named, allowing branching and exploration of different proof strategies
- **Performance Optimization**: Truncate-and-resume mechanism reduces token usage while improving success rates

### Integration Points

- LLM providers (OpenAI, Anthropic) via standard API clients
- Isabelle via HTTP API exposed by qisabelle server
- Local models via Hugging Face transformers
- Results tracking and benchmarking via structured output formats

## Development Notes

- The project uses Python ≥3.10 with type hints
- UV is used for package management (10-100x faster than pip)
- Ruff is used for linting and formatting (10-100x faster than black/flake8)
- Docker is required for reproducible Isabelle environments
- API keys should be configured via .env file or environment variables
- Isabelle heaps are large (~40GB) and need to be downloaded separately
- The truncate-and-resume approach is expected to improve success rates by ~40% over baseline methods

## Code Style Guidelines

- **Always use `uv run`** to execute Python scripts to ensure proper dependencies
- **Run linting before commits**: `uv run python scripts/check_quality.py`
- **Type hints required**: The project uses strict type checking with mypy
- **Lowercase comments**: Keep inline comments in lowercase (as per project style)
- **Imports**: Let ruff handle import sorting automatically
- **Line length**: 88 characters (ruff default)
- **NO emojis**: NEVER USE EMOJIS IN CODE, OUTPUT, OR PRINT STATEMENTS

## Mirascope Documentation

The project uses Mirascope for LLM integration. Full documentation is available in `./docs/mirascope/`:

- **quickstart.md**: Getting started with Mirascope basics
- **learn_mirascope.md**: Core concepts and patterns
- **calls.md**: Making LLM calls with different providers
- **prompts.md**: Prompt engineering with Mirascope
- **response_models.md**: Structured outputs with Pydantic models
- **tools.md**: Function calling and tool usage
- **streams.md**: Streaming responses
- **async.md**: Asynchronous operations
- **retries.md**: Retry mechanisms and error handling
- **output_parsers.md**: Parsing LLM outputs
- **json_mode.md**: JSON mode for structured outputs
- **chaining.md**: Chaining multiple LLM calls
- **agents.md**: Building agents with Mirascope
- **evals.md**: Evaluation framework
- **local_models.md**: Using local models

When working with Mirascope in this project, consult these documentation files for correct API usage and patterns.

## Recent Updates

### Beam Search Implementation (beamsearch.py)

- Implements beam search algorithm with configurable beam width
- Uses Mirascope for LLM integration with structured prompts
- Supports multiple temperatures [0.3, 0.6, 0.9, 1.2] for exploration
- Calculates probabilities using: p(n) = p(parent) × p(LLM) × r(isabelle)
- Includes visualization with probability tracking
- Features self-ask reasoning and ReAct patterns

### Development Tooling

- Added comprehensive linting setup with ruff
- Configured mypy for type checking
- Set up pytest with coverage reporting
- Created unified quality check script
- Added pre-commit hooks configuration

## Lilypad Documentation

Lilypad is an open-source prompt engineering framework for LLM observability and evaluation. Full documentation: `./docs/lilypad.md`

**Key Features**: Automatic tracing/versioning of LLM functions, cost/latency monitoring, systematic annotation, A/B testing

**Basic Usage**:

```python
import lilypad

lilypad.configure(auto_llm=True)  # Auto-trace LLM calls

@lilypad.trace(versioning="automatic")  # Version & trace function
def answer_question(question: str) -> str:
    # LLM call automatically captured
    return llm_response
```

**Potential for Isabella**: Track proof generation strategies, version prompts, monitor costs, build evaluation datasets from proof attempts
