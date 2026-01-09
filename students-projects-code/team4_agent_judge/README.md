# Team 4 - Agent Judge Prompt Optimization

This folder contains a Jupyter notebook that explores prompt optimization using two LLM agents: one agent rewrites a prompt, and a second agent judges which rewrite is better. The workflow evaluates whether the optimized prompt helps a base model answer correctly on reasoning datasets.

## Folder contents
- team4_DanielAnanya_Final.ipynb: Main notebook with the full pipeline, evaluation loops, and sample outputs.
- Team4_FinalReport.pdf: Project report.
- Team4_ProjectPresentation.pptx: Slides.

## What the notebook does
1. Installs dependencies and sets up Ollama-backed LLMs.
2. Loads datasets (BBH boolean expressions, LiveBench reasoning, and a medical reasoning set).
3. Builds a prompt optimization agent (Llama 3.1) and a judge agent (Phi 4).
4. Runs a baseline pass where the model answers without optimization.
5. Runs a multi-agent loop (MAS) that iteratively improves prompts until the model answers correctly or a max loop count is reached.
6. Reports success metrics and shows example outputs.

## Ollama installation and usage
The notebook uses `langchain-ollama`, which expects a local Ollama server. Install and run Ollama before executing the notebook.

### Install Ollama
- macOS (Homebrew):

```
brew install ollama
```

- Linux:

```
curl -fsSL https://ollama.com/install.sh | sh
```

- Windows: Download the installer from https://ollama.com/download and follow the prompts.

### Start the Ollama server and pull models
In a terminal:

```
ollama serve
ollama pull llama3.1:8b
ollama pull phi4:14b
ollama list
```

The notebook references these model names directly, so update the names if you use different models.

### Quick usage check
You should be able to run:

```
ollama run llama3.1:8b "Say hello"
```

If this works, the notebook can connect to Ollama.

## Dependencies
The notebook installs many packages inline. If you want a local environment, install at least:

```
pip install langchain langchain-ollama datasets tabulate matplotlib
```

Optional but helpful:

```
pip install colab-xterm
```

## Running the notebook
1. Start Ollama (`ollama serve`) and make sure the models are pulled.
2. Open `team4_DanielAnanya_Final.ipynb` in Jupyter or Colab.
3. Run cells in order from top to bottom.

### Key sections in the notebook
- Prompt optimizer and judge setup: defines system prompts and tools.
- MAS Test Run: quick sanity check of the optimizer/judge loop.
- Baseline: evaluates without prompt optimization.
- MAS Loop: iteratively improves prompts and reports metrics.

## Notes and gotchas
- Some evaluation steps are manual (`mode="manual"`) and require user input for correctness.
- The optimizer uses a `step_by_step` tool to add reasoning instructions when needed.
- If Ollama is running on a non-default host or port, set `OLLAMA_HOST` before launching the notebook.

