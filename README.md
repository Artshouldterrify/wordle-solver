# Wordle Solver Research

Code, inputs, and simulation outputs copied from the `worlde_v2` backup folder for the Wordle strategy comparison post.

## Contents

- `base_classes.py`, `wordle_feedback.py`, and `statistic_tracker.py`: game simulation, feedback, and metrics.
- `statistical_solver.py`, `entropy_solver.py`, `stubh_solver.py`, and `llm_solver.py`: the solvers compared in the post.
- `wordle_answers.txt` and `wordle_guesses.txt`: solution and allowed-guess lists.
- `unigram_freq.csv`, `common_bigrams.csv`, and `first_10_guesses.csv`: frequency and guess-ranking inputs.
- `guess_matrix.npz`: precomputed entropy feedback matrix used by `entropy_solver.py`.
- `usage_metadata.csv` and `saved_data/`: saved per-game results and LLM token-usage metadata.

The raw Kaggle human-game dataset cited for the human baseline was not present in the backup folder and is not included here. Refer to the post for its source and calculations.

## Setup

Use Python 3.10 or later:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run commands from this repository's root; the modules load their data using relative paths. `python base_classes.py` runs the configured exhaustive Stubh Solver simulation and writes its result under `saved_data/`. It can overwrite the matching saved result file. Importing `stubh_solver.py` downloads the NLTK `words` corpus if needed, so that operation requires network access.

The LLM solver additionally requires a running Ollama installation and the model configured by the caller. The included guess matrix is already built; running `python entropy_solver.py` regenerates it and may take time.
