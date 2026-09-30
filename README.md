# FM4017 Project - MP-08-26

*From Data to Decisions: Developing Vibration-Based Predictive Maintenance for Hydropower Plants*

---

## Overview

This repository contains the Python building blocks for investigating vibration-based predictive maintenance at the Grunnåi hydropower plant. The analysis combines operating signals, bearing measurements, and turbine data to study generator behaviour and expected vibration.

The repository currently contains data-loading, preprocessing, feature-engineering, and model prototypes. The files in `src/example/` are scaffolding for the planned training workflow; they are not yet a complete command-line application.

## Setup

### Prerequisites

- Python 3.10 or newer. Python 3.11 is the reference version for the original environment.
- Git
- Visual Studio Code is recommended, but not required.

### Clone the repository and create an environment

From PowerShell:

```powershell
git clone <repository-url>
cd fm4017-grunnai-pdm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current user and activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.\.venv\Scripts\Activate.ps1
```

On Linux or macOS, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

Upgrade pip and install the project dependencies from the repository root:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install makes the local `src` package importable regardless of
the current working directory. Use module-style execution for files inside
the package, for example:

```powershell
python -m src.data_analysis.sampling_rate
```

Avoid running package files by path, such as
`python src/data_analysis/sampling_rate.py`; that changes Python's import path
and can make `import src...` fail. If the editable install is not available,
run commands from the repository root instead.

Check the installed packages with:

```powershell
python -m pip check
```

For the team workflow, including the VS Code interpreter setup, import rules,
dependency changes, and pull-request checks, see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Data setup

The project data should be placed under `data/raw/`; do not commit any data to the repository. 

Use a folder for each measurement period, for example:

```text
data/
├── raw/
│   ├── 2024-06-01_2024-10-31/
│   │   └── <signal CSV files>
│   └── 2025-06-01_2025-10-31/
│       └── <signal CSV files>
└── processed/
	└── <date-range-and-processing-method>/
		└── <processed files>
```

The available signal names, descriptions, units, and data-quality warnings are documented in [docs/dataset.md](docs/dataset.md). In particular:

- Exclude `Hydrocord.Eta.G2_ETA_Value(=D1=LE1=LE1).csv` from analysis because its values are known to be faulty.
- Generator 1 active-power values slightly below zero near standstill can be ignored.
- Generator 1 turbine speed is stored as a percentage, not RPM; short-lived values above 100% may occur.
- The injector signals describe five Pelton injectors, not wicket gates.

## Verify the checkout

Run the test suite from the repository root:

```
python -m pytest
```

The tests currently describe an in-progress API and may require source/test alignment before they pass: `tests/test_data_loader.py` uses a different `DataLoader` constructor and method signature than `src/data_loader.py`, and it references validation data that is not part of the documented `data/` layout. Treat failures as development work rather than evidence that the environment itself is broken.

You can still verify that the Python source compiles with:

```powershell
python -m compileall src
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/` | Python package and project configuration |
| `src/data_loader.py` | CSV loading helper |
| `src/example/` | Prototype preprocessing, feature, model, training, validation, and prediction modules |
| `tests/` | Unit-test prototypes |
| `data/raw/` | Unmodified measurement exports |
| `data/processed/` | Derived datasets and intermediate outputs |
| `docs/dataset.md` | Signal catalogue and data-quality notes |
| `requirements.txt` | Runtime Python dependencies |

## Current workflow status

The intended workflow is:

1. Acquire the raw SCADA and Hydrocord exports and the signal metadata file.
2. Store raw files by measurement period under `data/raw/`.
3. Load and clean the data.
4. Engineer operating-condition and vibration features.
5. Train, validate, and use a predictive model.

The orchestration file `src/example/main.py` is currently a placeholder, and the example training/model files still contain incomplete interfaces and assumptions. There is therefore no supported end-to-end `python -m ...` training command yet. Use the modules as development references and update this section when the pipeline has a stable entry point.

## Development notes

- Run commands from the repository root so relative paths resolve consistently.
- Keep generated datasets in `data/processed/` and keep raw exports unchanged.
- Update `docs/dataset.md` when new signals or periods are added.
- Add or update tests when a data-loading or preprocessing contract changes.

## References

- [Dataset and sensor documentation](docs/dataset.md)
- [NumPy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [ruptures](https://centre-borelli.github.io/ruptures-docs/)
- [dcor](https://dcor.readthedocs.io/)
- [ydata-profiling](https://docs.profiling.ydata.ai/)
