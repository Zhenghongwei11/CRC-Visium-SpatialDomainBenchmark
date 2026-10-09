# Scientific environments

`analysis-python313.txt` records the package versions in the Python 3.13.7 environment used for the current regional statistics, simulations and readout-gene withholding analyses. Install it through the root `requirements.txt`.

Source refitting uses a separate Python 3.11 environment because the configured neural methods have older dependencies:

```bash
python3.11 -m venv .venv-source
source .venv-source/bin/activate
python -m pip install -r environments/neural-python311.txt
```

STAGATE also imports the compiled PyG extensions. Install wheels matching the installed PyTorch and CUDA build before fitting:

```bash
python - <<'PY'
import subprocess, sys, torch
version = torch.__version__.split('+')[0]
cuda = 'cpu' if torch.version.cuda is None else 'cu' + torch.version.cuda.replace('.', '')
wheel_url = f'https://data.pyg.org/whl/torch-{version}+{cuda}.html'
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', '-r', 'environments/pyg-extensions.txt', '--find-links', wheel_url], check=True)
PY
```

These source-environment specifications provide a proposed installation setup, rather than a complete record of the original Colab environment. The local validation of source fitting covers the four baseline families on M-ST-13 using the explicit native loader in the main analysis environment. A complete refit of all 22 sections with the neural methods and BayesSpace has not been validated in this local environment.

The configured STAGATE implementation is pinned to commit `ae1158ca8cf1eb6bb8ee198298552d44c9ac21db`; SpaGCN is 1.2.7. BayesSpace runs 1,000 iterations with gamma 3. The study uses K=4 and K=6, seeds 11, 23 and 37, and Ward graphs with 4, 6 and 8 neighbors. The full settings are in `config/analysis_pipeline/map_parameters.tsv`.

For BayesSpace, install R and the packages used by its wrapper:

```r
install.packages(c("BiocManager", "mclust", "hdf5r"))
BiocManager::install(c("SingleCellExperiment", "BayesSpace"))
```

R is also used to read the deposited GSE294385 annotation RDS files. That conversion uses base R and does not require BayesSpace. Fitting scripts check their dependencies and record the versions used; they do not install or replace packages during an analysis.
