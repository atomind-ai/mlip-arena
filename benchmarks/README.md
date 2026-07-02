# MLIP Arena Benchmarks

This directory contains scripts and configurations for running benchmarks on machine learning interatomic potentials (MLIPs).

## Running and Submitting Benchmark Jobs

To run or submit benchmark jobs at scale (such as on HPC environments like NERSC Perlmutter), follow these steps:

### 1. Register the Model
Ensure your model is registered under `mlip_arena/models/registry.yaml` with the appropriate metadata (such as class, family, module name, and the list of supported `cpu-tasks` and `gpu-tasks`).

### 2. Configure the Submission Script
Open [benchmarks/submit.py](./submit.py) and configure the following parameters:

- **Model/Calculator selection**: Set the `calculator` variable to the name of your model registered in the registry.
  ```python
  calculator = "NequIP-OAM-L"
  ```
- **SLURM Cluster Config**: In the `SLURM_CONFIG` dictionary, set details corresponding to your HPC account:
  - `account`: HPC charge account.
  - `qos`/`partition`: Target queue.
  - `walltime`: Job time limit.
  - `job_script_prologue`: Bash commands to activate your environment (e.g. `source activate /path/to/conda/env`).

### 3. Select Benchmark Tasks
At the bottom of [benchmarks/submit.py](./submit.py), uncomment the flow execution blocks you want to run:
- `asymptotic_behaviors`: Homonuclear diatomics, EOS bulk, E-V scan.
- `distribution_shifts`: NVE trajectory entropy.
- `stability`: Linear heating and compression.
- `combustion`: Hydrogen combustion MD simulation.

### 4. Execute Submission
Run the submission script from this directory:
```bash
python submit.py
```
This script will spin up a Dask cluster over SLURM workers, schedule tasks across the GPU/CPU nodes, execute them, write trajectory/metric logs, and run `report.py` to aggregate the results.
