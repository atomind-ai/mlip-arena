# Dynamical stability of 2D materials (A.10.2)

## Run

This benchmark requires parellel orchestration using Prefect utility. To run the benchmark, please modify the SLURM cluster setting (or change to desired job queing systems following [dask-jobqueuue documentation](https://jobqueue.dask.org/en/latest/)) in [run.py](run.py).


```
    nodes_per_alloc = 1
    gpus_per_alloc = 1
    ntasks = 1

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        shebang="#!/bin/bash",
        account="matgen",
        walltime="00:30:00",
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "module load cudatoolkit/12.4",
            "source activate /pscratch/sd/c/cyrusyc/.conda/mlip-arena",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J eos_bulk",
            "-q regular",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
            # "--exclusive",
        ],
    )
```

## Analysis

The example analysis code is provided in [analysis.ipynb](analysis.ipynb)