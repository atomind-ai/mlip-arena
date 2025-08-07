# Equation of state (EOS) benchmark on WBM structures (2.1)

The compiled WBM structures are available as ASE DB [here](../wbm_structures.db)

## Run

This benchmark requires parellel orchestration using Prefect utility. To run the benchmark, please modify the SLURM cluster setting (or change to desired job queing systems following [dask-jobqueuue documentation](https://jobqueue.dask.org/en/latest/)) in [run.py](run.py).


```
    nodes_per_alloc = 1
    gpus_per_alloc = 1
    ntasks = 1

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account="matgen",
        walltime="00:30:00",
        # job_cpu=128,
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/dev",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J c2db",
            "-q regular",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=25, maximum_jobs=50)
```

## Analysis

To analyze and gather the results, run [analyze.py](analyze.py) and generate summary. To plot the EOS curves, run [plot.py](plot.py)# Equation of state (EOS) benchmark on WBM structures

The compiled WBM structures are available as ASE DB file [here](../wbm_structures.db)

## Run

This benchmark requires parellel orchestration using Prefect utility. To run the benchmark, please modify the SLURM cluster setting (or change to desired job queing systems following [dask-jobqueuue documentation](https://jobqueue.dask.org/en/latest/)) in [run.py](run.py).


```
    nodes_per_alloc = 1
    gpus_per_alloc = 1
    ntasks = 1

    cluster_kwargs = dict(
        cores=1,
        memory="64 GB",
        processes=1,
        shebang="#!/bin/bash",
        account="matgen",
        walltime="00:30:00",
        # job_cpu=128,
        job_mem="0",
        job_script_prologue=[
            "source ~/.bashrc",
            "module load python",
            "source activate /pscratch/sd/c/cyrusyc/.conda/dev",
        ],
        job_directives_skip=["-n", "--cpus-per-task", "-J"],
        job_extra_directives=[
            "-J c2db",
            "-q regular",
            f"-N {nodes_per_alloc}",
            "-C gpu",
            f"-G {gpus_per_alloc}",
        ],
    )

    cluster = SLURMCluster(**cluster_kwargs)
    print(cluster.job_script())
    cluster.adapt(minimum_jobs=25, maximum_jobs=50)
```

## Analysis

To analyze and gather the results, run [analyze.py](analyze.py) and generate summary. To plot the EOS curves, run [plot.py](plot.py)