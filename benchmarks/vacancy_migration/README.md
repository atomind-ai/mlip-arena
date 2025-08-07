# Vacancy migration (2.4)

## Run

Two Prefect flow `run_fcc()` and `run_hcp()` are provided in [run.py](run.py). To run the benchmark, execute `python run.py` in terminal or directly call the functions in a notebook.


The reference PBE data [^1] is provided in [Table-A1-fcc.csv](./Table-A1-fcc.csv) and [Table-A2-hcp.csv](./Table-A2-hcp.csv)


[^1]: Angsten, Thomas, et al. "Elemental vacancy diffusion database from high-throughput first-principles calculations for fcc and hcp structures." New Journal of Physics 16.1 (2014): 015018.

## Analysis

Run `python analysis.py` to analyze the results and generate the plots.