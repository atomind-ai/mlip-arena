# MD Reactivity with Hydrogen Combustion (A.7)

The workflow and analysis are defined in a single Prefect flow, which can be imported by:

```python
from mlip_arena.flows.combustion import hydrogen_combustion

# Run the flow
hydrogen_combustion(
    run_dir=run_dir,
    calculator=model,  # MLIPEnum name, model name string, or ASE calculator object
)
```

See the [run.ipynb](./run.ipynb) notebook for details.
