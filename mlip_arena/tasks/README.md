

## Task

In the language of Prefect workflow manager, we define a task as *one operation on one input structure* that generates result for **one sample**. For example, [Structure optimization (OPT)](optimize.py) initiates one structure optimization on one structure and return the relaxed structure. 

It is possible to chain multiple subtasks into a single, complex task. For example, [Equation of states (EOS)](eos.py) first performs one full relaxed [OPT](optimize.py) task and parallelizes/serializes multiple constrained [OPT](optimize.py) tasks in one call, and returns the equation of state and bulk modulus of the structure.

There are some general tasks that can be reused:
- [Structure optimization (OPT)](optimize.py)
- [Molecular dynamics (MD)](md.py)
- [Equation of states (EOS)](eos.py)

## Flow

Flow is meant to be used to parallize multiple tasks and be orchestrated for production at scale on high-throughput cluster.


<!-- ## Note on task registration

1. Use `ast` to parse task classes from the uploaded script.
2. Add the classes and their supported tasks to the task registry file `registry.yaml`.
3. Run tests on HF Space to ensure the task is working as expected.
4. [Push task script to the Space](https://huggingface.co/docs/huggingface_hub/guides/upload) and sync with github repository.
5. Create task folder in [mlip-arena](https://huggingface.co/datasets/atomind/mlip-arena) HF Dataset.
6.  -->