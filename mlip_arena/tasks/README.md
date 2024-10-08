## Note on task registration

1. Use `ast` to parse task classes from the uploaded script.
2. Add the classes and their supported tasks to the task registry file `registry.yaml`.
3. Run tests on HF Space to ensure the task is working as expected.
4. [Push task script to the Space](https://huggingface.co/docs/huggingface_hub/guides/upload) and sync with github repository.
5. Create task folder in [mlip-arena](https://huggingface.co/datasets/atomind/mlip-arena) HF Dataset.
6. 