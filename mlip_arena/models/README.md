

## Note on model registration

1. Use `ast` to parse model classes from the uploaded script.
2. Add the classes and their supported tasks to the model registry file `registry.yaml`.
3. Run tests on HF Space to ensure the model is working as expected.
4. [Push files to the Hub](https://huggingface.co/docs/huggingface_hub/guides/upload) and sync with github repository.
5. Use [HF webhook](https://huggingface.co/docs/hub/en/webhooks) to check the status of benchmark tasks (pass, fail, null), run unfinisehd tasks and visualize the results on leaderboard. [[guide]](https://huggingface.co/docs/hub/en/webhooks-guide-metadata-review)