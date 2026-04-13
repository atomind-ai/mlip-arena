# Contribute and Development

PRs are welcome. Please clone the repo and submit PRs with changes.

To make change to huggingface space, fetch large files from git lfs first and run streamlit:

```bash
git lfs fetch --all
git lfs pull
streamlit run serve/app.py
```

## Add new MLIP models

If you have pretrained MLIP models that you would like to contribute to the MLIP Arena and show benchmark in real-time, there are two ways:

### External ASE Calculator (easy)

1. Implement new ASE Calculator class in `mlip_arena/models/externals`.
2. Name your class with awesome model name and add the same name to `registry` with metadata.

> **Caution:** Remove unneccessary outputs under `results` class attributes to avoid error for MD simulations. Please refer to `CHGNet` as an example.

### Hugging Face Model (recommended, difficult)

0. Inherit Hugging Face `ModelHubMixin` class to your awesome model class definition. We recommend `PyTorchModelHubMixin`.
1. Create a new Hugging Face Model repository and upload the model file using `push_to_hub` function.
2. Follow the template to code the I/O interface for your model.
3. Update model `registry` with metadata

## Add new benchmark

Please reuse, extend, or chain the general tasks defined and add new folder and script under `/benchmarks`.

## Documentation Development

We use [Mintlify](https://mintlify.com/) for our documentation website under the `docs/` folder.

To preview documentation changes locally:

1. Install Mintlify CLI globally (if you haven't already):
   ```bash
   npm i -g mint
   ```

   > **Note on Node.js Version**: Mintlify requires **Node.js 20.17 or higher**. If your system Node is too old, you can install the latest LTS version locally using `nvm` (Node Version Manager).

   > **Note on HPC environments (`npm error errno -122`)**:
   > If you encounter a `Disk Quota Exceeded` error (-122) when installing `npm` packages or `nvm`, your home directory is likely out of space. You can work around this by installing `nvm` and `npm` cache into your scratch space:
   > ```bash
   > # 1. Set NVM to install in scratch
   > export NVM_DIR="$SCRATCH/.nvm"
   > curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
   > source $NVM_DIR/nvm.sh
   >
   > # 2. Install latest LTS node
   > nvm install --lts
   > nvm use --lts
   >
   > # 3. Forward npm cache to scratch as well
   > mkdir -p $SCRATCH/.npm-cache
   > npm config set cache $SCRATCH/.npm-cache
   >
   > # 4. Finally, install mint
   > npm i -g mint
   > ```

2. Run the development server from the `docs/` directory:
   ```bash
   cd docs
   mint dev
   ```

3. View your documentation at `http://localhost:3000`.
