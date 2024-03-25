# mlip-arena

MLIP Arena is an open-source platform for benchmarking machine learning interatomic potentials (MLIPs). The platform provides a unified interface for users to evaluate the performance of their models on a variety of tasks, including single-point density functional theory calculations and molecular dynamics simulations. The platform is designed to be extensible, allowing users to contribute new models, benchmarks, and training data to the platform.

## Contribute

### Add new MLIP models

If you have pretrained MLIP models that you would like to contribute to the MLIP Arena and show benchmark in real-time, please follow these steps:

1. Create a new [Hugging Face Model](https://huggingface.co/new) repository and upload the model file.
2. Follow the template to code the I/O interface for your model, and upload the script along with metadata to the MLIP Arena [here]().
3. CPU benchmarking will be performed automatically. Due to the limited amount GPU compute, if you would like to be considered for GPU benchmarking, please create a pull request to demonstrate the offline performance of your model (published paper or preprint). We will review and select the models to be benchmarked on GPU.

### Add new benchmarks

#### Molecular dynamics calculations


#### Single-point density functional theory calculations

### Add new training datasets

[Hugging Face Auto-Train](https://huggingface.co/docs/hub/webhooks-guide-auto-retrain)


