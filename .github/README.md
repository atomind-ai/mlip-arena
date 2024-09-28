<div align="center">
    <h1>MLIP Arena</h1>
    <a href="https://zenodo.org/doi/10.5281/zenodo.13704399"><img src="https://zenodo.org/badge/776930320.svg" alt="DOI"></a>
    <a href="https://huggingface.co/spaces/atomind/mlip-arena"><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" style="height: 20px; background-color: white;" alt="Hugging Face"></a>
</div>

> [!CAUTION]
> MLIP Arena is currently in pre-alpha. The results are not stable. Please intepret them with care. 

> [!NOTE]
> If you're interested in joining the effort, please reach out to Yuan at [cyrusyc@berkeley.edu](mailto:cyrusyc@berkeley.edu).

MLIP Arena is an open-source platform for benchmarking machine learning interatomic potentials (MLIPs). The platform provides a unified interface for users to evaluate the performance of their models on a variety of tasks, including single-point density functional theory calculations and molecular dynamics simulations. The platform is designed to be extensible, allowing users to contribute new models, benchmarks, and training data to the platform.

## Contribute

MLIP Arena is now in pre-alpha. If you're interested in joining the effort, please reach out to Yuan at [cyrusyc@berkeley.edu](mailto:cyrusyc@berkeley.edu). See [project page](https://github.com/orgs/atomind-ai/projects/1) for some outstanding tasks. 

### Development

```
streamlit run serva/app.py
```

### Add new MLIP models

If you have pretrained MLIP models that you would like to contribute to the MLIP Arena and show benchmark in real-time, there are two ways:

#### Hugging Face Model (recommended, difficult)

0. Inherit Hugging Face [ModelHubMixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins) class to your awesome model class definition. We recommend [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin).
1. Create a new [Hugging Face Model](https://huggingface.co/new) repository and upload the model file using [push_to_hub function](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin.push_to_hub).
2. Follow the template to code the I/O interface for your model [here](../mlip_arena/models/README.md). 
3. Update model [registry](../mlip_arena/models/registry.yaml) with metadata

> [!NOTE] 
> CPU benchmarking will be performed automatically. Due to the limited amount GPU compute, if you would like to be considered for GPU benchmarking, please create a pull request to demonstrate the offline performance of your model (published paper or preprint). We will review and select the models to be benchmarked on GPU.

#### External ASE Calculator (easy)

1. Implement new ASE Calculator class in [mlip_arena/models/external.py](../mlip_arena/models/externals.py). 
2. Name your class with awesome model name and add the same name to [registry](../mlip_arena/models/registry.yaml) with metadata.

> [!CAUTION] 
> Remove unneccessary outputs under `results` class attributes to avoid error for MD simulations. Please refer to other class definition for example.

### Add new benchmark tasks

1. Follow the task template to implement the task class and upload the script along with metadata to the MLIP Arena [here](../mlip_arena/tasks/README.md).
2. Code a benchmark script to evaluate the performance of your model on the task. The script should be able to load the model and the dataset, and output the evaluation metrics.

### Add new datasets

1. Create a new [Hugging Face Dataset](https://huggingface.co/new-dataset) repository and upload the reference data (e.g. DFT, AIMD, experimental measurements such as RDF).


#### Single-point density functional theory calculations

- [ ] MPTrj
- [ ] [Alexandria](https://huggingface.co/datasets/atomind/alexandria)
- [ ] QM9

#### Molecular dynamics calculations

- [ ] [MD17](http://www.sgdml.org/#datasets)
- [ ] [MD22](http://www.sgdml.org/#datasets)


### [Hugging Face Auto-Train](https://huggingface.co/docs/hub/webhooks-guide-auto-retrain)

Planned but not yet impelemented.