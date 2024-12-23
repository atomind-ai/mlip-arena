<div align="center">
    <h1>MLIP Arena</h1>
    <a href="https://zenodo.org/doi/10.5281/zenodo.13704399"><img src="https://zenodo.org/badge/776930320.svg" alt="DOI"></a>
    <a href="https://huggingface.co/spaces/atomind/mlip-arena"><img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg" style="height: 20px; background-color: white;" alt="Hugging Face"></a>
    <!-- <a href="https://discord.gg/W8WvdQtT8T"><img alt="Discord" src="https://img.shields.io/discord/1299613474820984832?logo=discord"> -->
</a>
</div>

> [!CAUTION]
> MLIP Arena is currently in pre-alpha. The results are not stable. Please intepret them with care. 

> [!NOTE]
> Contributions of new tasks are very welcome! If you're interested in joining the effort, please reach out to Yuan at [cyrusyc@berkeley.edu](mailto:cyrusyc@berkeley.edu). See [project page](https://github.com/orgs/atomind-ai/projects/1) for some outstanding tasks, or propose new one in [Discussion](https://github.com/atomind-ai/mlip-arena/discussions/new?category=ideas).

MLIP Arena is a platform for evaluating foundation machine learning interatomic potentials (MLIPs) beyond conventional energy and force error metrics. It focuses on revealing the underlying physics and chemistry learned by these models and assessing their performance in molecular dynamics (MD) simulations. The platform's benchmarks are specifically designed to evaluate the readiness and reliability of open-source, open-weight models in accurately reproducing both qualitative and quantitative behaviors of atomic systems.

## Installation

### From PyPI (without model running capability)

```bash
pip install mlip-arena
```

### From source

```bash
git clone https://github.com/atomind-ai/mlip-arena.git
cd mlip-arena
pip install torch==2.2.0
bash scripts/install-pyg.sh
bash scripts/install-dgl.sh
pip install -e .[test]
pip install -e .[mace]
# DeePMD
DP_ENABLE_TENSORFLOW=0 pip install -e .[deepmd]
```

## Contribute

MLIP Arena is now in pre-alpha. If you're interested in joining the effort, please reach out to Yuan at [cyrusyc@berkeley.edu](mailto:cyrusyc@berkeley.edu). 

### Development

```
streamlit run serve/app.py
```

### Add new benchmark tasks (WIP)

> [!NOTE]
> Please reuse or extend the general tasks defined as Prefect / [Atomate2](https://github.com/materialsproject/atomate2) / [Quacc](https://github.com/Quantum-Accelerators/quacc) workflow. 
> The following are some tasks implemented:
> - [Prefect structure optimization (OPT)](../mlip_arena/tasks/optimize.py)
> - [Prefect molecular dynamics (MD)](../mlip_arena/tasks/md.py)
> - [Prefect equation of states (EOS)](../mlip_arena/tasks/eos.py)

1. Follow the task template to implement the task class and upload the script along with metadata to the MLIP Arena [here](../mlip_arena/tasks/README.md).
2. Code a benchmark script to evaluate the performance of your model on the task. The script should be able to load the model and the dataset, and output the evaluation metrics.

### Add new MLIP models 

If you have pretrained MLIP models that you would like to contribute to the MLIP Arena and show benchmark in real-time, there are two ways:

#### External ASE Calculator (easy)

1. Implement new ASE Calculator class in [mlip_arena/models/externals](../mlip_arena/models/externals). 
2. Name your class with awesome model name and add the same name to [registry](../mlip_arena/models/registry.yaml) with metadata.

> [!CAUTION] 
> Remove unneccessary outputs under `results` class attributes to avoid error for MD simulations. Please refer to other class definition for example.

#### Hugging Face Model (recommended, difficult)

0. Inherit Hugging Face [ModelHubMixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins) class to your awesome model class definition. We recommend [PytorchModelHubMixin](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin).
1. Create a new [Hugging Face Model](https://huggingface.co/new) repository and upload the model file using [push_to_hub function](https://huggingface.co/docs/huggingface_hub/en/package_reference/mixins#huggingface_hub.ModelHubMixin.push_to_hub).
2. Follow the template to code the I/O interface for your model [here](../mlip_arena/models/README.md). 
3. Update model [registry](../mlip_arena/models/registry.yaml) with metadata

> [!NOTE] 
> CPU benchmarking will be performed automatically. Due to the limited amount GPU compute, if you would like to be considered for GPU benchmarking, please create a pull request to demonstrate the offline performance of your model (published paper or preprint). We will review and select the models to be benchmarked on GPU.



### Add new datasets

The "ultimate" goal is to compile the copies of all the open data in a unified format for lifelong learning with [Hugging Face Auto-Train](https://huggingface.co/docs/hub/webhooks-guide-auto-retrain). 

1. Create a new [Hugging Face Dataset](https://huggingface.co/new-dataset) repository and upload the reference data (e.g. DFT, AIMD, experimental measurements such as RDF).

#### Single-point density functional theory calculations

- [ ] MPTrj
- [ ] [Alexandria](https://huggingface.co/datasets/atomind/alexandria)
- [ ] QM9
- [ ] SPICE

#### Molecular dynamics calculations

- [ ] [MD17](http://www.sgdml.org/#datasets)
- [ ] [MD22](http://www.sgdml.org/#datasets)
