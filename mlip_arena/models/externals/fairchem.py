from pathlib import Path

import yaml

try:
    from fairchem.core import OCPCalculator
except ImportError:
    from fairchem.core import FAIRChemCalculator as OCPCalculator

from huggingface_hub import hf_hub_download

with open(Path(__file__).parents[1] / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)


class eSEN(OCPCalculator):
    def __init__(
        self,
        checkpoint=REGISTRY["eSEN"]["checkpoint"],
        cache_dir=None,
        cpu=False,  # TODO: cannot assign device
        seed=0,
        **kwargs,
    ) -> None:
        # https://huggingface.co/facebook/OMAT24/resolve/main/esen_30m_oam.pt

        checkpoint_path = hf_hub_download(
            "fairchem/OMAT24",
            filename=checkpoint,
            revision="13ab5b8d71af67bd1c83fbbf53250c82cd87f506",
            cache_dir=cache_dir,
        )
        kwargs.pop("device", None)
        super().__init__(
            checkpoint_path=checkpoint_path,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )


class eqV2(OCPCalculator):
    def __init__(
        self,
        checkpoint=REGISTRY["eqV2(OMat)"]["checkpoint"],
        cache_dir=None,
        cpu=False,  # TODO: cannot assign device
        seed=0,
        **kwargs,
    ) -> None:
        """
        Initialize an eqV2 calculator.

        Parameters
        ----------
        checkpoint : str, default="eqV2_86M_omat_mp_salex.pt"
            The name of the eqV2 checkpoint to use.
        local_cache : str, default="/tmp/ocp/"
            The directory to store the downloaded checkpoint.
        cpu : bool, default=False
            Whether to run the model on CPU or GPU.
        seed : int, default=0
            The random seed for the model.

        Other Parameters
        ----------------
        **kwargs
            Any additional keyword arguments are passed to the superclass.
        """

        # https://huggingface.co/fairchem/OMAT24/resolve/main/eqV2_86M_omat_mp_salex.pt

        checkpoint_path = hf_hub_download(
            "fairchem/OMAT24",
            filename=checkpoint,
            revision="bf92f9671cb9d5b5c77ecb4aa8b317ff10b882ce",
            cache_dir=cache_dir,
        )
        kwargs.pop("device", None)
        super().__init__(
            checkpoint_path=checkpoint_path,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )
