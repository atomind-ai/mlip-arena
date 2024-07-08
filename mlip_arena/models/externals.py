import os
import urllib
from typing import Literal

import torch
from alignn.ff.ff import AlignnAtomwiseCalculator, get_figshare_model_ff
from ase import Atoms
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet
from fairchem.core import OCPCalculator
from mace.calculators import MACECalculator


# Avoid circular import
def get_freer_device() -> torch.device:
    """Get the GPU with the most free memory, or use MPS if available.
    s
        Returns:
            torch.device: The selected GPU device or MPS.

        Raises:
            ValueError: If no GPU or MPS is available.
    """
    device_count = torch.cuda.device_count()
    if device_count > 0:
        # If CUDA GPUs are available, select the one with the most free memory
        mem_free = [
            torch.cuda.get_device_properties(i).total_memory
            - torch.cuda.memory_allocated(i)
            for i in range(device_count)
        ]
        free_gpu_index = mem_free.index(max(mem_free))
        device = torch.device(f"cuda:{free_gpu_index}")
        print(
            f"Selected GPU {device} with {mem_free[free_gpu_index] / 1024**2:.2f} MB free memory from {device_count} GPUs"
        )
    elif torch.backends.mps.is_available():
        # If no CUDA GPUs are available but MPS is, use MPS
        print("No GPU available. Using MPS.")
        device = torch.device("mps")
    else:
        # Fallback to CPU if neither CUDA GPUs nor MPS are available
        print("No GPU or MPS available. Using CPU.")
        device = torch.device("cpu")

    return device


class MACE_MP_Medium(MACECalculator):
    def __init__(self, device=None, default_dtype="float32", **kwargs):
        checkpoint_url = "http://tinyurl.com/5yyxdm76"
        cache_dir = os.path.expanduser("~/.cache/mace")
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint_url) if c.isalnum() or c in "_"
        )
        cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
        if not os.path.isfile(cached_model_path):
            os.makedirs(cache_dir, exist_ok=True)
            # download and save to disk
            print(f"Downloading MACE model from {checkpoint_url!r}")
            _, http_msg = urllib.request.urlretrieve(checkpoint_url, cached_model_path)
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Model download failed, please check the URL {checkpoint_url}"
                )
            print(f"Cached MACE model to {cached_model_path}")
        model = cached_model_path
        msg = f"Using Materials Project MACE for MACECalculator with {model}"
        print(msg)

        device = device or str(get_freer_device())

        super().__init__(
            model_paths=model, device=device, default_dtype=default_dtype, **kwargs
        )


class CHGNet(CHGNetCalculator):
    def __init__(
        self,
        model: CHGNet | None = None,
        use_device: str | None = None,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> None:
        use_device = use_device or str(get_freer_device())
        super().__init__(
            model=model,
            use_device=use_device,
            stress_weight=stress_weight,
            on_isolated_atoms=on_isolated_atoms,
            **kwargs,
        )

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list | None = None,
        system_changes: list | None = None,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        # for ase.io.write compatibility
        self.results.pop("crystal_fea", None)


class EquiformerV2(OCPCalculator):
    def __init__(
        self,
        model_name="EquiformerV2-lE4-lF100-S2EFS-OC22",
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            local_cache=local_cache,
            cpu=cpu,
            seed=0,
            **kwargs,
        )

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        self.results.update(
            force=atoms.get_forces(),
        )


class eSCN(OCPCalculator):
    def __init__(
        self,
        model_name="eSCN-L6-M3-Lay20-S2EF-OC20-All+MD",
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=model_name,
            local_cache=local_cache,
            cpu=cpu,
            seed=0,
            **kwargs,
        )

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        self.results.update(
            force=atoms.get_forces(),
        )


class ALIGNN(AlignnAtomwiseCalculator):
    def __init__(self, dir_path: str = "/tmp/alignn/", device=None, **kwargs) -> None:
        model_path = get_figshare_model_ff(dir_path=dir_path)
        device = device or get_freer_device()
        super().__init__(model_path=model_path, device=device, **kwargs)

    def calculate(self, atoms, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
