from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import matgl
import requests
import torch
from alignn.ff.ff import AlignnAtomwiseCalculator, get_figshare_model_ff, default_path
from ase import Atoms
from chgnet.model.dynamics import CHGNetCalculator
from chgnet.model.model import CHGNet as CHGNetModel
from fairchem.core import OCPCalculator
from mace.calculators import MACECalculator
from matgl.ext.ase import PESCalculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from sevenn.sevennet_calculator import SevenNetCalculator
from deepmd.calculator import DP as DPCalculator

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
    def __init__(
        self,
        checkpoint="http://tinyurl.com/5yyxdm76",
        device: str | None = None,
        default_dtype="float32",
        **kwargs,
    ):
        cache_dir = Path.home() / ".cache" / "mace"
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint) if c.isalnum() or c in "_"
        )
        cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
        if not os.path.isfile(cached_model_path):
            import urllib

            os.makedirs(cache_dir, exist_ok=True)
            _, http_msg = urllib.request.urlretrieve(checkpoint, cached_model_path)
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Model download failed, please check the URL {checkpoint}"
                )
        model = cached_model_path

        device = device or str(get_freer_device())

        super().__init__(
            model_paths=model, device=device, default_dtype=default_dtype, **kwargs
        )


# TODO: could share the same class with MACE_MP_Medium
class MACE_OFF_Medium(MACECalculator):
    def __init__(
        self,
        checkpoint="https://github.com/ACEsuit/mace-off/raw/main/mace_off23/MACE-OFF23_medium.model?raw=true",
        device: str | None = None,
        default_dtype="float32",
        **kwargs,
    ):
        cache_dir = Path.home() / ".cache" / "mace"
        checkpoint_url_name = "".join(
            c for c in os.path.basename(checkpoint) if c.isalnum() or c in "_"
        )
        cached_model_path = f"{cache_dir}/{checkpoint_url_name}"
        if not os.path.isfile(cached_model_path):
            import urllib

            os.makedirs(cache_dir, exist_ok=True)
            _, http_msg = urllib.request.urlretrieve(checkpoint, cached_model_path)
            if "Content-Type: text/html" in http_msg:
                raise RuntimeError(
                    f"Model download failed, please check the URL {checkpoint}"
                )
        model = cached_model_path

        device = device or str(get_freer_device())

        super().__init__(
            model_paths=model, device=device, default_dtype=default_dtype, **kwargs
        )


class CHGNet(CHGNetCalculator):
    def __init__(
        self,
        checkpoint: CHGNetModel | None = None,  # TODO: specifiy version
        device: str | None = None,
        stress_weight: float | None = 1 / 160.21766208,
        on_isolated_atoms: Literal["ignore", "warn", "error"] = "warn",
        **kwargs,
    ) -> None:
        use_device = device or str(get_freer_device())
        super().__init__(
            model=checkpoint,
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


class M3GNet(PESCalculator):
    def __init__(
        self,
        checkpoint="M3GNet-MP-2021.2.8-PES",
        # TODO: cannot assign device
        state_attr: torch.Tensor | None = None,
        stress_weight: float = 1.0,
        **kwargs,
    ) -> None:
        potential = matgl.load_model(checkpoint)
        super().__init__(potential, state_attr, stress_weight, **kwargs)


class EquiformerV2(OCPCalculator):
    def __init__(
        self,
        checkpoint="EquiformerV2-lE4-lF100-S2EFS-OC22",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=checkpoint,
            local_cache=local_cache,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        self.results.update(
            force=atoms.get_forces(),
        )


class EquiformerV2OC20(OCPCalculator):
    def __init__(
        self,
        checkpoint="EquiformerV2-31M-S2EF-OC20-All+MD",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=checkpoint,
            local_cache=local_cache,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )


class eSCN(OCPCalculator):
    def __init__(
        self,
        checkpoint="eSCN-L6-M3-Lay20-S2EF-OC20-All+MD",  # TODO: import from registry
        # TODO: cannot assign device
        local_cache="/tmp/ocp/",
        cpu=False,
        seed=0,
        **kwargs,
    ) -> None:
        super().__init__(
            model_name=checkpoint,
            local_cache=local_cache,
            cpu=cpu,
            seed=seed,
            **kwargs,
        )

    def calculate(self, atoms: Atoms, properties, system_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        self.results.update(
            force=atoms.get_forces(),
        )


class ALIGNN(AlignnAtomwiseCalculator):
    def __init__(self, device=None, **kwargs) -> None:
        # TODO: cannot control version
        # _ = get_figshare_model_ff(dir_path=dir_path)
        model_path = default_path()

        device = device or get_freer_device()
        super().__init__(path=model_path, device=device, **kwargs)


class SevenNet(SevenNetCalculator):
    def __init__(
        self,
        checkpoint="7net-0",  # TODO: import from registry
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()
        super().__init__(checkpoint, device=device, **kwargs)


class ORB(ORBCalculator):
    def __init__(
        self,
        checkpoint="orbff-v1-20240827.ckpt",
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()

        cache_dir = Path.home() / ".cache" / "orb"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = cache_dir / "orbff-v1-20240827.ckpt"

        url = f"https://storage.googleapis.com/orbitalmaterials-public-models/forcefields/{checkpoint}"

        if not ckpt_path.exists():
            print(f"Downloading ORB model from {url} to {ckpt_path}...")
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                with open(ckpt_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download completed.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError("Failed to download ORB model.") from e

        orbff = pretrained.orb_v1(weights_path=ckpt_path, device=device)
        super().__init__(orbff, device=device, **kwargs)

class DeepMD(DPCalculator):
    def __init__(
        self,
        checkpoint="dp0808c_v024mixu.pth",
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()

        cache_dir = Path.home() / ".cache" / "deepmd"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / checkpoint

        url = f"https://bohrium-api.dp.tech/ds-dl/mlip-arena-tfpk-v1.zip"

        if not model_path.exists():
            import zipfile

            print(f"Downloading DeepMD model from {url} to {model_path}...")
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                with open(cache_dir/"temp.zip", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download completed.")
                with zipfile.ZipFile(cache_dir/"temp.zip", "r") as zip_ref:
                    zip_ref.extractall(cache_dir)
                print("Unzip completed.")
            except requests.exceptions.RequestException as e:
                raise RuntimeError("Failed to download DeepMD model.") from e

        
        super().__init__(model_path, device=device, **kwargs)