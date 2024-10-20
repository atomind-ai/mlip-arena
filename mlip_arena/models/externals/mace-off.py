from __future__ import annotations

import os
from pathlib import Path

from mace.calculators import MACECalculator

from mlip_arena.models.utils import get_freer_device


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