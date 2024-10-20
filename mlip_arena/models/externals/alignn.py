from __future__ import annotations

from alignn.ff.ff import AlignnAtomwiseCalculator, default_path

from mlip_arena.models.utils import get_freer_device


class ALIGNN(AlignnAtomwiseCalculator):
    def __init__(self, device=None, **kwargs) -> None:
        # TODO: cannot control version
        # _ = get_figshare_model_ff(dir_path=dir_path)
        model_path = default_path()

        device = device or get_freer_device()
        super().__init__(path=model_path, device=device, **kwargs)
