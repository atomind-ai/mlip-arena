from __future__ import annotations

import torchani
from mlip_arena.models.utils import get_freer_device


class ANI2x:
    def __new__(cls, device: str | None = None, **kwargs):
        device = device or str(get_freer_device())
        model = torchani.models.ANI2x().to(device)
        return model.ase()
