from __future__ import annotations

from nequip.ase import NequIPCalculator
from ase.data import chemical_symbols
from mlip_arena.models.utils import get_freer_device


class NequIP_OAM_L(NequIPCalculator):
    nequip_net_id = "nequip.net:mir-group/NequIP-OAM-L:0.1"

    def __new__(cls, device: str | None = None, **kwargs):
        device = device or get_freer_device()
        return super()._from_saved_model(cls.nequip_net_id, device=device)
