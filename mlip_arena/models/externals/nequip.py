"""NequIP and Allegro model integrations."""

from __future__ import annotations

import torch

torch.serialization.add_safe_globals([slice])

import e3nn  # noqa: E402

_orig_set_opt = e3nn.set_optimization_defaults
e3nn.set_optimization_defaults = lambda **kwargs: _orig_set_opt(**{k: v for k, v in kwargs.items() if k != "jit_mode"})

from nequip.integrations.ase import NequIPCalculator  # noqa: E402

from mlip_arena.models.utils import get_freer_device  # noqa: E402


class NequIP_OAM_L:
    def __new__(cls, device: str | None = None, **kwargs):
        device = device or get_freer_device()
        nequip_net_id = "nequip.net:mir-group/NequIP-OAM-L:0.1"
        return NequIPCalculator._from_saved_model(
            nequip_net_id,
            device=device,
            # chemical_species_to_atom_type_map=True
        )
