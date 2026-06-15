import torch

torch.serialization.add_safe_globals([slice])

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
