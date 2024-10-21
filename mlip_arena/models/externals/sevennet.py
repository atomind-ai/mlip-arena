from __future__ import annotations

from sevenn.sevennet_calculator import SevenNetCalculator

from mlip_arena.models.utils import get_freer_device


class SevenNet(SevenNetCalculator):
    def __init__(
        self,
        checkpoint="7net-0",  # TODO: import from registry
        device=None,
        **kwargs,
    ):
        device = device or get_freer_device()
        super().__init__(checkpoint, device=device, **kwargs)
