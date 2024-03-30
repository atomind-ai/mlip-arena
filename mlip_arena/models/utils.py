import importlib
import os
from enum import Enum

from mlip_arena.models import REGISTRY

MLIPEnum = Enum(
    "MLIPEnum",
    {
        model: getattr(
            importlib.import_module(f"{__package__}.{metadata['module']}"), model
        )
        for model, metadata in REGISTRY.items()
    },
)
