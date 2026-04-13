from pathlib import Path

import yaml

try:
    from prefect.logging import get_run_logger

    logger = get_run_logger()
except (ImportError, RuntimeError):
    from loguru import logger

try:
    from .elasticity import run as ELASTICITY
    from .eos import run as EOS
    from .md import run as MD
    from .neb import run as NEB
    from .neb import run_from_endpoints as NEB_FROM_ENDPOINTS
    from .optimize import run as OPT
    from .phonon import run as PHONON

    __all__ = ["ELASTICITY", "EOS", "MD", "NEB", "NEB_FROM_ENDPOINTS", "OPT", "PHONON"]
except (ImportError, TypeError, NameError) as e:
    logger.warning(e)


with open(Path(__file__).parent / "registry.yaml", encoding="utf-8") as f:
    REGISTRY = yaml.safe_load(f)
