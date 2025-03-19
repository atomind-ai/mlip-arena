from pathlib import Path

from loguru import logger

license_path = Path(__file__).parent / "LICENSE"

logger.info(f"""
This module {__name__} is kindly shared by @jan-janssen <https://github.com/jan-janssen>. If you use this module, you agree to cite the reference
    
- Alvi, S. M. A. A., Janssen, J., Khatamsaz, D., Perez, D., Allaire, D., & Arroyave, R. (2024). Hierarchical Gaussian Process-Based Bayesian Optimization for Materials Discovery in High Entropy Alloy Spaces. *arXiv preprint arXiv:2410.04314*.
- Gehringer, D., Friák, M., & Holec, D. (2023). Models of configurationally-complex alloys made simple. *Computer Physics Communications, 286*, 108664.
""")