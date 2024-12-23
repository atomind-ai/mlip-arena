from pathlib import Path
from loguru import logger

license_path = Path(__file__).parent / "LICENSE"

logger.info(f"""
The module '{__name__}' is adapted from the repository: https://github.com/MPA2suite/k_SRME. 
By using this module, you agree to the terms and conditions specified in the following license: 

https://github.com/MPA2suite/k_SRME/blob/main/LICENSE

Additionally, please ensure proper attribution by citing the following reference: 

- Póta, B., Ahlawat, P., Csányi, G., & Simoncelli, M. (2024). Thermal Conductivity Predictions with Foundation Atomistic Models. arXiv preprint arXiv:2408.00755.

A local copy of the LICENSE file can be found at: {license_path}.
""")