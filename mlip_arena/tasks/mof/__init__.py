from pathlib import Path
from loguru import logger

license_path = Path(__file__).parent / "LICENSE"

logger.info(f"""
The module '{__name__}' is adapted from the repository: https://github.com/hspark1212/DAC-SIM. 
By using this module, you agree to the terms and conditions specified in the following license: 

https://github.com/hspark1212/DAC-SIM/blob/main/LICENSE

Additionally, please ensure proper attribution by citing the reference: 

Lim, Y., Park, H., Walsh, A., & Kim, J. (2024). Accelerating CO₂ Direct Air Capture Screening for Metal-Organic Frameworks with a Transferable Machine Learning Force Field.

A local copy of the LICENSE file can be found at: {license_path}.
""")
