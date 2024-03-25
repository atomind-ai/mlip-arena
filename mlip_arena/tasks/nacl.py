
import yaml
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator

from mlip_arena.models import REGISTRY_FILE

yaml.load(open(REGISTRY_FILE), Loader=yaml.FullLoader)