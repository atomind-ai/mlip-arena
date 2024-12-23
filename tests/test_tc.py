
import sys

import pytest
from ase.build import bulk
from prefect import flow
from prefect.testing.utilities import prefect_test_harness

from mlip_arena.models import MLIPEnum
from mlip_arena.tasks.thermal_conductivity import run as TC