
import enum

from mlip_arena.models import MLIP
from mlip_arena.tasks import Task


class Machine(enum.Enum):
    """Enum class for machine"""
    HFCPU = "Hugging Face CPU Basic"
    PERLCPU = "NERSC Perlmutter CPU"
    PERLA100 = "NERSC Perlmutter A100 40GB"
    PERLA100L = "NERSC Perlmutter A100 80GB"

class Job:
    def __init__(self, model: MLIP, task: Task, machine: Machine, **kwargs):
        self.calculator = model
        self.task = task
        self.machine = machine
        self.kwargs = kwargs

    def __str__(self):
        return f"Job: {self.task.name} on {self.machine.value}"
    
    def run(self):
        if self.machine == Machine.HFCPU:
            print(f"Running {self.name} on {self.machine.value}")
            "run the task on Hugging Face CPU Basic"
            raise NotImplementedError
        elif self.machine == Machine.PERLCPU:
            print(f"Running {self.name} on {self.machine.value}")
            "send the task to NERSC Perlmutter CPU node and listen for the results"
            raise NotImplementedError
        elif self.machine == Machine.PERLA100:
            print(f"Running {self.name} on {self.machine.value}")
            "send the task to NERSC Perlmutter GPU node and listen for the results"
            raise NotImplementedError
        