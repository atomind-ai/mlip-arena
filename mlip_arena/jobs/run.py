import importlib

from mlip_arena.models import REGISTRY as MODEL_REGISTRY
from mlip_arena.tasks import REGISTRY as TASK_REGISTRY

print(MODEL_REGISTRY)
print(TASK_REGISTRY)

for task, metadata in TASK_REGISTRY.items():

    print(f"mlip_arena.tasks.{task}")
    module = importlib.import_module(f"mlip_arena.tasks.{task}")
    module.whoami()