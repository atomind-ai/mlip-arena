from pathlib import Path

from loguru import logger
from tqdm.auto import tqdm

from mlip_arena.models import REGISTRY, MLIPEnum
from mlip_arena.tasks.stability.analysis import gather_results
from mlip_arena.tasks.stability.data import get_atoms_from_db

if __name__ == "__main__":

    compositions = []
    sizes = []
    for atoms in tqdm(get_atoms_from_db("random-mixture.db")):
        if len(atoms) == 0:
            continue
        compositions.append(atoms.get_chemical_formula())

    for model in MLIPEnum:
        try:
            run_dir = Path(__file__).parent / f"{REGISTRY[model.name]['family']}"
            df = gather_results(run_dir, prefix=model.name, run_type="nvt")

            df = df[
                df["formula"].isin(compositions[:120])
            ].copy()  # tentatively we only take the first 120 structures

            assert len(df) > 0

            df.to_parquet(run_dir / f"{model.name}-heating.parquet", index=False)
        except Exception as e:
            logger.warning(f"Error processing model {model.name}: {e}")

    for model in MLIPEnum:
        try:
            run_dir = Path(__file__).parent / f"{REGISTRY[model.name]['family']}"
            df = gather_results(run_dir, prefix=model.name, run_type="npt")

            df = df[
                df["formula"].isin(compositions[:80])
            ].copy()  # tentatively we only take the first 80 structures

            assert len(df) > 0

            df.to_parquet(run_dir / f"{model.name}-compression.parquet", index=False)
        except Exception as e:
            logger.warning(f"Error processing model {model.name}: {e}")
