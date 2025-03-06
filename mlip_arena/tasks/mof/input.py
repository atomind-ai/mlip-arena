import os
from pathlib import Path
from typing import Generator, Iterable
from loguru import logger
from huggingface_hub import HfApi, hf_hub_download
from prefect import task

from ase import Atoms
from ase.db import connect


def save_to_db(
    atoms_list: list[Atoms] | Iterable[Atoms] | Atoms,
    db_path: Path | str,
    upload: bool = True,
    hf_token: str | None = os.getenv("HF_TOKEN", None),
    repo_id: str = "atomind/mlip-arena",
    repo_type: str = "dataset",
    subfolder: str = Path(__file__).parent.name,
):
    """Save ASE Atoms objects to an ASE database and optionally upload to Hugging Face Hub."""

    if upload and hf_token is None:
        raise ValueError("HF_TOKEN is required to upload the database.")

    db_path = Path(db_path)

    if isinstance(atoms_list, Atoms):
        atoms_list = [atoms_list]
    
    with connect(db_path) as db:
        for atoms in atoms_list:
            if not isinstance(atoms, Atoms):
                raise ValueError("atoms_list must contain ASE Atoms objects.")
            db.write(atoms)

    if upload:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo=f"{subfolder}/{db_path.name}",
            repo_id=repo_id,
            repo_type=repo_type,
        )
        logger.info(f"{db_path.name} uploaded to {repo_id}/{subfolder}")

    return db_path

@task
def get_atoms_from_db(
    db_path: Path | str,
    hf_token: str | None = os.getenv("HF_TOKEN", None),
    repo_id: str = "atomind/mlip-arena",
    repo_type: str = "dataset",
    subfolder: str = Path(__file__).parent.name,
) -> Generator[Atoms, None, None]:
    """Retrieve ASE Atoms objects from an ASE database."""
    db_path = Path(db_path)
    if not db_path.exists():
        db_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            subfolder=subfolder,
            filename=str(db_path),
            token=hf_token,
        )
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms()
