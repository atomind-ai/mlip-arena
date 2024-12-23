"""
Generates a database of special quasi-random structures (SQS) from a template structure.

This script utilizes the `structuretoolkit <https://github.com/pyiron/structuretoolkit/tree/main>`_
to call `sqsgenerator <https://sqsgenerator.readthedocs.io/en/latest/index.html#>`_ to generate
SQS structures. The generated structures are saved to an ASE database file and optionally uploaded
to the Hugging Face Hub.

References
~~~~~~~~~~
- Alvi, S. M. A. A., Janssen, J., Khatamsaz, D., Perez, D., Allaire, D., & Arroyave, R. (2024).
  Hierarchical Gaussian Process-Based Bayesian Optimization for Materials Discovery in High
  Entropy Alloy Spaces. *arXiv preprint arXiv:2410.04314*.
- Gehringer, D., Friák, M., & Holec, D. (2023). Models of configurationally-complex alloys made
  simple. *Computer Physics Communications, 286*, 108664.

Authors
~~~~~~~
- Jan Janssen (`@jan-janssen <https://github.com/jan-janssen>`_)
- Yuan Chiang (`@chiang-yuan <https://github.com/chiang-yuan>`_)
"""

import os
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from prefect import task
from tqdm.auto import tqdm

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
        print(f"{db_path.name} uploaded to {repo_id}/{subfolder}")
    
    return db_path

@task
def get_atoms_from_db(
    db_path: Path | str,
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
        )
    with connect(db_path) as db:
        for row in db.select():
            yield row.toatoms()


def body_order(n=32, b=5):
    """
    Generate all possible combinations of atomic counts for `b` species
    that sum to `n`.
    """
    if b == 2:
        return [[i, n - i] for i in range(n + 1)]
    return [[i] + j for i in range(n + 1) for j in body_order(n=n - i, b=b - 1)]


def generate_sqs(structure_template, elements, counts):
    """
    Generate a special quasi-random structure (SQS) based on mole fractions.
    """
    import structuretoolkit as stk

    mole_fractions = {
        el: c / len(structure_template) for el, c in zip(elements, counts)
    }
    return stk.build.sqs_structures(
        structure=structure_template,
        mole_fractions=mole_fractions,
    )[0]


def get_endmember(structure, conc_lst, elements):
    """
    Assign a single element to all atoms in the structure to create an endmember.
    """
    structure.symbols[:] = np.array(elements)[conc_lst != 0][0]
    return structure


def generate_alloy_db(
    structure_template: Atoms,
    elements: list[str],
    db_path: Path | str,
    upload: bool = True,
    hf_token: str | None = os.getenv("HF_TOKEN", None),
    repo_id: str = "atomind/mlip-arena",
    repo_type: str = "dataset",
) -> Path:
    
    if upload and hf_token is None:
        raise ValueError("HF_TOKEN is required to upload the database.")
    
    num_atoms = len(structure_template)
    num_species = len(elements)

    # Generate all possible atomic configurations
    configurations = np.array(body_order(n=num_atoms, b=num_species))

    # Prepare the database
    db_path = (
        Path(db_path) or Path(__file__).resolve().parent / f"sqs_{'-'.join(elements)}.db"
    )
    db_path.unlink(missing_ok=True)

    atoms_list = []
    for i, composition in tqdm(
        enumerate(configurations), total=len(configurations)
    ):
        # Skip trivial cases where only one element is present
        if sum(composition == 0) != len(elements) - 1:
            atoms = generate_sqs(
                structure_template=structure_template,
                elements=np.array(elements)[composition != 0],
                counts=composition[composition != 0],
            )
        else:
            atoms = get_endmember(
                structure=structure_template.copy(),
                conc_lst=composition,
                elements=elements,
            )
        atoms_list.append(atoms)


    return save_to_db(
        atoms_list=atoms_list,
        db_path=db_path,
        upload=upload,
        hf_token=hf_token,
        repo_id=repo_id,
        repo_type=repo_type,
    )
