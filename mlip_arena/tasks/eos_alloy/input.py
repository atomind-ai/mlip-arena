"""
:Authors: Jan Janssen `@jan-jassen <https://github.com/jan-janssen>`_, Yuan Chiang `@chiang-yuan <https://github.com/chiang-yuan>`_
"""

import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import HfApi
from tqdm.auto import tqdm

from ase import Atoms
from ase.build import bulk
from ase.db import connect


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
    local_path: Path | None = None,
    upload: bool = True,
    repo_id: str = "atomind/mlip-arena",
) -> Path:
    # Load Hugging Face API token
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", None)

    if upload and hf_token is None:
        raise ValueError("HF_TOKEN environment variable not set.")

    num_atoms = len(structure_template)
    num_species = len(elements)

    # Generate all possible atomic configurations
    configurations = np.array(body_order(n=num_atoms, b=num_species))

    # Prepare the database
    db_path = local_path or Path(__file__).resolve().parent / f"sqs_{'-'.join(elements)}.db"
    db_path.unlink(missing_ok=True)

    # Generate and save structures
    with connect(db_path) as db:
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
            db.write(atoms)

    # Upload the database to Hugging Face Hub
    if upload:
        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=db_path,
            path_in_repo=f"{Path(__file__).parent.name}/{db_path.name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"Database uploaded: {db_path}")

    return db_path


if __name__ == "__main__":
    structure_template = bulk("Al", a=3.6, cubic=True).repeat([2, 2, 2])
    elements = ["Fe", "Ni", "Cr"]
    generate_alloy_db(structure_template, elements, upload=True)
