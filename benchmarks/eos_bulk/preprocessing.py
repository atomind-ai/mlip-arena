import json

from ase.db import connect
from pymatgen.core import Structure

with open("wbm_structures.json") as f:
    structs = json.load(f)

with connect("wbm_structures.db") as db:
    for id, s in structs.items():
        atoms = Structure.from_dict(s).to_ase_atoms(msonable=False)
        db.write(atoms, wbm_id=id)
