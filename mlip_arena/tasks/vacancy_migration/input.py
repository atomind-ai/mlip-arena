from mp_api.client import MPRester

from ase import Atom
from ase.data import covalent_radii
from ase.spacegroup import crystal

fcc_elements = [
    "Ac",
    "Ag",
    "Al",
    "Ar",
    "Au",
    "Ba",
    "Be",
    "Ca",
    "Cd",
    "Ce",
    "Co",
    "Cs",
    "Cu",
    "Dy",
    "Er",
    "Fe",
    "Ga",
    "Ge",
    "He",
    "Hf",
    "Ho",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Mg",
    "Mn",
    "Na",
    "Ni",
    "Os",
    "Pa",
    "Pb",
    "Pd",
    "Pr",
    "Pt",
    "Rb",
    "Re",
    "Rh",
    "Ru",
    "Sc",
    "Sn",
    "Sr",
    "Ta",
    "Tb",
    "Tc",
    "Th",
    "Ti",
    "Tl",
    "W",
    "Xe",
    "Y",
    "Zr"
]

def get_fcc_pristine(mp_api_key = None):
    for element in fcc_elements:
        with MPRester(mp_api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=element, spacegroup_number=225, fields=["structure", "energy_above_hull"]
            )

            docs = sorted(docs, key=lambda x: x.energy_above_hull)

            if len(docs) != 0:
                pristine = docs[0].structure.to_conventional().to_ase_atoms(msonable=False) * (3, 3, 3)
            
                if len(pristine) != 108:
                    v = pristine.get_volume() / len(pristine)
                    r = v**(1/3)
                    a = 2*(2**0.5)*r

                    pristine = crystal(
                        symbols=[element]*4,
                        basis=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                        spacegroup=225,
                        cellpar=[a, a, a, 90, 90, 90],
                    ) * (3, 3, 3)
            else:
                r = covalent_radii[Atom(element).number] or 4
                a = 2*(2**0.5)*r
                pristine = crystal(
                    symbols=[element]*4,
                    basis=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                    spacegroup=225,
                    cellpar=[a, a, a, 90, 90, 90],
                ) * (3, 3, 3)
        yield pristine

hcp_elements = [
    "Ag",
    "Al",
    "Ar",
    "Au",
    "Ba",
    "Be",
    "Ca",
    "Cd",
    "Ce",
    "Co",
    "Cr",
    "Cs",
    "Cu",
    "Fe",
    "Ga",
    "Ge",
    "He",
    "Hf",
    "Ho",
    "In",
    "Ir",
    "K",
    "Kr",
    "La",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "Nb",
    "Ne",
    "Ni",
    "Os",
    "P",
    "Pb",
    "Pd",
    "Pt",
    "Rb",
    "Re",
    "Rh",
    "Ru",
    "Sc",
    "Si",
    "Sn",
    "Sr",
    "Ta",
    "Tc",
    "Te",
    "Th",
    "Ti",
    "Tl",
    "V",
    "W",
    "Xe",
    "Y",
    "Zn",
    "Zr"
]

def get_hcp_pristine(mp_api_key = None):
    for element in hcp_elements:
        with MPRester(mp_api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=element, spacegroup_number=194, fields=["structure", "energy_above_hull"]
            )

            docs = sorted(docs, key=lambda x: x.energy_above_hull)

            if len(docs) != 0:
                pristine = docs[0].structure.to_conventional().to_ase_atoms(msonable=False) * (3, 3, 1)
            
                if len(pristine) != 36:
                    v = pristine.get_volume() / len(pristine)
                    r = v**(1/3)
                    a = 2*r
                    c = 4 * ((2/3) ** 0.5) * r

                    pristine = crystal(
                        [element],
                        [(1.0 / 3.0, 2.0 / 3.0, 3.0 / 4.0)],
                        spacegroup=194,
                        cellpar=[a, a, c, 90, 90, 120],
                    ) * (3, 3, 2)
            else:
                r = covalent_radii[Atom(element).number] or 4
                a = 2*r
                c = 4 * ((2/3) ** 0.5) * r

                pristine = crystal(
                    [element],
                    [(1.0 / 3.0, 2.0 / 3.0, 3.0 / 4.0)],
                    spacegroup=194,
                    cellpar=[a, a, c, 90, 90, 120],
                ) * (3, 3, 2)
        yield pristine