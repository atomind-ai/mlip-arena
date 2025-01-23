import numpy as np
import torch

# TODO: consider using vesin
from matscipy.neighbours import neighbour_list
from torch_geometric.data import Data

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def get_neighbor(
    atoms: Atoms, cutoff: float, self_interaction: bool = False
):
    pbc = atoms.pbc
    cell = atoms.cell.array
    
    i, j, S = neighbour_list(
        quantities="ijS",
        pbc=pbc,
        cell=cell,
        positions=atoms.positions,
        cutoff=cutoff
    )

    if not self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = i == j
        true_self_edge &= np.all(S == 0, axis=1)
        keep_edge = ~true_self_edge

        i = i[keep_edge]
        j = j[keep_edge]
        S = S[keep_edge]

    edge_index = np.stack((i, j)).astype(np.int64)
    edge_shift = np.dot(S, cell)

    return edge_index, edge_shift



def collate_fn(batch: list[Atoms], cutoff: float) -> Data:
    """Collate a list of Atoms objects into a single batched Atoms object."""

    # Offset the edge indices for each graph to ensure they remain disconnected
    offset = 0

    node_batch = []

    numbers_batch = []
    positions_batch = []
    # ec_batch = []

    forces_batch = []
    charges_batch = []
    magmoms_batch = []
    dipoles_batch = []

    edge_index_batch = []
    edge_shift_batch = []

    cell_batch = []
    natoms_batch = []

    energy_batch = []
    stress_batch = []

    for i, atoms in enumerate(batch):
    
        edge_index, edge_shift = get_neighbor(atoms, cutoff=cutoff, self_interaction=False)

        edge_index[0] += offset
        edge_index[1] += offset
        edge_index_batch.append(torch.tensor(edge_index))
        edge_shift_batch.append(torch.tensor(edge_shift))

        natoms = len(atoms)
        offset += natoms
        node_batch.append(torch.ones(natoms, dtype=torch.long) * i)
        natoms_batch.append(natoms)

        cell_batch.append(torch.tensor(atoms.cell.array))
        numbers_batch.append(torch.tensor(atoms.numbers))
        positions_batch.append(torch.tensor(atoms.positions))

        # ec_batch.append([Atom(int(a)).elecronic_encoding for a in atoms.numbers])

        charges_batch.append(
            atoms.get_initial_charges()
            if atoms.get_initial_charges().any()
            else torch.full((natoms,), torch.nan)
        )
        magmoms_batch.append(
            atoms.get_initial_magnetic_moments()
            if atoms.get_initial_magnetic_moments().any()
            else torch.full((natoms,), torch.nan)
        )

    # Create the new 'arrays' data for the batch

    cell_batch = torch.stack(cell_batch, dim=0)
    node_batch = torch.cat(node_batch, dim=0)
    positions_batch = torch.cat(positions_batch, dim=0)
    numbers_batch = torch.cat(numbers_batch, dim=0)
    natoms_batch = torch.tensor(natoms_batch, dtype=torch.long)

    charges_batch = torch.cat(charges_batch, dim=0) if charges_batch else None
    magmoms_batch = torch.cat(magmoms_batch, dim=0) if magmoms_batch else None

    # ec_batch = list(map(lambda a: Atom(int(a)).elecronic_encoding, numbers_batch))
    # ec_batch = torch.stack(ec_batch, dim=0)

    edge_index_batch = torch.cat(edge_index_batch, dim=1)
    edge_shift_batch = torch.cat(edge_shift_batch, dim=0)

    arrays_batch_concatenated = {
        "cell": cell_batch,
        "positions": positions_batch,
        "edge_index": edge_index_batch,
        "edge_shift": edge_shift_batch,
        "numbers": numbers_batch,
        "num_nodes": offset,
        "batch": node_batch,
        "charges": charges_batch,
        "magmoms": magmoms_batch,
        # "ec": ec_batch,
        "natoms": natoms_batch,
        "cutoff": torch.tensor(cutoff),
    }

    # TODO: custom fields

    # Create a new Data object with the concatenated arrays data
    batch_data = Data.from_dict(arrays_batch_concatenated)

    return batch_data


def decollate_fn(batch_data: Data) -> list[Atoms]:
    """Decollate a batched Data object into a list of individual Atoms objects."""

    # FIXME: this function is not working properly when the batch_data is on GPU.
    # TODO: create a new Cell class using torch tensor to handle device placement.
    # As a temporary fix, detach the batch_data from the GPU and move it to CPU.
    batch_data = batch_data.detach().cpu()

    # Initialize empty lists to store individual data entries
    individual_entries = []

    # Split the 'batch' attribute to identify data entries
    unique_batches = batch_data.batch.unique(sorted=True)

    for i in unique_batches:
        # Identify the indices corresponding to the current data entry
        entry_indices = (batch_data.batch == i).nonzero(as_tuple=True)[0]

        # Extract the attributes for the current data entry
        cell = batch_data.cell[i]
        numbers = batch_data.numbers[entry_indices]
        positions = batch_data.positions[entry_indices]
        # edge_index = batch_data.edge_index[:, entry_indices]
        # edge_shift = batch_data.edge_shift[entry_indices]
        # batch_data.ec[entry_indices] if batch_data.ec is not None else None

        # Optional fields
        energy = batch_data.energy[i] if "energy" in batch_data else None
        forces = batch_data.forces[entry_indices] if "forces" in batch_data else None
        stress = batch_data.stress[i] if "stress" in batch_data else None

        # charges = batch_data.charges[entry_indices] if "charges" in batch_data else None
        # magmoms = batch_data.magmoms[entry_indices] if "magmoms" in batch_data else None
        # dipoles = batch_data.dipoles[entry_indices] if "dipoles" in batch_data else None

        # TODO: cumstom fields

        # Create an 'Atoms' object for the current data entry
        atoms = Atoms(
            cell=cell,
            positions=positions,
            numbers=numbers,
            # forces=None if torch.any(torch.isnan(forces)) else forces,
            # charges=None if torch.any(torch.isnan(charges)) else charges,
            # magmoms=None if torch.any(torch.isnan(magmoms)) else magmoms,
            # dipoles=None if torch.any(torch.isnan(dipoles)) else dipoles,
            # energy=None if torch.isnan(energy) else energy,
            # stress=None if torch.any(torch.isnan(stress)) else stress,
        )

        atoms.calc = SinglePointCalculator(
            energy=energy,
            forces=forces,
            stress=stress,
            # charges=charges,
            # magmoms=magmoms,
        ) # type: ignore

        # Append the individual data entry to the list
        individual_entries.append(atoms)

    return individual_entries
