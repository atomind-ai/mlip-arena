"""
Grid search for accessible positions


This script is heavily adapted from the `DAC-SIM <https://github.com/hspark1212/DAC-SIM>`_ package. Please cite the original work if you use this script.

References
~~~~~~~~~~~
- Lim, Y., Park, H., Walsh, A., & Kim, J. (2024). Accelerating CO₂ Direct Air Capture Screening for Metal-Organic Frameworks with a Transferable Machine Learning Force Field.
"""

import MDAnalysis as mda
import numpy as np

from ase import Atoms


def get_accessible_positions(
    structure: Atoms,
    grid_spacing: float = 0.5,
    cutoff_distance: float = 10.0,
    min_interplanar_distance: float = 2.0,
) -> dict:
    # get the supercell structure
    cell_volume = structure.get_volume()
    cell_vectors = np.array(structure.cell)
    dist_a = cell_volume / np.linalg.norm(np.cross(cell_vectors[1], cell_vectors[2]))
    dist_b = cell_volume / np.linalg.norm(np.cross(cell_vectors[2], cell_vectors[0]))
    dist_c = cell_volume / np.linalg.norm(np.cross(cell_vectors[0], cell_vectors[1]))
    plane_distances = np.array([dist_a, dist_b, dist_c])
    supercell = np.ceil(min_interplanar_distance / plane_distances).astype(int)
    if np.any(supercell > 1):
        print(
            f"Making supercell: {supercell} to prevent interplanar distance < {min_interplanar_distance}"
        )
    structure = structure.repeat(supercell)
    # get position for grid
    grid_size = np.ceil(np.array(structure.cell.cellpar()[:3]) / grid_spacing).astype(
        int
    )
    indices = np.indices(grid_size).reshape(3, -1).T  # (G, 3)
    pos_grid = indices.dot(cell_vectors / grid_size)  # (G, 3)
    # get positions for atoms
    pos_atoms = structure.get_positions()  # (N, 3)
    # distance matrix
    dist_matrix = mda.lib.distances.distance_array(
        pos_grid, pos_atoms, box=structure.cell.cellpar()
    )  # (G, N) # TODO: check if we could use other packages instead of mda

    # calculate the accessible positions
    min_dist = np.min(dist_matrix, axis=1)  # (G,)
    idx_accessible_pos = np.where(min_dist > cutoff_distance)[0]

    # result
    return {
        "pos_grid": pos_grid,
        "idx_accessible_pos": idx_accessible_pos,
        "accessible_pos": pos_grid[idx_accessible_pos],
        "structure": structure,
    }
