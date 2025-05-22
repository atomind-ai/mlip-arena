"""
Define equivariance testing task.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from ase import Atoms
from prefect import task
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


def generate_random_unit_vector():
    """Generate a random unit vector."""
    vec = np.random.normal(0, 1, 3)
    return vec / np.linalg.norm(vec)


def rotate_molecule_arbitrary(
    atoms: Atoms, angle: float, axis: np.ndarray
) -> tuple[Atoms, np.ndarray]:
    """Rotate molecule around arbitrary axis."""
    rotated_atoms = atoms.copy()
    positions = rotated_atoms.get_positions()
    rot = R.from_rotvec(np.radians(angle) * axis)
    rotation_mat = rot.as_matrix()
    rotated_positions = rot.apply(positions)
    rotated_atoms.set_positions(rotated_positions)
    cell = atoms.get_cell()
    rotated_cell = rot.apply(cell)
    rotated_atoms.set_cell(rotated_cell)
    return rotated_atoms, rotation_mat


def compare_forces(
    original_forces: np.ndarray,
    rotated_forces: np.ndarray,
    rotation_mat: np.ndarray,
    zero_threshold: float = 1e-10,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compare forces before and after rotation, with handling of 0 force case.

    Args:
        original_forces: Forces before rotation (N x 3 array)
        rotated_forces: Forces after rotation (N x 3 array)
        rotation_mat: 3 x 3 rotation matrix
        zero_threshold: Threshold below which forces are considered zero

    Returns:
        tuple containing:
            - mae: Mean absolute error between forces
            - cosine_similarity: Cosine similarity between force vectors
    """
    rotated_original_forces = np.dot(original_forces, rotation_mat.T)
    force_diff = rotated_original_forces - rotated_forces
    mae = np.mean(np.abs(force_diff))

    original_magnitudes = np.linalg.norm(rotated_original_forces, axis=1)
    rotated_magnitudes = np.linalg.norm(rotated_forces, axis=1)

    zero_original = original_magnitudes < zero_threshold
    zero_rotated = rotated_magnitudes < zero_threshold
    both_zero = zero_original & zero_rotated
    either_zero = zero_original | zero_rotated
    one_zero = either_zero & ~both_zero

    cosine_similarity = np.zeros(len(original_forces))

    valid_forces = ~either_zero
    if np.any(valid_forces):
        norms_product = np.linalg.norm(
            rotated_original_forces[valid_forces], axis=1
        ) * np.linalg.norm(rotated_forces[valid_forces], axis=1)
        dot_products = np.sum(
            rotated_original_forces[valid_forces] * rotated_forces[valid_forces], axis=1
        )
        cosine_similarity[valid_forces] = dot_products / norms_product

    # If both forces are 0, cosine similarity should be 1. If one is 0, we take the conservative -1.
    cosine_similarity[both_zero] = 1.0
    cosine_similarity[one_zero] = -1.0

    return mae, cosine_similarity


def save_molecule_results(
    aggregate_results: dict, idx_list: np.ndarray, save_path: str | Path
) -> None:
    """
    Save all molecule results from equivariance testing to .npy files.
    Save the index list of the atoms for further analysis.

    Args:
        aggregate_results: Dictionary containing the aggregated results from run()
        idx_list: List of the indices of the atoms in the original dataset
        save_path: Path to save the .npy files
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    all_molecule_results = aggregate_results["molecule_results"]
    rotation_angles = list(all_molecule_results[0]["results_by_angle"].keys())

    num_molecules = len(all_molecule_results)
    num_angles = len(rotation_angles)
    num_random_axes = len(
        all_molecule_results[0]["results_by_angle"][rotation_angles[0]]["maes"]
    )
    num_atoms = len(
        all_molecule_results[0]["results_by_angle"][rotation_angles[0]][
            "cosine_similarities"
        ][0]
    )

    maes = np.zeros((num_molecules, num_angles, num_random_axes))
    cosine_similarities = np.zeros((num_molecules, num_angles, num_random_axes))

    for mol_idx, molecule in enumerate(all_molecule_results):
        for angle_idx, angle in enumerate(rotation_angles):
            angle_results = molecule["results_by_angle"][angle]
            maes[mol_idx, angle_idx, :] = angle_results["maes"]
            cosine_similarities[mol_idx, angle_idx, :] = np.mean(
                angle_results["cosine_similarities"], axis=-1
            )

    np.save(save_path.with_name(f"{save_path.stem}_maes.npy"), maes)
    np.save(
        save_path.with_name(f"{save_path.stem}_cosine_similarities.npy"),
        cosine_similarities,
    )
    np.save(save_path.with_name(f"{save_path.stem}_idx_list.npy"), idx_list)


@task(
    name="Equivariance testing",
    task_run_name=_generate_task_run_name,
    cache_policy=TASK_SOURCE + INPUTS,
)
def run(
    atoms_list: Sequence[Atoms],
    idx_list: np.ndarray,
    calculator: BaseCalculator,
    save_path: str | Path | None = None,
    rotation_angles: list[float] | np.ndarray = None,
    num_random_axes: int = 100,
    threshold: float = 1e-3,
    seed: int | None = None,
) -> dict:
    """
    Test equivariance of force predictions under rotations for multiple structures.

    Args:
        atoms_list: List of input atomic structures
        idx_list: List of the indices of the atoms in the original dataset
        calculator: Calculator to use
        num_rotations: Number of random rotations to test
        rotation_angle: Angle of rotation in degrees
        threshold: Threshold for considering forces equivariant
        seed: Random seed

    Returns:
        Dictionary containing test results
    """
    if seed is not None:
        np.random.seed(seed)

    if rotation_angles is None:
        rotation_angles = np.arange(30, 361, 30)
    rotation_angles = np.array(rotation_angles)

    all_results = []

    cross_molecule_cosine_sims = {angle: [] for angle in rotation_angles}
    cross_molecule_mae = {angle: [] for angle in rotation_angles}

    rotation_axes = [generate_random_unit_vector() for _ in range(num_random_axes)]

    total_tests = len(atoms_list) * len(rotation_angles) * num_random_axes
    pbar = tqdm(total=total_tests, desc="Testing rotations")

    for atom_idx, atoms in enumerate(atoms_list):
        atoms = atoms.copy()
        atoms.calc = calculator
        original_forces = atoms.get_forces()

        results_by_angle = {
            angle: {
                "mae": [],
                "cosine_similarities": [],
                "passed_tests": 0,
                "passed_mae": 0,
                "passed_cosine_similarity": 0,
            }
            for angle in rotation_angles
        }
        # Test each angle with multiple random axes
        for angle in rotation_angles:
            for axis in rotation_axes:
                rotated_atoms, rotation_mat = rotate_molecule_arbitrary(
                    atoms, angle, axis
                )
                rotated_atoms.calc = calculator
                rotated_forces = rotated_atoms.get_forces()
                mae, cosine_similarity = compare_forces(
                    original_forces, rotated_forces, rotation_mat
                )
                results_by_angle[angle]["mae"].append(mae)
                results_by_angle[angle]["cosine_similarities"].append(cosine_similarity)

                cross_molecule_cosine_sims[angle].append(
                    float(np.mean(cosine_similarity))
                )
                cross_molecule_mae[angle].append(float(np.mean(mae)))

                mae_check = mae < threshold
                cosine_check = all(cosine_similarity > (1 - threshold))
                results_by_angle[angle]["passed_tests"] += int(
                    mae_check and cosine_check
                )
                results_by_angle[angle]["passed_mae"] += int(mae_check)
                results_by_angle[angle]["passed_cosine_similarity"] += int(cosine_check)

                pbar.update(1)
        # Compute summary statistics
        for angle in rotation_angles:
            results = results_by_angle[angle]
            results["mean_cosine_similarity"] = float(
                np.mean(results["cosine_similarities"])
            )
            results["avg_mae"] = float(np.mean(results["mae"]))
            results["equivariant_ratio"] = results["passed_tests"] / num_random_axes
            results["mae_passed_ratio"] = results["passed_mae"] / num_random_axes
            results["cosine_passed_ratio"] = (
                results["passed_cosine_similarity"] / num_random_axes
            )
            results["passed"] = results["passed_tests"] == num_random_axes
            results["passed_mae"] = results["passed_mae"] == num_random_axes
            results["passed_cosine_similarity"] = (
                results["passed_cosine_similarity"] == num_random_axes
            )
            results["maes"] = [float(x) for x in results["mae"]]
            results["cosine_similarities"] = [
                [float(y) for y in x] for x in results["cosine_similarities"]
            ]

        molecule_results = {
            "mol_idx": idx_list[atom_idx],
            "results_by_angle": results_by_angle,
            "all_passed": all(
                results_by_angle[angle]["passed"] for angle in rotation_angles
            ),
            "avg_cosine_similarity_by_molecule": float(
                np.mean(
                    [
                        results_by_angle[angle]["mean_cosine_similarity"]
                        for angle in rotation_angles
                    ]
                )
            ),
            "avg_mae_by_molecule": float(
                np.mean(
                    [results_by_angle[angle]["avg_mae"] for angle in rotation_angles]
                )
            ),
            "overall_equivariant_ratio": float(
                np.mean(
                    [
                        results_by_angle[angle]["equivariant_ratio"]
                        for angle in rotation_angles
                    ]
                )
            ),
        }

        all_results.append(molecule_results)

    pbar.close()

    aggregate_results = {
        "num_molecules": len(atoms_list),
        "all_molecules_passed": all(result["all_passed"] for result in all_results),
        "average_equivariant_ratio": float(
            np.mean([result["overall_equivariant_ratio"] for result in all_results])
        ),
        "average_cosine_similarity_by_angle": {
            angle: float(np.mean(sims))
            for angle, sims in cross_molecule_cosine_sims.items()
        },
        "average_mae_by_angle": {
            angle: float(np.mean(diffs)) for angle, diffs in cross_molecule_mae.items()
        },
        "molecule_results": all_results,
    }

    if save_path:
        save_molecule_results(aggregate_results, idx_list, save_path)
        np.save(
            str(save_path.with_name(f"{save_path.stem}_molecule_results.npy")),
            all_results,
        )

    return aggregate_results
