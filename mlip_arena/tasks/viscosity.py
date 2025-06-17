from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from ase.md.md import MolecularDynamics
from prefect import task
from prefect.cache_policies import INPUTS, TASK_SOURCE
from prefect.runtime import task_run

from mlip_arena.tasks.md import run as MD
from mlip_arena.tasks.utils import logger


class StressTensorCorrelator:
    """
    Class to calculate stress tensor autocorrelation functions for viscosity calculation.
    Implements LAMMPS fix ave/correlate functionality with 'ave running' behavior.

    Based on LAMMPS implementation:
    - Correlation C_ij(dt) = <V_i(t) * V_j(t+dt)> where V_i, V_j are stress components
    - For viscosity: correlate pxy, pxz, pyz with themselves (type=auto)
    - Uses 'ave running': correlations accumulate continuously without reset
    - Integration using trapezoidal rule with ASE unit conversion (eV/Å³ → Pa·s)
    """

    def __init__(self, nevery=1, nrepeat=100, nfreq=None, prefactor=1.0):
        """
        Initialize stress tensor correlator following LAMMPS ave/correlate syntax.

        Parameters:
        nevery: sample input values every this many timesteps
        nrepeat: number of correlation time windows to accumulate
        nfreq: calculate correlation averages every this many timesteps (if None, uses nrepeat*nevery)
        prefactor: scaling factor for correlation data

        Note: Uses 'ave running' behavior - correlations accumulate continuously
        """
        self.nevery = nevery
        self.nrepeat = nrepeat
        self.nfreq = nfreq or nrepeat * nevery
        self.prefactor = prefactor

        # Ensure nfreq >= (nrepeat - 1) * nevery as required by LAMMPS
        min_nfreq = (nrepeat - 1) * nevery
        if self.nfreq < min_nfreq:
            self.nfreq = min_nfreq

        # Storage for stress tensor components (pxy, pxz, pyz)
        # Use deque with maxlen to automatically handle circular buffer
        max_samples = self.nfreq // nevery + 1
        self.stress_history = deque(maxlen=max_samples)

        # Correlation arrays: [nrepeat x 3] for pxy, pxz, pyz
        self.correlations = np.zeros((nrepeat, 3))
        self.correlation_counts = np.zeros(nrepeat)

        # Time deltas in timesteps
        self.time_deltas = np.arange(nrepeat) * nevery

        self.step_count = 0
        self.last_output_step = -1

    def add_stress_data(self, stress_tensor):
        """Add stress tensor data point following LAMMPS sampling protocol."""

        # Only sample every nevery steps
        if self.step_count % self.nevery == 0:
            # Extract off-diagonal stress components (shear stress)
            # ASE stress tensor: eV/Å³, negative sign for pressure convention
            pxy = -stress_tensor[0, 1]
            pxz = -stress_tensor[0, 2]
            pyz = -stress_tensor[1, 2]

            current_stress = np.array([pxy, pxz, pyz])
            self.stress_history.append(current_stress)

            # Update correlations incrementally (ave running behavior)
            self._update_correlations_running()

        self.step_count += 1

    def _update_correlations_running(self):
        """
        Update correlations using 'ave running' method - continuously accumulate.
        This matches LAMMPS 'ave running' behavior where correlations are never reset.

        Uses incremental averaging: new_avg = (old_avg * old_count + new_data) / new_count
        """
        if len(self.stress_history) < 2:
            return

        # Get the most recent stress sample
        current_stress = self.stress_history[-1]
        history_length = len(self.stress_history)

        # Update correlations for all possible time lags with the new sample
        max_lag = min(self.nrepeat, history_length)

        for dt_idx in range(max_lag):
            if dt_idx < history_length:
                # Get the sample at time (t - dt*nevery)
                past_index = -(dt_idx + 1)
                if abs(past_index) <= len(self.stress_history):
                    past_stress = self.stress_history[past_index]

                    # Calculate autocorrelation: past_stress(t-dt) * current_stress(t)
                    corr = (
                        past_stress * current_stress
                    )  # element-wise for pxy, pxz, pyz

                    # Update running average
                    old_count = self.correlation_counts[dt_idx]
                    new_count = old_count + 1

                    if old_count == 0:
                        self.correlations[dt_idx] = corr * self.prefactor
                    else:
                        # Running average: new_avg = (old_avg * old_count + new_value) / new_count
                        self.correlations[dt_idx] = (
                            self.correlations[dt_idx] * old_count
                            + corr * self.prefactor
                        ) / new_count

                    self.correlation_counts[dt_idx] = new_count

    def get_correlation_data(self):
        """
        Return correlation data in LAMMPS format.

        Returns:
        dict with keys:
        - time_deltas: array of time delays (in timesteps)
        - correlations: array of shape [nrepeat, 3] for pxy, pxz, pyz correlations
        - counts: number of samples contributing to each correlation
        """
        return {
            "time_deltas": self.time_deltas,
            "correlations": self.correlations.copy(),
            "counts": self.correlation_counts.copy(),
        }

    def integrate_correlations(self, dt_fs):
        """
        Integrate correlation functions using trapezoidal rule.

        Parameters:
        dt_fs: timestep in femtoseconds

        Returns:
        List of integrals for [pxy, pxz, pyz] correlations
        """
        integrals = []

        for i in range(3):  # pxy, pxz, pyz
            # Only integrate over time lags where we have data
            valid_mask = self.correlation_counts > 0

            if np.any(valid_mask):
                # Convert time delays from timesteps to femtoseconds
                time_vals = self.time_deltas[valid_mask] * dt_fs
                corr_vals = self.correlations[valid_mask, i]

                # Trapezoidal integration
                if len(time_vals) > 1:
                    integral = np.trapezoid(corr_vals, x=time_vals)
                else:
                    integral = 0.0

                integrals.append(integral)
            else:
                integrals.append(0.0)

        return integrals


def _generate_task_run_name():
    task_name = task_run.task_name
    parameters = task_run.parameters

    atoms = parameters["atoms"]
    calculator = parameters["calculator"]

    return f"{task_name}: {atoms.get_chemical_formula()} - {calculator}"

@task(
    name="VISCOSITY", task_run_name=_generate_task_run_name, cache_policy=TASK_SOURCE + INPUTS
)
def run(
    atoms: Atoms,
    calculator: BaseCalculator,
    temperature: float,
    pressure: float,
    time_step: float | None = None,  # fs
    npt_eq_time: float = 10_000,  # fs
    dynamics: str | MolecularDynamics = "nose-hoover",
    nve_eq_time: float = 1_000,  # fs
    prod_time: float = 100_000,  # fs
    velocity_seed: int | None = None,
    method: Literal["green-kubo"] = "green-kubo",
    correlation_length: int = 500,  # correlation time windows
    sample_interval: int = 1,  # sampling interval in timesteps
):
    """
    Calculate shear viscosity using Green-Kubo method following LAMMPS methodology.

    The viscosity is calculated from:
    η = (V/(k_B*T)) * ∫[0→∞] <P_αβ(0) * P_αβ(t)> dt

    Where:
    - V is the volume
    - k_B is Boltzmann constant
    - T is temperature
    - P_αβ are the off-diagonal stress tensor components (pxy, pxz, pyz)
    - <...> denotes ensemble average
    """
    atoms = atoms.copy()
    atoms.calc = calculator

    assert method == "green-kubo", (
        "Only Green-Kubo method is currently implemented for viscosity calculation."
    )

    # First stage: NPT equilibration from initial state to target temperature and pressure
    T0 = atoms.get_temperature()
    try:
        P0 = -atoms.get_stress().trace() / 3.0
    except Exception as e:
        logger.error(
            "Failed to get initial pressure from stress tensor. "
            "Ensure the calculator supports stress calculation. "
            "Error raised from: " + str(e)
        )
        logger.warning(
            "Assuming initial pressure is zero. "
            "This may lead to instability if the system is not close to equilibrium."
        )
        P0 = 0.0  # Assume initial pressure is zero if stress tensor is not available

    if npt_eq_time > 0:
        result = MD(
            atoms=atoms,
            calculator=calculator,
            ensemble="npt",
            dynamics=dynamics,
            temperature=[T0, temperature],
            pressure=[P0, pressure],
            time_step=time_step,
            total_time=npt_eq_time,
            velocity_seed=velocity_seed,
        )
        atoms = result["atoms"]

    # Second stage: NVE equilibration

    if nve_eq_time > 0:
        result = MD(
            atoms=atoms,
            calculator=calculator,
            ensemble="nve",
            dynamics="velocityverlet",
            temperature=temperature,
            pressure=pressure,
            time_step=time_step,
            total_time=nve_eq_time,
            velocity_seed=velocity_seed,
        )
        atoms = result["atoms"]

    # Third stage: NVE production run for viscosity calculation
    # Initialize correlator with LAMMPS-compatible parameters

    # Calculate nfreq to ensure we get final correlation calculation
    total_steps = int(prod_time / result["time_step"])
    nfreq = min(total_steps, max(correlation_length * sample_interval, 1000))

    correlator = StressTensorCorrelator(
        nevery=sample_interval,
        nrepeat=correlation_length,
        nfreq=nfreq,
        prefactor=1.0,  # Will handle unit conversion later
    )

    def store_stress(dyn: MolecularDynamics):
        """Callback to store stress tensor data during MD simulation."""
        stress = dyn.atoms.get_stress(voigt=False)  # Get 3x3 stress tensor
        correlator.add_stress_data(stress)

    # Run production MD with stress data collection
    result = MD(
        atoms=result["atoms"],
        calculator=calculator,
        ensemble="nve",
        dynamics="velocityverlet",
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        total_time=prod_time,
        velocity_seed=velocity_seed,
        callbacks=[(store_stress, 1)],  # Store stress every step
    )

    # Get final correlation data
    corr_data = correlator.get_correlation_data()

    # Integrate correlation functions
    dt_fs = result["time_step"]  # timestep in fs
    integrals = correlator.integrate_correlations(dt_fs)

    # Unit conversion for ASE units to SI
    # ASE units: stress in eV/Å³, time in fs, volume in Å³

    # Physical constants
    kB_SI = 1.380649e-23  # J/K - Boltzmann constant
    eV_to_J = 1.602176634e-19  # J/eV
    fs_to_s = 1.0e-15  # s/fs
    A3_to_m3 = 1.0e-30  # m³/Å³

    # Unit conversion factor for Green-Kubo viscosity
    # stress²·time·volume / (kB·T) → Pa·s
    # (eV/Å³)² · fs · Å³ / (J/K · K) → Pa·s
    # = eV²·fs / (Å³·J/K·K) → Pa·s
    # = (eV²·fs·K) / (Å³·J) → Pa·s

    # Get system volume and temperature
    volume_A3 = result["atoms"].get_volume()  # Volume in Å³
    temp_K = temperature  # Temperature in K

    # Calculate viscosity components using Green-Kubo formula
    # η = (V / (k_B * T)) * ∫ <P_αβ(0) * P_αβ(t)> dt

    viscosity_components = []
    for integral_eV2_fs_per_A3 in integrals:
        # Convert integral from (eV/Å³)²·fs to J²·s/m⁶
        integral_SI = integral_eV2_fs_per_A3 * (eV_to_J**2) * fs_to_s / (A3_to_m3**2)

        # Apply Green-Kubo formula: η = (V / (k_B * T)) * integral
        viscosity_SI = (volume_A3 * A3_to_m3) / (kB_SI * temp_K) * integral_SI
        viscosity_components.append(viscosity_SI)

    # Average the three components (pxy, pxz, pyz)
    eta_Pa_s = np.mean(viscosity_components, axis=0)

    # Convert to mPa·s (millipascal-seconds)
    eta_mPa_s = eta_Pa_s * 1e3

    logger.info(
        f"Calculated viscosity: {eta_mPa_s.mean():.3f} mPa·s "
    )

    return {
        "viscosity": {
            "units": "mPa·s",
            "components": {
                "pxy": viscosity_components[0] * 1e3,
                "pxz": viscosity_components[1] * 1e3,
                "pyz": viscosity_components[2] * 1e3,
            },
            "average": eta_mPa_s,
            "final": eta_mPa_s.mean(),
        },
        "correlation_data": corr_data,
        "integrals": integrals,
        "system_info": {
            "volume_A3": volume_A3,
            "temperature_K": temp_K,
            "timestep_fs": dt_fs,
            "total_steps": correlator.step_count,
            "correlation_length": correlation_length,
        },
    }
