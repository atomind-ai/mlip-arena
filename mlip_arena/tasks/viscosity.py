from __future__ import annotations

from collections import deque

import numpy as np
from ase import Atoms
from ase.calculators.calculator import BaseCalculator
from prefect import task

from mlip_arena.tasks.md import run as MD


class StressTensorCorrelator:
    """
    Class to calculate stress tensor autocorrelation functions for viscosity calculation.
    Emulates LAMMPS fix ave/correlate functionality.
    """

    def __init__(self, nevery=1, nrepeat=100, correlation_length=None, prefactor=1.0):
        """
        Initialize stress tensor correlator.

        Parameters:
        nevery: sampling interval (timesteps)
        nrepeat: number of correlation time windows
        correlation_length: maximum correlation time (if None, uses nrepeat*nevery)
        prefactor: scaling factor for correlation data
        """
        self.nevery = nevery
        self.nrepeat = nrepeat
        self.correlation_length = correlation_length or nrepeat * nevery
        self.prefactor = prefactor

        # Storage for stress tensor components (pxy, pxz, pyz)
        self.stress_history = deque(maxlen=self.correlation_length // nevery + 1)
        self.correlations = np.zeros((nrepeat, 3))  # For pxy, pxz, pyz
        self.correlation_counts = np.zeros(nrepeat)
        self.time_deltas = np.arange(nrepeat) * nevery

        self.step_count = 0

    def add_stress_data(self, stress_tensor):
        """Add stress tensor data point and update correlations incrementally."""
        if self.step_count % self.nevery == 0:
            # Extract off-diagonal stress components (pxy, pxz, pyz)
            pxy = stress_tensor[0, 1]  # or stress_tensor[1, 0]
            pxz = stress_tensor[0, 2]  # or stress_tensor[2, 0]
            pyz = stress_tensor[1, 2]  # or stress_tensor[2, 1]

            current_stress = np.array([pxy, pxz, pyz])
            self.stress_history.append(current_stress)
            
            # Update correlations incrementally
            self._update_correlations_incremental(current_stress)

        self.step_count += 1

    def _update_correlations_incremental(self, current_stress):
        """Update correlation functions incrementally with new data point."""
        history_length = len(self.stress_history)
        
        if history_length < 2:
            return
            
        # Update correlations for all possible time lags with the new sample
        max_lag = min(self.nrepeat, history_length)
        
        for dt_idx in range(max_lag):
            if dt_idx < history_length:
                # Get the sample at time (t - dt)
                past_sample = self.stress_history[-(dt_idx + 1)]
                
                # Calculate autocorrelation: <S(t-dt) * S(t)>
                corr = past_sample * current_stress
                
                # Update running average
                old_count = self.correlation_counts[dt_idx]
                new_count = old_count + 1
                
                if old_count == 0:
                    self.correlations[dt_idx] = corr * self.prefactor
                else:
                    # Running average update: new_avg = (old_avg * old_count + new_value) / new_count
                    self.correlations[dt_idx] = (
                        self.correlations[dt_idx] * old_count + corr * self.prefactor
                    ) / new_count
                
                self.correlation_counts[dt_idx] = new_count

    def calculate_correlations(self):
        """Calculate autocorrelation functions (fallback method for validation)."""
        if len(self.stress_history) < 2:
            return

        history = np.array(self.stress_history)
        n_samples = len(history)

        # Calculate autocorrelations for each time lag
        for dt_idx in range(min(self.nrepeat, n_samples)):
            correlations = []
            count = 0

            # Calculate correlation for this time lag
            for i in range(n_samples - dt_idx):
                # Autocorrelation: <S(t) * S(t + dt)>
                corr = history[i] * history[i + dt_idx]
                correlations.append(corr)
                count += 1

            if correlations:
                self.correlations[dt_idx] = (
                    np.mean(correlations, axis=0) * self.prefactor
                )
                self.correlation_counts[dt_idx] = count

    def integrate_correlations(self, dt):
        """Integrate correlation functions using trapezoidal rule."""
        integrations = []
        for i in range(3):  # pxy, pxz, pyz
            # Only integrate over time lags where we have sufficient statistics
            valid_indices = self.correlation_counts > 0
            if np.any(valid_indices):
                time_vals = self.time_deltas[valid_indices] * dt
                corr_vals = self.correlations[valid_indices, i]
                integral = np.trapezoid(corr_vals, x=time_vals)
                integrations.append(integral)
            else:
                integrations.append(0.0)
        return integrations


@task
def run(
    atoms: Atoms,
    calculator: BaseCalculator,
    temperature: float,
    pressure: float,
    time_step: float | None = None,  # fs
    npt_eq_time: float = 10_000,  # fs
    nve_eq_time: float = 1_000,  # fs
    prod_time: float = 100_000,  # fs
    velocity_seed: int | None = None,
):
    atoms = atoms.copy()
    atoms.calc = calculator

    # First stage: NPT equilibration from initial state to target temperature and pressure

    T0 = atoms.get_temperature()
    P0 = -atoms.get_stress().trace() / 3.0

    result = MD(
        atoms=atoms,
        calculator=calculator,
        ensemble="npt",
        temperature=[T0, temperature],
        pressure=[P0, pressure],
        time_step=time_step,
        total_time=npt_eq_time,
        velocity_seed=velocity_seed,
    )

    # Second stage: NVE equilibration

    result = MD(
        atoms=result["atoms"],
        calculator=calculator,
        ensemble="nve",
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        total_time=nve_eq_time,
        velocity_seed=velocity_seed,
    )

    # Third stage: NVE production run for viscosity calculation

    sacf = StressTensorCorrelator(
        nevery=1,  # Store stress every step
        nrepeat=500,  # Number of correlation time windows
        correlation_length=500,  # Maximum correlation time in steps
    )

    def store_stress():
        stress = result["atoms"].get_stress(voigt=False)
        sacf.add_stress_data(stress)

    result = MD(
        atoms=result["atoms"],
        calculator=calculator,
        ensemble="nve",
        temperature=temperature,
        pressure=pressure,
        time_step=time_step,
        total_time=prod_time,
        velocity_seed=velocity_seed,
        callbacks=[(store_stress, 1)],  # Store stress every step
    )

    # Calculate correlations after production run
    # sacf.calculate_correlations()
    
    integrations = sacf.integrate_correlations(result["time_step"])

    kB_SI = 1.380649e-23  # J/K

    # Calculate viscosity for each component
    viscosity_components = []
    for integral in integrations:
        eta = (result["atoms"].get_volume() * (1e-30 / kB_SI) / temperature) * integral
        viscosity_components.append(eta)

    etas = np.mean(viscosity_components, axis=0)

    return {
        "viscosity": {
            "units": "mPa·s",
            "etas": etas * 1e3,  # convert to mPa·s
            "final": etas[-1] * 1e3,  # average viscosity in mPa·s
        }
    }
