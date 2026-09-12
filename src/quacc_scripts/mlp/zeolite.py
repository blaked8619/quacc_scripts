import torch
# Fix PyTorch 2.6 compatibility with e3nn
torch.serialization.add_safe_globals([slice])

import matplotlib.pyplot as plt
from jobflow import job
from jobflow_remote import submit_flow, set_run_config
from ase.io import write
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from matcalc import PhononCalc, RelaxCalc
import numpy as np
from contextlib import redirect_stdout
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE2

from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase import units
from emmet.core.symmetry import PointGroupData
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc._qha import QHACalc

from ase.calculators.mixing import SumCalculator
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.bond_valence import BVAnalyzer

from dftd4.ase import DFTD4

import json
from monty.json import MontyEncoder

import logging
logging.basicConfig(level=logging.INFO)

import time
from pymatgen.core import Structure
import os

def mini_choose_calc(method):
    if method =="meta":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        model_name = "uma-s-1p2p1"
        predictor = pretrained_mlip.get_predict_unit(model_name)
        calc = FAIRChemCalculator(predictor, task_name="omol")
    elif method == "mace-medium":
        from mace.calculators import mace_polar
        calc = mace_polar(
        model="polar-1-m",
        device="cpu",
        default_dtype="float64"
        )
    elif method == "mace-large":
        from mace.calculators import mace_polar
        calc = mace_polar(
        model="polar-1-l",
        device="cpu",
        default_dtype="float64"
        )
    
    return calc

@job
def relax_gas(atoms, fmax, spin_multiplicity, method):
    
    atoms.info['spin'] = spin_multiplicity

    #atoms.set_cell([20, 20, 20])
    #atoms.pbc = False
    #atoms.center()
    
    calc = mini_choose_calc(method)
    atoms.calc = calc
    
    dyn = BFGS(atoms, trajectory='relaxation.traj', logfile='relax.log')
    dyn.run(fmax=fmax)
    
    write('structure.xyz', atoms)
    
    mlip_energy = atoms.get_potential_energy()
    
    with open("output_energy.txt", "w") as file:
        file.write(str(mlip_energy))
    
    return {"output_atoms": atoms, "mlip_energy": mlip_energy, "spin_multiplicity": atoms.info['spin']}

@job
def gas_vibrations(atoms, mlip_energy, spin_multiplicity, method):
    
 
    calc = mini_choose_calc(method)
    atoms.calc = calc
    atoms.info['spin'] = spin_multiplicity

    #atoms.pbc = False
    
    vib = Vibrations(atoms)
    vib.run()
    vib_energies = vib.get_energies()

    real_energies = []
    imag_energies = []

    for energy in vib_energies:
        if np.iscomplex(energy) or energy < 0:
            # Imaginary frequency (unstable mode)
            imag_energies.append(float(np.abs(energy)))
        else:
            real_energies.append(float(energy))

    # Or save both in one file with labels
    with open("vib_energies_summary.txt", "w") as f:
        f.write("# Vibrational Energies Summary\n")
        f.write(f"# Total modes: {len(vib_energies)}\n")
        f.write(f"# Real modes: {len(real_energies)}\n")
        f.write(f"# Imaginary modes: {len(imag_energies)}\n\n")
        f.write("Real energies (eV):\n")
        for e in real_energies:
            f.write(f"{e:.6f}\n")
        f.write("\nImaginary energies (eV, absolute values):\n")  # now outside loop
        for e in imag_energies:
            f.write(f"{e:.6f}\n")
    
    #find the symmetry number
    mol = AseAtomsAdaptor().get_molecule(atoms, charge_spin_check=False)
    point_group_data = PointGroupData().from_molecule(mol)
    sigma = point_group_data.rotation_number or 1
    sym = sigma

    #find the geometry
    natoms = len(atoms)
    if natoms == 1:
        geometry = "monatomic"
    elif point_group_data.linear:
        geometry = "linear"
    else:
        geometry = "nonlinear"

    #find the spin quantum number
    spin = (atoms.info['spin'] - 1)/2 # need to change spin_multiplcity from atoms.info to a total spin
    
    igt = IdealGasThermo(vib_energies, geometry, potentialenergy=mlip_energy, atoms=atoms, symmetrynumber=sym, spin=spin)

    #find the corrections
    temperatures = np.arange(0, 1001, 1)  # 0 to 1000 K, step of 1 K
    G_free_energy = []
    enthalpy = []
    entropy = []

    pressure = 100000  # Pa (1bar), 0 atm would diverge to infinity for Gibbs

    kB = units.kB #boltzmann constant

    for T in temperatures:
        G = igt.get_gibbs_energy(T, pressure)
        H = igt.get_enthalpy(T)
        S = igt.get_entropy(T, pressure)

        G_free_energy.append(G)
        enthalpy.append(H)
        entropy.append(S)
    
    temperatures = np.array(temperatures)
    G_free_energy = np.array(G_free_energy)
    enthalpy = np.array(enthalpy)
    entropy = np.array(entropy)

    data = np.column_stack((temperatures, G_free_energy, enthalpy, entropy))

    header_full = "T (K)          Gibbs         Enthalpy         Entropy"
    np.savetxt("thermal_properties_full.txt", data,
           fmt="%12.3f %15.7f %15.7f %15.7f",
           header=header_full)

    idx_298 = np.where(temperatures == 298)[0][0]
    G_298K = float(G_free_energy[idx_298])

    
    return {"output_atoms": atoms, "G_298K": G_298K, "real_vibration_energies": real_energies, "imag_vibration_energies": imag_energies ,"spin_multiplicity": spin_multiplicity, "spin_quantum_number": spin, "geometry": geometry, "thermal_properties": data}

