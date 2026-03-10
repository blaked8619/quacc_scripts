import torch
# Fix PyTorch 2.6 compatibility with e3nn
torch.serialization.add_safe_globals([slice])
import pickle
import quacc
from jobflow import job
from jobflow_remote import submit_flow, set_run_config
from ase.io import write
from mace.calculators import MACECalculator
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from matcalc import PhononCalc, RelaxCalc
import numpy as np
from contextlib import redirect_stdout
from pymatgen.core import Structure
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import matplotlib.pyplot as plt

from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase import units
from emmet.core.symmetry import PointGroupData
from pymatgen.io.ase import AseAtomsAdaptor

from matcalc._qha import QHACalc

import json
from monty.json import MontyEncoder

@job
def relax_mof(atoms, model_path, fmax):
    write('POSCAR', atoms, format='vasp')
    calc = MACECalculator(
        model_paths=[model_path],
        device="cuda",
        default_dtype="float64",
        head="pbe_d3"
        )
    
    runner = RelaxCalc(calculator = calc, optimizer = BFGS, max_steps = 100000, traj_file = "relax.traj", fmax=fmax, relax_atoms = True, relax_cell = True)
    result = runner.calc(atoms)
    energy = atoms.get_potential_energy()

    with open("relax_results.json", "w") as f:
        json.dump(result, f, cls=MontyEncoder, indent=2)
        
    write('CONTCAR', atoms, format='vasp')

    return {"output_atoms": atoms, "energy": energy}


@job
def QHA_mof(atoms, model_path, fmax):
    
    calc = MACECalculator(
        model_paths=[model_path],
        device="cuda",
        default_dtype="float64",
        head="pbe_d3"
        )

    result = QHACalc(
    calc,
    t_step=1,
    t_max=650,
    pressure=1e-4,
    fmax=fmax,
    max_steps=10000,
    optimizer="LBFGS",
    on_imaginary_modes="warn",
    imaginary_freq_tol=-0.1,
    fix_imaginary_attempts=1,
    scale_factors=tuple(np.arange(0.97, 1.03, 0.01).tolist()),
    phonon_calc_kwargs={
        "min_length": 20.0,
        "atom_disp": 0.01,
        "write_total_dos": True ,
        "write_band_structure": True
    },
    ).calc(atoms)
    
    cp = result["heat_capacity_P"][300]  # J/mol/K
    gibb = result["gibbs_free_energies"][300]
    gibb2 = result["gibbs_free_energies"][-1]

    sub_data= [cp, gibb, gibb2]
    np.savetxt("select_values.txt", sub_data)
    
    result["qha"].plot_qha()
    plt.savefig(f"QHA.png")

    raw_G = result["gibbs_free_energies"]
    gibbs_energies = np.insert(raw_G, 0, np.nan)
    temperatures = result["temperatures"]

    data = np.column_stack((temperatures, gibbs_energies))

    header_full = "T (K)          Gibbs(eV)     "
    np.savetxt("thermal_properties_full.txt", data,
           fmt="%12.3f %15.7f",
           header=header_full)

    return {"thermal_properties": data}


@job
def relax_mp(atoms):
    write('POSCAR', atoms, format='vasp')

    model_name = "uma-s-1p1"
    predictor = pretrained_mlip.get_predict_unit(model_name, device="cuda")
    calc = FAIRChemCalculator(predictor, task_name="omat")

    runner = RelaxCalc(calculator = calc, optimizer = BFGS, max_steps = 100000, traj_file = "relax.traj", fmax=1e-6, relax_atoms = True, relax_cell = True)

    result = runner.calc(atoms)
    energy = atoms.get_potential_energy()


    with open("relax_results.json", "w") as f:
        json.dump(result, f, cls=MontyEncoder, indent=2)
        
    write('CONTCAR', atoms, format='vasp')

    return {"output_atoms": atoms, "energy": energy}

@job
def phonon_mp(atoms):


    model_name = "uma-s-1p1"
    predictor = pretrained_mlip.get_predict_unit(model_name, device = "cuda")
    calc = FAIRChemCalculator(predictor, task_name="omat")


    supercell_matrix = np.diag(
    np.round(np.ceil(20.0 / atoms.cell.lengths()))
    )

    phonon_calc = PhononCalc(calc, supercell_matrix = supercell_matrix).calc(atoms)

    phonon = phonon_calc["phonon"]

    phonon.run_mesh([20, 20, 20])
    phonon.run_thermal_properties(t_step=1,
                              t_max=1000,
                              t_min=0)
    tp_dict = phonon.get_thermal_properties_dict()
    temperatures   = np.array(tp_dict['temperatures'])
    free_energy    = np.array(tp_dict['free_energy'])
    entropy        = np.array(tp_dict['entropy'])
    heat_capacity  = np.array(tp_dict['heat_capacity'])

    data = np.column_stack((temperatures, free_energy, entropy, heat_capacity))

    # Define a header and save to a text file
    header = "T (K)          Free_energy          Entropy          Heat_capacity"
    np.savetxt("thermal_properties.txt", data, fmt="%12.3f %15.7f %15.7f %15.7f", header=header)

    return {"thermal_properties": data}

@job
def QHA_mp(atoms):
    atom_disp = 0.01
    fmax = 1e-7
    min_lengths = 20.0
    supercell_matrix = np.diag(
    np.round(np.ceil(min_lengths / atoms.cell.lengths()))
    )

    model_name = "uma-s-1p1"
    predictor = pretrained_mlip.get_predict_unit(model_name, device = "cuda")
    calc = FAIRChemCalculator(predictor, task_name="omat")

    qha_calc = QHACalc(calc, fmax=fmax, t_step = 1, pressure = 0.0001, optimizer="BFGS", relax_calc_kwargs={"traj_file": "relax.traj", "max_steps":100000}, phonon_calc_kwargs={"supercell_matrix": supercell_matrix, "atom_disp": atom_disp, "write_total_dos": True ,"write_band_structure": True})
    result = qha_calc.calc(atoms)

    raw_G = result["gibbs_free_energies"]
    gibbs_energies = np.insert(raw_G, 0, np.nan)
    temperatures = result["temperatures"]

    data = np.column_stack((temperatures, gibbs_energies))

    header_full = "T (K)          Gibbs(eV)     "
    np.savetxt("thermal_properties_full.txt", data,
           fmt="%12.3f %15.7f",
           header=header_full)

    return {"thermal_properties": data}


@job
def relax_gas(atoms):
    try:
        magmoms = atoms.get_initial_magnetic_moments()
        total_magmom = np.sum(magmoms)

        # Spin multiplicity = |total magnetization| + 1
        # I'm not sure if this is exactly correct but I verified for each gas that the spin_multiplicity matched what is expected
        spin_multiplicity = int(round(abs(total_magmom))) + 1

        # Store it
        atoms.info['spin'] = spin_multiplicity

        print(f"Total magnetization: {total_magmom:.2f}")
        print(f"Inferred spin multiplicity: {spin_multiplicity}")
    except:
        magmoms = None
        print("No magnetic moments found (non-spin-polarized calculation)")
        atoms.info['spin'] = 1  # Default to singlet

    model_name = "uma-s-1p1"
    predictor = pretrained_mlip.get_predict_unit(model_name, device="cuda")
    calc = FAIRChemCalculator(predictor, task_name="omol")

    runner = RelaxCalc(calculator = calc, optimizer = BFGS, max_steps = 100000, traj_file = "relax.traj", fmax=1e-3, relax_atoms = True, relax_cell = True)
    result = runner.calc(atoms)
    
    with open("relax_results.json", "w") as f:
        json.dump(result, f, cls=MontyEncoder, indent=2)
        
    mlip_energy = atoms.get_potential_energy()
    
    return {"result": result, "output_atoms": atoms, "mlip_energy": mlip_energy, "magmoms": magmoms, "spin_multiplicity": atoms.info['spin']}

@job
def gas_vibrations(atoms, mlip_energy):
    model_name = "uma-s-1p1"
    predictor = pretrained_mlip.get_predict_unit(model_name, device="cuda")
    atoms.calc = FAIRChemCalculator(predictor, task_name="omol")

    spin_multiplicity = atoms.info['spin']
    
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
            f.write("\nImaginary energies (eV, absolute values):\n")
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

    
    return {"output_atoms": atoms, "real_vibration_energies": real_energies, "imag_vibration_energies": imag_energies ,"spin_multiplicity": spin_multiplicity, "spin_quantum_number": spin, "geometry": geometry, "thermal_properties": data}

