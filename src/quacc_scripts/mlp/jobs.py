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

from dftd4.ase import DFTD4

import json
from monty.json import MontyEncoder

import logging
logging.basicConfig(level=logging.INFO)

import time
from pymatgen.core import Structure
import os

metals_3d = ["V", "Cr", "Mn", "Fe", "Co", "Ni", "W", "Mo"]

#just for oxides
hubbard_dict = {"Fe": 5.3, "Co": 3.32, "Cr": 3.7, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2, "O": 0.0}

#need to call the energy correction for the relax_job as well

def obtain_energy_correction(calc_name, structure):
    correction = 0.0
    potcar_map = None
    # get unique elements in order they appear in structure
    elements = list(dict.fromkeys(str(s.specie.symbol) for s in structure))
    
    if calc_name == "MACE_MPA_0" or calc_name == "PET_OAM_XL":
        potcar_base_path = "/home/ROSENGROUP/software/vasp/ase_potcars/vasp_potcars.original/potpaw_PBE"
        # default MP potcar map (from MPRelaxSet.yaml)
        potcar_map = {
            "Ac": "Ac", "Ag": "Ag", "Al": "Al", "Ar": "Ar", "As": "As",
            "Au": "Au", "B": "B", "Ba": "Ba_sv", "Be": "Be_sv", "Bi": "Bi",
            "Br": "Br", "C": "C", "Ca": "Ca_sv", "Cd": "Cd", "Ce": "Ce",
            "Cl": "Cl", "Co": "Co", "Cr": "Cr_pv", "Cs": "Cs_sv", "Cu": "Cu_pv",
            "Dy": "Dy_3", "Er": "Er_3", "Eu": "Eu", "F": "F", "Fe": "Fe_pv",
            "Ga": "Ga_d", "Gd": "Gd", "Ge": "Ge_d", "H": "H", "He": "He",
            "Hf": "Hf_pv", "Hg": "Hg", "Ho": "Ho_3", "I": "I", "In": "In_d",
            "Ir": "Ir", "K": "K_sv", "Kr": "Kr", "La": "La", "Li": "Li_sv",
            "Lu": "Lu_3", "Mg": "Mg_pv", "Mn": "Mn_pv", "Mo": "Mo_pv",
            "N": "N", "Na": "Na_pv", "Nb": "Nb_pv", "Nd": "Nd_3", "Ne": "Ne",
            "Ni": "Ni_pv", "Np": "Np", "O": "O", "Os": "Os_pv", "P": "P",
            "Pa": "Pa", "Pb": "Pb_d", "Pd": "Pd", "Pm": "Pm_3", "Pr": "Pr_3",
            "Pt": "Pt", "Pu": "Pu", "Rb": "Rb_sv", "Re": "Re_pv", "Rh": "Rh_pv",
            "Ru": "Ru_pv", "S": "S", "Sb": "Sb", "Sc": "Sc_sv", "Se": "Se",
            "Si": "Si", "Sm": "Sm_3", "Sn": "Sn_d", "Sr": "Sr_sv", "Ta": "Ta_pv",
            "Tb": "Tb_3", "Tc": "Tc_pv", "Te": "Te", "Th": "Th", "Ti": "Ti_pv",
            "Tl": "Tl_d", "Tm": "Tm_3", "U": "U", "V": "V_pv", "W": "W_pv",
            "Xe": "Xe", "Y": "Y_sv", "Yb": "Yb_2", "Zn": "Zn", "Zr": "Zr_sv",
        }

    elif calc_name == "UMA_OMAT":
        potcar_base_path = "/home/ROSENGROUP/software/vasp/ase_potcars/vasp_potcars.54/potpaw_PBE"
        #defualt OMAT potcar map
        potcar_map = {
            "Ac": "Ac", "Ag": "Ag", "Al": "Al", "Ar": "Ar", "As": "As",
            "Au": "Au", "B": "B", "Ba": "Ba_sv", "Be": "Be_sv", "Bi": "Bi",
            "Br": "Br", "C": "C", "Ca": "Ca_sv", "Cd": "Cd", "Ce": "Ce",
            "Cl": "Cl", "Co": "Co", "Cr": "Cr_pv", "Cs": "Cs_sv", "Cu": "Cu_pv",
            "Dy": "Dy_3", "Er": "Er_3", "Eu": "Eu", "F": "F", "Fe": "Fe_pv",
            "Ga": "Ga_d", "Gd": "Gd", "Ge": "Ge_d", "H": "H", "He": "He",
            "Hf": "Hf_pv", "Hg": "Hg", "Ho": "Ho_3", "I": "I", "In": "In_d",
            "Ir": "Ir", "K": "K_sv", "Kr": "Kr", "La": "La", "Li": "Li_sv",
            "Lu": "Lu_3", "Mg": "Mg_pv", "Mn": "Mn_pv", "Mo": "Mo_pv",
            "N": "N", "Na": "Na_pv", "Nb": "Nb_pv", "Nd": "Nd_3", "Ne": "Ne",
            "Ni": "Ni_pv", "Np": "Np", "O": "O", "Os": "Os_pv", "P": "P",
            "Pa": "Pa", "Pb": "Pb_d", "Pd": "Pd", "Pm": "Pm_3", "Pr": "Pr_3",
            "Pt": "Pt", "Pu": "Pu", "Rb": "Rb_sv", "Re": "Re_pv", "Rh": "Rh_pv",
            "Ru": "Ru_pv", "S": "S", "Sb": "Sb", "Sc": "Sc_sv", "Se": "Se",
            "Si": "Si", "Sm": "Sm_3", "Sn": "Sn_d", "Sr": "Sr_sv", "Ta": "Ta_pv",
            "Tb": "Tb_3", "Tc": "Tc_pv", "Te": "Te", "Th": "Th", "Ti": "Ti_pv",
            "Tl": "Tl_d", "Tm": "Tm_3", "U": "U", "V": "V_pv", "W": "W_sv",
            "Xe": "Xe", "Y": "Y_sv", "Yb": "Yb_3", "Zn": "Zn", "Zr": "Zr_sv",
        }
        

    if potcar_map != None:
        labels = []
        for element in elements:
            subdir = potcar_map.get(element, element)
            potcar_path = os.path.join(potcar_base_path, subdir, "POTCAR")
        
            if not os.path.exists(potcar_path):
                raise FileNotFoundError(f"POTCAR not found for {element} at {potcar_path}")
        
            with open(potcar_path, "r") as f:
                title = f.readline().strip()
        
            labels.append(title)

    if calc_name in ["UMA_OMAT", "PET_OAM_XL", "MACE_MPA_0"]:
        if  any(element in metals_3d for element in elements) and "O" in elements:
            hubbards = {element: hubbard_dict[element] for element in elements if element in hubbard_dict}
            processed_entry = ComputedStructureEntry(
                structure = structure,
                energy = 0.0,
                parameters = {'run_type': 'GGA+U', 'potcar_symbols': labels, 'hubbards': hubbards, 'is_hubbard': True}
            )
        else:
            processed_entry = ComputedStructureEntry(
                structure = structure,
                energy = 0.0,
                parameters = {'run_type': 'GGA', 'potcar_symbols': labels}
            )

        processed_entry.energy_adjustments = MaterialsProject2020Compatibility().get_adjustments(processed_entry)
        correction = processed_entry.correction

    return correction

def choose_calc(calc_name, atoms, dispersion_correction, dtype):
    
    if calc_name == "vasp_OMAT":
        import os
        os.environ["ASE_VASP_COMMAND"] = "srun vasp_std"
        
        from quacc.calculators.vasp import Vasp
        from fairchem.data.omat.vasp.sets import OMat24StaticSet

        calc_defaults = MPtoASEConverter(atoms=atoms).convert_input_set(OMat24StaticSet())
        calc_defaults |= {"pp_version": "54", "incar_copilot": "light"}
        
        calc = Vasp(calc_defaults)

    elif calc_name == "UMA_OMAT":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.predict import InferenceSettings
        
        checkpoint = "/home/bd8619/.cache/fairchem/models--facebook--UMA/snapshots/7210de6fe86ad94854b21b881fefbcfdfeab373b/checkpoints/uma-s-1p2.pt"
        predictor = load_predict_unit(checkpoint, device="cuda", inference_settings=InferenceSettings(base_precision_dtype=dtype))
        calc = FAIRChemCalculator(predictor, task_name="omat")

    elif calc_name == "PET_OAM_XL":
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(checkpoint_path="/scratch/gpfs/ROSENGROUP/bd8619/checkpoints/PET-OAM-XL/pet-oam-xl-v1.0.0.ckpt", dtype= dtype, device="cuda")

    
    elif calc_name == "PET_OMATPES_L":
        from upet.calculator import UPETCalculator
        calc = UPETCalculator(checkpoint_path="/scratch/gpfs/ROSENGROUP/bd8619/checkpoints/PET-OMATPES-L/pet-omatpes-l-v0.1.0.ckpt", dtype=dtype, device="cuda")

    elif calc_name == "MACE_MPA_0":
        from mace.calculators import MACECalculator

        calc = MACECalculator(model_paths=["/scratch/gpfs/ROSENGROUP/bd8619/mlip_models/MACE-MPA-0/mace-mpa-0-medium.model"], device="cuda", default_dtype=dtype)

    elif calc_name == "MACE_MATPES_r2SCAN_0":
        from mace.calculators import MACECalculator

        calc = MACECalculator(
        model_paths=["/scratch/gpfs/ROSENGROUP/bd8619/mlip_models/MACE-MATPES-r2SCAN-0/MACE-matpes-r2scan-omat-ft.model"], device="cuda", default_dtype=dtype)

    elif calc_name == "MACE_MH_1_MATPES_r2SCAN":  #the built in dispersion correction here is just the TorchDFTD3Calculator
        from mace.calculators import mace_mp

        if dispersion_correction == True:
            #calc = mace_mp(model="/scratch/gpfs/ROSENGROUP/bd8619/mlip_models/MACE-MH-1-MATPES-R2SCAN/mace-mh-1.model", default_dtype=dtype, device="cuda", dispersion=True, dispersion_xc="r2scan", head="matpes_r2scan")
            calc = mace_mp(model="/scratch/gpfs/ROSENGROUP/bd8619/mlip_models/MACE-MH-1-MATPES-R2SCAN/mace-mh-1.model", default_dtype=dtype, device="cuda", head="matpes_r2scan", dispersion=False)
            dft_d4 = DFTD4(method="r2scan")
            calc = SumCalculator([calc, dft_d4])
        else:
            calc = mace_mp(model="/scratch/gpfs/ROSENGROUP/bd8619/mlip_models/MACE-MH-1-MATPES-R2SCAN/mace-mh-1.model", default_dtype=dtype, device="cuda", head="matpes_r2scan")
    
    elif calc_name == "TensorNet_MatPES_r2SCAN":
        import matgl
        from matgl.ext.ase import PESCalculator
        import torch

        model = matgl.load_model("/home/bd8619/.cache/matgl/models--materialyze--TensorNet-PES-MatPES-r2SCAN-2025.2-m/snapshots/0e4ef6457eb41db1e8b957bed9337fd4fbac3d89/")
        
        if dtype == "float64":
            matgl.float_th = torch.float64
            model = model.double()
        elif dtype == "float32":
            matgl.float_th = torch.float32
            model = model.float()
            
        calc = PESCalculator(potential=model)

    if dispersion_correction==True and calc_name in ["UMA_OMAT", "PET_OAM_XL", "MACE_MPA_0"]:
        device="cpu"
        dft_d3 = TorchDFTD3Calculator(device=device, xc="pbe", damping="bj")
        calc = SumCalculator([calc, dft_d3])
    elif dispersion_correction==True and calc_name in ["PET_OMATPES_L", "MACE_MATPES_r2SCAN_0", "TensorNet_MatPES_r2SCAN"]:
        device="cpu"
        #dft_d3 = TorchDFTD3Calculator(device=device, xc="r2scan", damping="bj")
        dft_d4 = DFTD4(method="r2scan")
        calc = SumCalculator([calc, dft_d4])
    
    return calc

@job
def QHA_material(atoms, calc_name, fmax, dispersion_correction=False, dtype="float64"):

    start_time = time.perf_counter()
    calc = choose_calc(calc_name, atoms, dispersion_correction, dtype)

    structure = AseAtomsAdaptor.get_structure(atoms)
    energy_correction = obtain_energy_correction(calc_name, structure)
    
    result = QHACalc(
    calc,
    t_step=1,
    t_max=1000,
    pressure=0.000101325,
    fmax=fmax,
    max_steps=100000,
    optimizer="FIRE",
    on_imaginary_modes="warn",
    imaginary_freq_tol=-0.1,
    fix_imaginary_attempts=3,
    scale_factors=tuple(np.arange(0.97, 1.03, 0.01).tolist()),
    phonon_calc_kwargs={
        "min_length": 20.0,
        "atom_disp": 0.01,
        "write_total_dos": True ,
        "write_band_structure": True
    },
    write_ha_phonon=True,
    store_ha_phonon=True
    ).calc(atoms)

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    result["qha"].plot_qha()
    plt.savefig(f"QHA.png")
    
    gibbs_energies = result["gibbs_free_energies"] + energy_correction
    temperatures = result["temperatures"]

    data = np.column_stack((temperatures, gibbs_energies))

    header_full = "T (K)          Gibbs(eV)     "
    np.savetxt("thermal_properties_full.txt", data,
           fmt="%12.3f %15.7f",
           header=header_full)

    for i, ha_result in enumerate(result["ha"]):
        volume_index = i
        scale_factor = result["scale_factors"][i]
        phonopy_obj = ha_result["phonon"]
        mesh_dict   = phonopy_obj.get_mesh_dict()
        frequencies_all = mesh_dict["frequencies"]
        qpoints = mesh_dict['qpoints']

        np.savetxt(f"all_frequencies_vol_{volume_index}_scale_{scale_factor:.3f}.txt", frequencies_all.flatten(),
        header=f"All frequencies from mesh (THz), volume {volume_index}, scale {scale_factor:.3f}")

    # Or save with q-point info
        with open(f"frequencies_by_qpoint_vol_{volume_index}_scale_{scale_factor:.3f}.txt", "w") as f:
            f.write(f"# Volume {volume_index}, scale {scale_factor:.3f}\n")
            f.write("# q-point_index  qx  qy  qz  frequencies(THz)\n")
            for q_indx, (qpt, freqs) in enumerate(zip(qpoints, frequencies_all)):
                f.write(f"{i}  {qpt[0]:.6f}  {qpt[1]:.6f}  {qpt[2]:.6f}  ")
                f.write("  ".join(f"{freq:.6f}" for freq in freqs))
                f.write("\n")

        #Save phonopy mesh settings(not sure if right)
        if i == len(result["ha"]) // 2:  # save from middle (scale~1.0) volume
            phonopy_settings = {
            "supercell_matrix": phonopy_obj.supercell_matrix.tolist(),
            "mesh":             list(phonopy_obj.mesh_numbers),
            "primitive_matrix": phonopy_obj.primitive_matrix.tolist()
                                if hasattr(phonopy_obj.primitive_matrix, "tolist")
                                else str(phonopy_obj.primitive_matrix),
            "symprec":          phonopy_obj.symmetry.tolerance,
            "atom_disp":        0.01,   # from phonon_calc_kwargs
            "n_atoms_supercell": len(phonopy_obj.supercell),
            "n_atoms_primitive": len(phonopy_obj.primitive),
        }

    
    return {"thermal_properties": data, "energy_correction": energy_correction, "time": execution_time, "phonopy_settings": phonopy_settings}


@job
def relax_material(atoms, calc_name, fmax, dispersion_correction=False, dtype="float64"):
    start_time = time.perf_counter()
    write('POSCAR', atoms, format='vasp')

    calc = choose_calc(calc_name, atoms, dispersion_correction, dtype)
    atoms.calc = calc
    
    filtered_atoms = FrechetCellFilter(atoms)

    dyn = BFGS(filtered_atoms, trajectory='relaxation.traj', logfile='relax.log')
    dyn.run(fmax=fmax)

    structure = AseAtomsAdaptor.get_structure(atoms)
    energy_correction = obtain_energy_correction(calc_name, structure)

    energy = atoms.get_potential_energy() + energy_correction
        
    write('CONTCAR', atoms, format='vasp')
    atoms.info = {}

    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    return {"output_atoms": atoms, "energy": energy, "energy_correction": energy_correction, "time": execution_time}

def mini_choose_calc(method):
    if method =="meta":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator
        model_name = "uma-s-1p2"
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

#@job
def relax_gas(atoms, fmax, spin_multiplicity, method):

    #spin_multiplicity = atoms.info['spin']
    #try:
    #    magmoms = atoms.get_initial_magnetic_moments()
    #    total_magmom = np.sum(magmoms)

        # Spin multiplicity = |total magnetization| + 1
        # I'm not sure if this is exactly correct but I verified for each gas that the spin_multiplicity matched what is expected
       # spin_multiplicity = int(round(abs(total_magmom))) + 1

        # Store it
      #  atoms.info['spin'] = spin_multiplicity

       # print(f"Total magnetization: {total_magmom:.2f}")
        #print(f"Inferred spin multiplicity: {spin_multiplicity}")
   # except:
    #    magmoms = None
     #   print("No magnetic moments found (non-spin-polarized calculation)")
      #  atoms.info['spin'] = 1  # Default to singlet

    
    atoms.info['spin'] = spin_multiplicity

    atoms.set_cell([20, 20, 20])
    atoms.pbc = False
    atoms.center()
    
    calc = mini_choose_calc(method)
    atoms.calc = calc
    
    dyn = BFGS(atoms, trajectory='relaxation.traj', logfile='relax.log')
    dyn.run(fmax=fmax)
    
    write('CONTCAR', atoms, format='vasp')
    
    mlip_energy = atoms.get_potential_energy()
    
    with open("output_energy.txt", "w") as file:
        file.write(str(mlip_energy))
    
    return {"output_atoms": atoms, "mlip_energy": mlip_energy, "magmoms": magmoms, "spin_multiplicity": atoms.info['spin']}

#@job
def gas_vibrations(atoms, mlip_energy, spin_multiplicity, method):
    
 
    calc = mini_choose_calc(method)
    atoms.calc = calc
    atoms.info['spin'] = spin_multiplicity

    atoms.pbc = False
    
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

    
    return {"output_atoms": atoms, "real_vibration_energies": real_energies, "imag_vibration_energies": imag_energies ,"spin_multiplicity": spin_multiplicity, "spin_quantum_number": spin, "geometry": geometry, "thermal_properties": data}

