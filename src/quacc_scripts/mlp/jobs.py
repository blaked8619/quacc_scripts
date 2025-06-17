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

@job
def relax_mof(atoms, model_path):
    write('POSCAR', atoms, format='vasp')
    calc = MACECalculator(
        model_paths=[model_path],
        device="cuda",
        default_dtype="float64"
        )
    runner = RelaxCalc(calculator = calc, optimizer = BFGS, max_steps = 100000, traj_file = "relax.traj", fmax=1e-3, relax_atoms = True, relax_cell = True)
    result = runner.calc(atoms)
    
    write('CONTCAR', atoms, format='vasp')

    return {"output_atoms": atoms}

@job
def phonon_mof(atoms, model_path):
    
    calc = MACECalculator(
        model_paths=[model_path],
        device="cuda",
        default_dtype="float64"
        )
    supercell_matrix = np.diag(
    np.round(np.ceil(20.0 / atoms.cell.lengths()))
    )
    
    phonon_calc = PhononCalc(calc, supercell_matrix = supercell_matrix).calc(atoms)

    phonon = phonon_calc["phonon"]
    heat_capacity = phonon.get_thermal_properties_dict()["heat_capacity"]


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


