import quacc
from jobflow import job
from jobflow_remote import submit_flow, set_run_config
from ase.io import write
from mace.calculators import MACECalculator
from ase.optimize import BFGS
from ase.calculators.emt import EMT
from matcalc import PhononCalc
import numpy as np
from contextlib import redirect_stdout
from pymatgen.core import Structure


@job
def relax_mof(atoms, model_path):
    calc = MACECalculator(
        model_paths=[model_path],
        stress=True
        )
    atoms.calc = calc

    opt = BFGS(atoms, logfile="relax.log", trajectory="relax.traj")
    opt.run(fmax=1e-3, steps=1000000)

    write('CONTCAR', atoms, format='vasp')

    return {"output_atoms": atoms}

@job
def phonon_mof(atoms, model_path):
    structure = Structure.from_ase_atoms(atoms)
    
    calc = MACECalculator(
        model_paths=[model_path],
        stress=True
        )
    atoms.calc = calc

    phonon_calc = PhononCalc(calc, supercell_matrix = ((1,0,0),(0,1,0),(0,0,1))).calc(structure)

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


