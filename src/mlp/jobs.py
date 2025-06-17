import quacc
from jobflow import job
from jobflow_remote import submit_flow, set_run_config
from ase.io import write
from mace.calculators import MACECalculator
from ase.optimize import BFGS



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

    return {"final_atoms": atoms}

