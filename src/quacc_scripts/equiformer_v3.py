import torch
import torch.serialization
torch.serialization.add_safe_globals([slice])

from ase.io import read, write
from ase.visualize import view
from ase.optimize import BFGS
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.velocitydistribution import Stationary
from ase.md.langevin import Langevin
from ase.md.nose_hoover_chain import MTKNPT
from ase.units import fs, kB
from ase.constraints import FixCom
import time
from ase.io import Trajectory
import numpy as np
import sys

from pymatgen.io.ase import AseAtomsAdaptor
from jobflow import Flow, job
from jobflow.core.job import Job
from jobflow_remote import submit_flow, set_run_config
from monty.serialization import loadfn, dumpfn

import json
from monty.json import MontyEncoder
from ase.filters import FrechetCellFilter

@job
def single_point(structure, checkpoint_path, taskname):
  from fairchem.core.common.relaxation.ase_utils import OCPCalculator
  atoms = AseAtomsAdaptor().get_atoms(structure)  # convert back inside job

  calc = OCPCalculator(
        checkpoint_path=checkpoint_path + "inference_ckpt.pt"
    )
  
  atoms.calc = calc

  energy = atoms.get_potential_energy()
  forces = atoms.get_forces()
  stress = atoms.get_stress(voigt=True)

  max_force = np.max(np.linalg.norm(forces, axis = 1))
  
  return {"energy": float(energy), "forces": forces.tolist(), "stress": stress.tolist(), "max_force": float(max_force)}
