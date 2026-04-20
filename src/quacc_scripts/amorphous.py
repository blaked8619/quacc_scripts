import torch
import torch.serialization
torch.serialization.add_safe_globals([slice])

from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from ase.io import read
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

adaptor = AseAtomsAdaptor()
GPa_to_eV_A3 = 0.0062415

@job
def nvt_sim(structure, checkpoint_path):
  atoms = adaptor.get_atoms(structure) * (2,2,2)
  
  predictor = load_predict_unit(checkpoint_path+"inference_ckpt.pt")
  calc = FAIRChemCalculator(predictor, task_name="odac")
  atoms.calc = calc
  
  BFGS(atoms).run(fmax=0.01)

  # ── Convergence parameters ────────────────────────────────────────────────────  
  TARGET_TEMP    = 300.0   # K
  TEMP_TOL       = 2.0    # K — stop if avg temp within this of target
  WINDOW         = 50     # steps to average temperature over
  CHECK_INTERVAL = 10     # check every N steps
  
  temp_history = []
  conv_log = open("convergence.log", "w", buffering=1)
  converged = False
  def check_convergence():
    global converged
    ekin = atoms.get_kinetic_energy()
    temp = ekin / (1.5 * len(atoms) * kB)
    temp_history.append(temp)
    if len(temp_history) >= WINDOW:
        avg_temp = np.mean(temp_history[-WINDOW:])
        std_temp = np.std(temp_history[-WINDOW:])
        conv_log.write(f"Step {dyn.get_number_of_steps()}: "
                       f"T={temp:.1f} K, avg={avg_temp:.1f} K, std={std_temp:.1f} K\n")
        conv_log.flush()
        if abs(avg_temp - TARGET_TEMP) < TEMP_TOL and std_temp < TEMP_TOL * 3:
            conv_log.write(f"Converged at step {dyn.get_number_of_steps()}\n")
            conv_log.flush()
            converged = True
            raise StopIteration

  c = FixCom()
  atoms.set_constraint(c)
  MaxwellBoltzmannDistribution(atoms, temperature_K=300.0)
  Stationary(atoms)

  dyn = Langevin(
    atoms=atoms,
    timestep=1.0*fs,
    temperature_K=300,
    friction=1e-2,
    logfile="npt_log.log",
  )
  traj = Trajectory("md.traj", "w", atoms)
  dyn.attach(traj.write,          interval=50)
  dyn.attach(check_convergence,   interval=CHECK_INTERVAL)

  start = time.time()

  try:
    dyn.run(steps=10000)
  except (StopIteration, RuntimeError):
    if converged:
        print("NVT converged", flush=True)

  traj.close()
  elapsed = time.time() - start
  conv_log.close()
  print(f"NVT took {elapsed:.1f} s ({elapsed/60:.2f} min)")

  momenta = atoms.get_momenta()
  np.save("momenta.npy", momenta)

  atoms.set_constraint([])
  return {"output_atoms": atoms}

@job
def npt_sim(atoms, checkpoint_path):
  predictor = load_predict_unit(checkpoint_path+"inference_ckpt.pt")
  calc = FAIRChemCalculator(predictor, task_name="odac")
  atoms.calc = calc

  dyn = MTKNPT(
    atoms=atoms,
    timestep=1*fs,
    temperature_K = 300, #K
    pressure_au =  0.5*GPa_to_eV_A3, #eV/Ang^3
    tdamp = 100 * fs,
    pdamp = 1000 * fs,
    logfile="npt_log.log",
    trajectory = "md_npt.traj"
    )

  start = time.time()
  dyn.run(steps=5000)
  elapsed = time.time() - start

  print(f"NPT took {elapsed:.1f} s ({elapsed/60:.2f} min)")
  return {"output_atoms": atoms}

@job
def spring_sim(atoms, checkpoint_path):
  predictor = load_predict_unit(checkpoint_path+"inference_ckpt.pt")
  calc = FAIRChemCalculator(predictor, task_name="odac")
  atoms.calc = calc
  
  dyn = MTKNPT(
    atoms=atoms,
    timestep=1*fs,
    temperature_K = 300, #K
    pressure_au =  0.0001*GPa_to_eV_A3, #eV/Ang^3
    tdamp = 100 * fs,
    pdamp = 1000 * fs,
    logfile="spring_log.log",
    trajectory = "md_spring.traj"
    )

  start = time.time()
  dyn.run(steps=5000)
  elapsed = time.time() - start

  print(f"Spring took {elapsed:.1f} s ({elapsed/60:.2f} min)")
  return {"output_atoms": atoms}


  
  
