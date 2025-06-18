import torch
import numpy as np
import scipy as sp
import random
import os
import sys
import gc

import pickle

from snqs_kpm.chem import MolecularData
from snqs_kpm.transforms import get_fermion_operator, jordan_wigner
from snqs_kpm.chem import dipole_moments

def load_molecule(fname, hamiltonian_fname=None, 
                  verbose=True):
    path = os.path.split(fname)[0]
    fn = os.path.split(fname)[1]
    
    if os.path.isdir(fname):
        fname = os.path.join(fname, os.path.split(fname)[-1])

    print(f"Loading molecule from {fname}.hdf5", end="...")
    molecule = MolecularData(path+"/"+fn,filename=fn)
    molecule.load()
    print("done.")

    active_space_start = 0
    active_space_stop = molecule.n_orbitals

    if hamiltonian_fname is None:
        hamiltonian_fname = fname + "_qubit_hamiltonian.pkl"

    try:
        print(f"Loading molecule from {hamiltonian_fname}", end="...")
        with open(hamiltonian_fname, 'rb') as f:
            qubit_hamiltonian = pickle.load(f)
        print("done.")

    except:
        print("failed.  Reverting to solving for qubit_hamiltonian", end="...")
        # Get the Hamiltonian in an active space.
        molecular_hamiltonian = molecule.get_molecular_hamiltonian(
            occupied_indices=range(active_space_start),
            active_indices=range(active_space_start, active_space_stop)
        )

        # Map operator to fermions and qubits.
        fermion_hamiltonian = get_fermion_operator(molecular_hamiltonian)
        qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
        print("done")

    if verbose:
        print('{}.hdf5 has:'.format(fname))
        print('\tHartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))
        print('\tMP2 energy of {} Hartree.'.format(molecule.mp2_energy))
        print('\tCCSD energy of {} Hartree.'.format(molecule.ccsd_energy))
        print('\tFCI energy of {} Hartree.'.format(molecule.fci_energy))

        print(f"\nHamiltonian for {fname}.hdf5 has:")
        # # display(qubit_hamiltonian)
        n_qubits = qubit_hamiltonian.many_body_order()
        n_alpha = molecule.get_n_alpha_electrons()
        n_beta = molecule.get_n_beta_electrons()
        print(f"\t{n_qubits} qubits (orbitals), with {molecule.n_electrons} electrons ({n_alpha}/{n_beta} alpha/beta).")

    return molecule, qubit_hamiltonian

def load_dipole(fname,
                dipole_X_fname=None, 
                dipole_Y_fname=None, 
                dipole_Z_fname=None):

    if os.path.isdir(fname):
        fname = os.path.join(fname, os.path.split(fname)[-1])

    print(f"Loading molecule from {fname}.hdf5", end="...")
    molecule = MolecularData('',filename=fname)
    molecule.load()
    print("done.")

    if dipole_X_fname is None:
        dipole_X_fname = fname + "_qubit_dipole_X.pkl"

    if dipole_Y_fname is None:
        dipole_Y_fname = fname + "_qubit_dipole_Y.pkl"

    if dipole_Z_fname is None:
        dipole_Z_fname = fname + "_qubit_dipole_Z.pkl"

    try: 
        print(f"Loading molecule from {dipole_X_fname}", end="...")
        with open(dipole_X_fname, 'rb') as f:
            qubit_dipole_X = pickle.load(f)
        print("done.")
    except:
        print("failed.  Reverting to solving for qubit_dipole_X", end="...")
        dipole_obs = dipole_moments(
            molecule.symbols, 
            molecule.coordinates.flatten(), 
            charge=0)
        qubit_dipole_X = dipole_obs[0]

    try:
        print(f"Loading molecule from {dipole_Y_fname}", end="...")
        with open(dipole_Y_fname, 'rb') as f:
            qubit_dipole_Y = pickle.load(f)
        print("done.")
    except:
        print("failed.  Reverting to solving for qubit_dipole_Y", end="...")
        dipole_obs = dipole_moments(
            molecule.symbols, 
            molecule.coordinates.flatten(), 
            charge=0)
        qubit_dipole_Y = dipole_obs[1]

    try:
        print(f"Loading molecule from {dipole_Z_fname}", end="...")
        with open(dipole_Z_fname, 'rb') as f:
            qubit_dipole_Z = pickle.load(f)
        print("done.")
    except:
        print("failed.  Reverting to solving for qubit_dipole_Z", end="...")
        dipole_obs = dipole_moments(
            molecule.symbols, 
            molecule.coordinates.flatten(), 
            charge=0)
        qubit_dipole_Z = dipole_obs[2]
    return qubit_dipole_X, qubit_dipole_Y, qubit_dipole_Z

def load_dipole_series(fname,
                dipole_series_fname=None):

    if os.path.isdir(fname):
        fname = os.path.join(fname, os.path.split(fname)[-1])

    # print(f"Loading molecule from {fname}.hdf5", end="...")
    # molecule = MolecularData(filename=fname)
    # molecule.load()
    # print("done.")

    if dipole_series_fname is None:
        dipole_series_fname = fname + "_qubit_dipole_series.pkl"

    try: 
        print(f"Loading molecule from {dipole_series_fname}", end="...")
        with open(dipole_series_fname, 'rb') as f:
            qubit_dipole_series = pickle.load(f)
        print("done.")
    except:
        print("failed.  Reverting to solving for qubit_dipole_series", end="...")
        sys.exit(1)  # 终止程序，返回值为1表示异常终止
        # dipole_obs = dipole_of(
        #     molecule.symbols, 
        #     molecule.coordinates.flatten(), 
        #     charge=0)
        # qubit_dipole_X = dipole_obs[0]
    return qubit_dipole_series

def set_global_seed(seed=-1):
    if seed < 0:
        seed = random.randint(0, 2 ** 32)
    print("\n------------------------------------------")
    print(f"\tSetting global seed using {seed}.")
    print("------------------------------------------\n")
    random.seed(seed)
    np.random.seed(random.randint(0, 2 ** 32))
    np.random.seed(random.randint(0, 2 ** 32))
    torch.manual_seed(random.randint(0, 2 ** 32))
    torch.cuda.manual_seed(random.randint(0, 2 ** 32))
    torch.cuda.manual_seed_all(random.randint(0, 2 ** 32))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = True

def mk_dir(dir, quiet=False):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
            if not quiet:
                print('created directory: ', dir)
        except OSError as exc:  # Guard against race condition
            if exc.errno != exc.errno.EEXIST:
                raise
        except Exception:
            pass
    else:
        if not quiet:
            print('directory already exists: ', dir)

def print_memory(name, arr):
    try:
        b = arr.nbytes
    except:
        b = sys.getsizeof(arr.storage())
        # try:
        #     b = sys.getsizeof(arr.storage())
        # except:
        #     b = asizeof.asizeof(arr)
    if b > 10**6:
        print(f"{name} ({arr.dtype}) : {b/10**9:.4f}GB")
    else:
        print(f"{name} ({arr.dtype}) : {b/10**6:.4f}MB")

def print_tensors_on_gpu():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data')):
                if obj.is_cuda:
                    print(f"type : {type(obj)}, size : {obj.size()}, dtype : {obj.dtype}, device : {obj.device}, has_grads : {obj.grad is not None}")
        except:
            pass
    for i in range(4):
        try:
            print( torch.cuda.memory_summary(device=i, abbreviated=False) )
        except:
            pass

import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map