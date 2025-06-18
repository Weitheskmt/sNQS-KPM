import pickle
import os
from snqs_kpm.chem import (run_psi4,
                           geometry_from_pubchem,
                           get_H2_chains,
                           MolecularData,
                           dipole_moments,
                           active_space)
from snqs_kpm.transforms import get_fermion_operator, jordan_wigner
import numpy as np

MOLECULE_LIST = np.array(["H2O", "NH3", "C2", "N2", "O2", "H2S", "C2H4O", "Li2CO3", "C6H8", "CNa2O3"])

H2_CHAINS_LIST = np.array(["H2_1", "H2_2", "H2_3", "H2_4", "H2_5","H2_6", "H2_7", "H2_8", "H2_9", "H2_10", 
                  "H2_11", "H2_12", "H2_13", "H2_14", "H2_15"])


def get_geometry(molecule_name, MOLECULE_LIST=MOLECULE_LIST, verbose=True):
    if verbose and (molecule_name not in MOLECULE_LIST):
        print(f"Warning: {molecule_name} is not one of the molecules used in the paper" + 
               "- that's not wrong, but just know it's not recreating the published results!")
        
    # the molecules below are not in PubChem - don't know why.
    if molecule_name=="C2":
        geometry = [('C', [0.0, 0.0, 0.0]), ('C', [0.0, 0.0, 1.26])]

    elif molecule_name=="CNa2O3":
        geometry = [('Na', [5.4641, 0.25, 0.0]),
                    ('Na', [2.0, 0.25, 0.0]),
                    ('O', [4.5981, 0.75, 0.0]),
                    ('O', [2.866, 0.75, 0.0]),
                    ('O', [3.732, -0.75, 0.0]),
                    ('C', [3.732, 0.25, 0.0])]
    elif molecule_name=="C2H4O":
        geometry = [('O', [4.269, 0.2915, 0.0]),
                    ('C', [2.5369, 0.2915, 0.0]),
                    ('C', [3.403, -0.2085, 0.0]),
                    ('H', [2.0, 0.6015, 0.0]),
                    ('H', [2.8469, 0.8285, 0.0]),
                    ('H', [2.2269, -0.2454, 0.0]),
                    ('H', [3.403, -0.8285, 0.0])]
        
    elif molecule_name=="C3H6":
        geometry = [('O', [4.269, 0.2915, 0.0]),
                    ('C', [2.5369, 0.2915, 0.0]),
                    ('C', [3.403, -0.2085, 0.0]),
                    ('H', [2.0, 0.6015, 0.0]),
                    ('H', [2.8469, 0.8285, 0.0]),
                    ('H', [2.2269, -0.2454, 0.0]),
                    ('H', [3.403, -0.8285, 0.0])]
    else:
        if molecule_name=="Li2O":
            # Li2O returns a different molecule - again, don't know why.
            molecule_name = "Lithium Oxide"
        geometry = geometry_from_pubchem(molecule_name)
        
    return geometry


def generate_mol_data(mol_name,
                      path_root="./molecules",
                      H2_CHAINS_LIST=H2_CHAINS_LIST, 
                      MOLECULE_LIST=MOLECULE_LIST,
                      basis='sto-3g',
                      active_electrons=None,
                      active_orbitals=None,
                      run_scf=True, 
                      run_mp2=False, 
                      run_cisd=False, 
                      run_ccsd=True, 
                      run_fci=True):
    
    path = path_root + "/" + f"{mol_name}"

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

    if mol_name in H2_CHAINS_LIST:
        geometry = get_H2_chains(int(mol_name[3:]))
    else:
        geometry = get_geometry(mol_name,MOLECULE_LIST)

    symbols = [atom[0] for atom in geometry]
    coordinates = np.array([atom[1] for atom in geometry])
    molecule = MolecularData(
                path=path,
                geometry=geometry,
                basis=basis,
                multiplicity=1,
                charge=0,
                filename=mol_name)
    
    molecule = run_psi4(molecule, 
                        run_scf=run_scf, 
                        run_mp2=run_mp2, 
                        run_cisd=run_cisd, 
                        run_ccsd=run_ccsd, 
                        run_fci=run_fci)
    
    
    core, active = active_space(molecule.n_electrons, 
                                molecule.n_orbitals, 
                                active_electrons=active_electrons, 
                                active_orbitals=active_orbitals)
    
    hamiltonian = molecule.get_molecular_hamiltonian(occupied_indices=core,
                                                     active_indices=active)
    
    dipole = dipole_moments(path=path,
                            symbols=molecule.symbols, 
                            coordinates=molecule.coordinates.flatten(),
                            core=core,
                            active=active, 
                            charge=0,
                            basis=basis)
    
    fermion_hamiltonian = get_fermion_operator(hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)

    mol_dipole_series =[]
    for i in range(3):
        if len(dipole[i].terms) !=0:
            mol_dipole_series.append(dipole[i])


    with open(path+f"/{mol_name}"+"_qubit_dipole_series.pkl",'wb') as f:
        pickle.dump(mol_dipole_series,f)

    with open(path+f"/{mol_name}"+"_qubit_hamiltonian.pkl",'wb') as f:
        pickle.dump(qubit_hamiltonian,f)

    molecule.save()

