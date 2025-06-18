import os
import numpy as np
from snqs_kpm.chem.get_molecular_data import MolecularData


# Bohr-Angstrom correlation coefficient (https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0)
bohr_angs = 0.529177210903

def meanfield(
    path,
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    outpath=".",
):  # pylint: disable=too-many-arguments
    r"""Generates a file from which the mean field electronic structure
    of the molecule can be retrieved.

    This function performs the Hartree-Fock (HF) calculation 
    for the polyatomic system using the quantum
    chemistry packages ``PySCF``. The mean field electronic
    structure is saved in an hdf5-formatted file.

    The charge of the molecule can be given to simulate cationic/anionic systems.
    Also, the spin multiplicity can be input to determine the number of unpaired electrons
    occupying the HF orbitals. 

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): molecule label
        charge (int): net charge of the system
        mult (int): Spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` for
            :math:`N_\mathrm{unpaired}` unpaired electrons occupying the HF orbitals.
            Possible values for ``mult`` are :math:`1, 2, 3, \ldots`. If not specified,
            a closed-shell HF state is assumed.
        basis (str): Atomic basis set used to represent the HF orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): Quantum chemistry package used to solve the Hartree-Fock equations.
        outpath (str): path to output directory

    Returns:
        str: absolute path to the file containing the mean field electronic structure

    **Example**

    >>> symbols, coordinates = (['H', 'H'], np.array([0., 0., -0.66140414, 0., 0., 0.66140414]))
    >>> meanfield(symbols, coordinates, name="h2")
    ./h2_pyscf_sto-3g
    """

    if coordinates.size != 3 * len(symbols):
        raise ValueError(
            f"The size of the array 'coordinates' has to be 3*len(symbols) = {3 * len(symbols)};"
            f" got 'coordinates.size' = {coordinates.size}"
        )

    package = package.strip().lower()

    if package not in "pyscf":
        error_message = (
            f"Integration with quantum chemistry package '{package}' is not available. \n Please set"
            f" 'package' to 'pyscf'."
        )
        raise TypeError(error_message)

    filename = name + "_" + package.lower() + "_" + basis.strip()
    path_to_file = os.path.join(outpath.strip(), filename)

    geometry = [
        [symbol, tuple(np.array(coordinates)[3 * i : 3 * i + 3] * bohr_angs)]
        for i, symbol in enumerate(symbols)
    ]

    molecule = MolecularData(path, geometry, basis, mult, charge, filename=path_to_file)

    if package == "pyscf":
        # pylint: disable=import-outside-toplevel
        from snqs_kpm.chem import run_pyscf

        run_pyscf(molecule, run_scf=1, verbose=0)

    return path_to_file
