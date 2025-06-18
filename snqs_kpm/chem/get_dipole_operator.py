from snqs_kpm.chem import atomic_numbers
from snqs_kpm.chem import meanfield
from snqs_kpm.chem import one_particle
from snqs_kpm.chem import two_particle
from snqs_kpm.chem import MolecularData
from snqs_kpm.ops import FermionOperator
from snqs_kpm.transforms import jordan_wigner
import numpy as np

def dipole_moments(
    path,
    symbols,
    coordinates,
    name="molecule",
    charge=0,
    mult=1,
    basis="sto-3g",
    package="pyscf",
    core=None,
    active=None,
    mapping="jordan_wigner",
    cutoff=1.0e-12,
    outpath=".",
    wires=None,
):
    r"""Computes the electric dipole moment operator in the Pauli basis.

    The second quantized dipole moment operator :math:`\hat{D}` of a molecule is given by

    .. math::

        \hat{D} = -\sum_{\alpha, \beta} \langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}] + \hat{D}_\mathrm{n}.

    In the equation above, the indices :math:`\alpha, \beta` run over the basis of Hartree-Fock
    molecular orbitals and the operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the
    electron creation and annihilation operators, respectively. The matrix elements of the
    position operator :math:`\hat{{\bf r}}` are computed as

    .. math::

        \langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle = \sum_{i, j}
         C_{\alpha i}^*C_{\beta j} \langle i \vert \hat{{\bf r}} \vert j \rangle,

    where :math:`\vert i \rangle` is the wave function of the atomic orbital,
    :math:`C_{\alpha i}` are the coefficients defining the molecular orbitals,
    and :math:`\langle i \vert \hat{{\bf r}} \vert j \rangle`
    is the representation of operator :math:`\hat{{\bf r}}` in the atomic basis.

    The contribution of the nuclei to the dipole operator is given by

    .. math::

        \hat{D}_\mathrm{n} = \sum_{i=1}^{N_\mathrm{atoms}} Z_i {\bf R}_i \hat{I},


    where :math:`Z_i` and :math:`{\bf R}_i` denote, respectively, the atomic number and the
    nuclear coordinates of the :math:`i`-th atom of the molecule.

    Args:
        symbols (list[str]): symbols of the atomic species in the molecule
        coordinates (array[float]): 1D array with the atomic positions in Cartesian
            coordinates. The coordinates must be given in atomic units and the size of the array
            should be ``3*N`` where ``N`` is the number of atoms.
        name (str): name of the molecule
        charge (int): charge of the molecule
        mult (int): spin multiplicity :math:`\mathrm{mult}=N_\mathrm{unpaired} + 1` of the
            Hartree-Fock (HF) state based on the number of unpaired electrons occupying the
            HF orbitals
        basis (str): Atomic basis set used to represent the molecular orbitals. Basis set
            availability per element can be found
            `here <www.psicode.org/psi4manual/master/basissets_byelement.html#apdx-basiselement>`_
        package (str): quantum chemistry package (pyscf) used to solve the
            mean field electronic structure problem
        core (list): indices of core orbitals
        active (list): indices of active orbitals
        mapping (str): transformation (``'jordan_wigner'``, ``'parity'``, or ``'bravyi_kitaev'``) used to
            map the fermionic operator to the Pauli basis
        cutoff (float): Cutoff value for including the matrix elements
            :math:`\langle \alpha \vert \hat{{\bf r}} \vert \beta \rangle`. The matrix elements
            with absolute value less than ``cutoff`` are neglected.
        outpath (str): path to the directory containing output files
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        list[pennylane.Hamiltonian]: the qubit observables corresponding to the components
        :math:`\hat{D}_x`, :math:`\hat{D}_y` and :math:`\hat{D}_z` of the dipole operator in
        atomic units.

    **Example**

    >>> symbols = ["H", "H", "H"]
    >>> coordinates = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])
    >>> dipole_obs = dipole_of(symbols, coordinates, charge=1)
    >>> print([(h.wires) for h in dipole_obs])
    [Wires([0, 1, 2, 3, 4, 5]), Wires([0, 1, 2, 3, 4, 5]), Wires([0])]

    >>> dipole_obs[0] # x-component of D
    (
        0.4781123173263876 * Z(0)
      + 0.4781123173263876 * Z(1)
      + -0.3913638489489803 * (Y(0) @ Z(1) @ Y(2))
      + -0.3913638489489803 * (X(0) @ Z(1) @ X(2))
      + -0.3913638489489803 * (Y(1) @ Z(2) @ Y(3))
      + -0.3913638489489803 * (X(1) @ Z(2) @ X(3))
      + 0.2661114704527088 * (Y(0) @ Z(1) @ Z(2) @ Z(3) @ Y(4))
      + 0.2661114704527088 * (X(0) @ Z(1) @ Z(2) @ Z(3) @ X(4))
      + 0.2661114704527088 * (Y(1) @ Z(2) @ Z(3) @ Z(4) @ Y(5))
      + 0.2661114704527088 * (X(1) @ Z(2) @ Z(3) @ Z(4) @ X(5))
      + 0.7144779061810713 * Z(2)
      + 0.7144779061810713 * Z(3)
      + -0.11734958781031017 * (Y(2) @ Z(3) @ Y(4))
      + -0.11734958781031017 * (X(2) @ Z(3) @ X(4))
      + -0.11734958781031017 * (Y(3) @ Z(4) @ Y(5))
      + -0.11734958781031017 * (X(3) @ Z(4) @ X(5))
      + 0.24190977644645698 * Z(4)
      + 0.24190977644645698 * Z(5)
    )
    """

    if mult != 1:
        raise ValueError(
            f"Currently, this functionality is constrained to Hartree-Fock states with spin multiplicity = 1;"
            f" got multiplicity 2S+1 =  {mult}"
        )

    for i in symbols:
        if i not in atomic_numbers:
            raise ValueError(f"Requested element {i} doesn't exist")

    hf_file = meanfield(path,symbols, coordinates, name, charge, mult, basis, package, outpath)

    hf = MolecularData(path,filename=hf_file.strip())

    # Load dipole matrix elements in the atomic basis
    # pylint: disable=import-outside-toplevel
    from pyscf import gto

    mol = gto.M(
        atom=hf.geometry, basis=hf.basis, charge=hf.charge, spin=0.5 * (hf.multiplicity - 1)
    )
    dip_ao = mol.intor_symmetric("int1e_r", comp=3).real

    # Transform dipole matrix elements to the MO basis
    n_orbs = hf.n_orbitals
    c_hf = hf.canonical_orbitals

    dip_mo = np.zeros((3, n_orbs, n_orbs))
    for comp in range(3):
        for alpha in range(n_orbs):
            for beta in range(alpha + 1):
                dip_mo[comp, alpha, beta] = c_hf[:, alpha] @ dip_ao[comp] @ c_hf[:, beta]

        dip_mo[comp] += dip_mo[comp].T - 2*np.diag(np.diag(dip_mo[comp]))

    # Compute the nuclear contribution
    # dip_n = np.zeros(3)
    # for comp in range(3):
    #     for i, symb in enumerate(symbols):
    #         dip_n[comp] += atomic_numbers[symb] * coordinates[3 * i + comp]

    # Build the observable
    dip = []
    for i in range(3):
        fermion_obs = one_particle(dip_mo[i], core=core, active=active, cutoff=cutoff)
        dip.append(get_dip([-fermion_obs])) #,init_term=dip_n[i])
        # dip.append(observable([-fermion_obs], init_term=dip_n[i], mapping=mapping, wires=wires))

    return dip

def get_dip(fermion_ops,init_term=0):
    # Initialize the FermionOperator
    mb_obs = FermionOperator("") * init_term
    for ops in fermion_ops:
        if not isinstance(ops, FermionOperator):
            raise TypeError(
                f"Elements in the lists are expected to be of type 'FermionOperator'; got {type(ops)}"
            )
        mb_obs += ops

    return jordan_wigner(mb_obs)