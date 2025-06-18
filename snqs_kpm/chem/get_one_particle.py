import numpy as np
from snqs_kpm.ops import FermionOperator

def one_particle(matrix_elements, core=None, active=None, cutoff=1.0e-12):
    r"""Generates the `FermionOperator <https://github.com/quantumlib/OpenFermion/blob/master/docs/
    tutorials/intro_to_openfermion.ipynb>`_ representing a given one-particle operator
    required to build many-body qubit observables.

    Second quantized one-particle operators are expanded in the basis of single-particle
    states as

    .. math::

        \hat{T} = \sum_{\alpha, \beta} \langle \alpha \vert \hat{t} \vert \beta \rangle
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}].

    In the equation above the indices :math:`\alpha, \beta` run over the basis of spatial
    orbitals :math:`\phi_\alpha(r)`. Since the operator :math:`\hat{t}` acts only on the
    spatial coordinates, the spin quantum numbers are indicated explicitly with the up/down arrows.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation
    and annihilation operators, respectively, and
    :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle` denotes the matrix elements of
    the operator :math:`\hat{t}`

    .. math::

        \langle \alpha \vert \hat{t} \vert \beta \rangle = \int dr ~ \phi_\alpha^*(r)
        \hat{t}(r) \phi_\beta(r).

    If an active space is defined (see :func:`~.active_space`), the summation indices
    run over the active orbitals and the contribution due to core orbitals is computed as
    :math:`t_\mathrm{core} = 2 \sum_{\alpha\in \mathrm{core}}
    \langle \alpha \vert \hat{t} \vert \alpha \rangle`.

    Args:
        matrix_elements (array[float]): 2D NumPy array with the matrix elements
            :math:`\langle \alpha \vert \hat{t} \vert \beta \rangle`
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function
        cutoff (float): Cutoff value for including matrix elements. The
            matrix elements with absolute value less than ``cutoff`` are neglected.

    Returns:
        FermionOperator: an instance of OpenFermion's ``FermionOperator`` representing the
        one-particle operator :math:`\hat{T}`.

    **Example**

    >>> import numpy as np
    >>> matrix_elements = np.array([[-1.27785301e+00,  0.00000000e+00],
    ...                             [ 1.52655666e-16, -4.48299696e-01]])
    >>> t_op = one_particle(matrix_elements)
    >>> print(t_op)
    -1.277853006156875 [0^ 0] +
    -1.277853006156875 [1^ 1] +
    -0.44829969610163756 [2^ 2] +
    -0.44829969610163756 [3^ 3]
    """

    orbitals = matrix_elements.shape[0]

    if matrix_elements.ndim != 2:
        raise ValueError(
            f"'matrix_elements' must be a 2D array; got matrix_elements.ndim = {matrix_elements.ndim}"
        )

    if not core:
        t_core = 0
    else:
        if any(i > orbitals - 1 or i < 0 for i in core):
            raise ValueError(
                f"Indices of core orbitals must be between 0 and {orbitals - 1}; got core = {core}"
            )

        # Compute contribution due to core orbitals
        t_core = 2 * np.sum(matrix_elements[np.array(core), np.array(core)])

    if active is None:
        if not core:
            active = list(range(orbitals))
        else:
            active = [i for i in range(orbitals) if i not in core]

    if any(i > orbitals - 1 or i < 0 for i in active):
        raise ValueError(
            f"Indices of active orbitals must be between 0 and {orbitals - 1}; got active = {active}"
        )

    # Indices of the matrix elements with absolute values >= cutoff
    indices = np.nonzero(np.abs(matrix_elements) >= cutoff)

    # Single out the indices of active orbitals
    num_indices = len(indices[0])
    pairs = [
        [indices[0][i], indices[1][i]]
        for i in range(num_indices)
        if all(indices[j][i] in active for j in range(len(indices)))
    ]

    # Build the FermionOperator representing T
    t_op = FermionOperator("") * t_core
    for pair in pairs:
        alpha, beta = pair
        element = matrix_elements[alpha, beta]

        # spin-up term
        a = 2 * active.index(alpha)
        b = 2 * active.index(beta)
        t_op += FermionOperator(((a, 1), (b, 0)), element)

        # spin-down term
        t_op += FermionOperator(((a + 1, 1), (b + 1, 0)), element)

    return t_op