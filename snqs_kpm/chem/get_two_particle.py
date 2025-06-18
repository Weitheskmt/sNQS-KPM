import numpy as np
from snqs_kpm.ops import FermionOperator

def two_particle(matrix_elements, core=None, active=None, cutoff=1.0e-12):
    r"""Generates the `FermionOperator <https://github.com/quantumlib/OpenFermion/blob/master/docs/
    tutorials/intro_to_openfermion.ipynb>`_ representing a given two-particle operator
    required to build many-body qubit observables.

    Second quantized two-particle operators are expanded in the basis of single-particle
    states as

    .. math::

        \hat{V} = \frac{1}{2} \sum_{\alpha, \beta, \gamma, \delta}
        \langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle
        ~ &[& \hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow}^\dagger
        \hat{c}_{\gamma\uparrow} \hat{c}_{\delta\uparrow} + \hat{c}_{\alpha\uparrow}^\dagger
        \hat{c}_{\beta\downarrow}^\dagger \hat{c}_{\gamma\downarrow} \hat{c}_{\delta\uparrow} \\
        &+& \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\uparrow}^\dagger
        \hat{c}_{\gamma\uparrow} \hat{c}_{\delta\downarrow} + \hat{c}_{\alpha\downarrow}^\dagger
        \hat{c}_{\beta\downarrow}^\dagger \hat{c}_{\gamma\downarrow} \hat{c}_{\delta\downarrow}~].

    In the equation above the indices :math:`\alpha, \beta, \gamma, \delta` run over the basis
    of spatial orbitals :math:`\phi_\alpha(r)`. Since the operator :math:`v` acts only on the
    spatial coordinates the spin quantum numbers are indicated explicitly with the up/down arrows.
    The operators :math:`\hat{c}^\dagger` and :math:`\hat{c}` are the particle creation and
    annihilation operators, respectively, and
    :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle` denotes the
    matrix elements of the operator :math:`\hat{v}`

    .. math::

        \langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle =
        \int dr_1 \int dr_2 ~ \phi_\alpha^*(r_1) \phi_\beta^*(r_2) ~\hat{v}(r_1, r_2)~
        \phi_\gamma(r_2) \phi_\delta(r_1).

    If an active space is defined (see :func:`~.active_space`), the summation indices
    run over the active orbitals and the contribution due to core orbitals is computed as

    .. math::

        && \hat{V}_\mathrm{core} = v_\mathrm{core} +
        \sum_{\alpha, \beta \in \mathrm{active}} \sum_{i \in \mathrm{core}}
        (2 \langle i, \alpha \vert \hat{v} \vert \beta, i \rangle -
        \langle i, \alpha \vert \hat{v} \vert i, \beta \rangle)~
        [\hat{c}_{\alpha\uparrow}^\dagger \hat{c}_{\beta\uparrow} +
        \hat{c}_{\alpha\downarrow}^\dagger \hat{c}_{\beta\downarrow}] \\
        && v_\mathrm{core} = \sum_{\alpha,\beta \in \mathrm{core}}
        [2 \langle \alpha, \beta \vert \hat{v} \vert \beta, \alpha \rangle -
        \langle \alpha, \beta \vert \hat{v} \vert \alpha, \beta \rangle].

    Args:
        matrix_elements (array[float]): 4D NumPy array with the matrix elements
            :math:`\langle \alpha, \beta \vert \hat{v} \vert \gamma, \delta \rangle`
        core (list): indices of core orbitals, i.e., the orbitals that are
            not correlated in the many-body wave function
        active (list): indices of active orbitals, i.e., the orbitals used to
            build the correlated many-body wave function
        cutoff (float): Cutoff value for including matrix elements. The
            matrix elements with absolute value less than ``cutoff`` are neglected.

    Returns:
        FermionOperator: an instance of OpenFermion's ``FermionOperator`` representing the
        two-particle operator :math:`\hat{V}`.

    **Example**

    >>> import numpy as np
    >>> matrix_elements = np.array([[[[ 6.82389533e-01, -1.45716772e-16],
    ...                               [-2.77555756e-17,  1.79000576e-01]],
    ...                              [[-2.77555756e-17,  1.79000576e-16],
    ...                               [ 6.70732778e-01, 0.00000000e+00]]],
    ...                             [[[-1.45716772e-16,  6.70732778e-16],
    ...                               [ 1.79000576e-01, -8.32667268e-17]],
    ...                              [[ 1.79000576e-16, -8.32667268e-17],
    ...                               [ 0.00000000e+00,  7.05105632e-01]]]])
    >>> v_op = two_particle(matrix_elements)
    >>> print(v_op)
    0.3411947665 [0^ 0^ 0 0] +
    0.089500288 [0^ 0^ 2 2] +
    0.3411947665 [0^ 1^ 1 0] +
    0.089500288 [0^ 1^ 3 2] +
    0.335366389 [0^ 2^ 2 0] +
    0.335366389 [0^ 3^ 3 0] +
    0.3411947665 [1^ 0^ 0 1] +
    0.089500288 [1^ 0^ 2 3] +
    0.3411947665 [1^ 1^ 1 1] +
    0.089500288 [1^ 1^ 3 3] +
    0.335366389 [1^ 2^ 2 1] +
    0.335366389 [1^ 3^ 3 1] +
    0.089500288 [2^ 0^ 2 0] +
    0.089500288 [2^ 1^ 3 0] +
    0.352552816 [2^ 2^ 2 2] +
    0.352552816 [2^ 3^ 3 2] +
    0.089500288 [3^ 0^ 2 1] +
    0.089500288 [3^ 1^ 3 1] +
    0.352552816 [3^ 2^ 2 3] +
    0.352552816 [3^ 3^ 3 3]
    """

    orbitals = matrix_elements.shape[0]

    if matrix_elements.ndim != 4:
        raise ValueError(
            f"'matrix_elements' must be a 4D array; got 'matrix_elements.ndim = ' {matrix_elements.ndim}"
        )

    if not core:
        v_core = 0
    else:
        if any(i > orbitals - 1 or i < 0 for i in core):
            raise ValueError(
                f"Indices of core orbitals must be between 0 and {orbitals - 1}; got core = {core}"
            )

        # Compute the contribution of core orbitals
        v_core = sum(
            [
                2 * matrix_elements[alpha, beta, beta, alpha]
                - matrix_elements[alpha, beta, alpha, beta]
                for alpha in core
                for beta in core
            ]
        )

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
    quads = [
        [indices[0][i], indices[1][i], indices[2][i], indices[3][i]]
        for i in range(num_indices)
        if all(indices[j][i] in active for j in range(len(indices)))
    ]

    # Build the FermionOperator representing V
    v_op = FermionOperator("") * v_core

    # add renormalized (due to core orbitals) "one-particle" operators
    if core:
        for alpha in active:
            for beta in active:
                element = 2 * np.sum(
                    matrix_elements[np.array(core), alpha, beta, np.array(core)]
                ) - np.sum(matrix_elements[np.array(core), alpha, np.array(core), beta])

                # up-up term
                a = 2 * active.index(alpha)
                b = 2 * active.index(beta)
                v_op += FermionOperator(((a, 1), (b, 0)), element)

                # down-down term
                v_op += FermionOperator(((a + 1, 1), (b + 1, 0)), element)

    # add two-particle operators
    for quad in quads:
        alpha, beta, gamma, delta = quad
        element = matrix_elements[alpha, beta, gamma, delta]

        # up-up-up-up term
        a = 2 * active.index(alpha)
        b = 2 * active.index(beta)
        g = 2 * active.index(gamma)
        d = 2 * active.index(delta)
        v_op += FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # up-down-down-up term
        b = 2 * active.index(beta) + 1
        g = 2 * active.index(gamma) + 1
        v_op += FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # down-up-up-down term
        a = 2 * active.index(alpha) + 1
        b = 2 * active.index(beta)
        g = 2 * active.index(gamma)
        d = 2 * active.index(delta) + 1
        v_op += FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

        # down-down-down-down term
        b = 2 * active.index(beta) + 1
        g = 2 * active.index(gamma) + 1
        v_op += FermionOperator(((a, 1), (b, 1), (g, 0), (d, 0)), 0.5 * element)

    return v_op