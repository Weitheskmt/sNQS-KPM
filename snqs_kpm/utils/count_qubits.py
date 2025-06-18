from snqs_kpm.ops import (
    FermionOperator,
    QubitOperator,
    MajoranaOperator,
    DiagonalCoulombHamiltonian,
    PolynomialTensor,
    IsingOperator
)

import numpy

def count_qubits(operator):
    """Compute the minimum number of qubits on which operator acts.

    Args:
        operator: FermionOperator, QubitOperator, DiagonalCoulombHamiltonian,
            or PolynomialTensor.

    Returns:
        num_qubits (int): The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.
    """
    # Handle FermionOperator.
    if isinstance(operator, FermionOperator):
        num_qubits = 0
        for term in operator.terms:
            for ladder_operator in term:
                if ladder_operator[0] + 1 > num_qubits:
                    num_qubits = ladder_operator[0] + 1
        return num_qubits

    # Handle QubitOperator.
    elif isinstance(operator, QubitOperator):
        num_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > num_qubits:
                    num_qubits = term[-1][0] + 1
        return num_qubits

    # Handle MajoranaOperator.
    if isinstance(operator, MajoranaOperator):
        num_qubits = 0
        for term in operator.terms:
            for majorana_index in term:
                if numpy.ceil((majorana_index + 1) / 2) > num_qubits:
                    num_qubits = int(numpy.ceil((majorana_index + 1) / 2))
        return num_qubits

    # Handle DiagonalCoulombHamiltonian
    elif isinstance(operator, DiagonalCoulombHamiltonian):
        return operator.one_body.shape[0]

    # Handle PolynomialTensor
    elif isinstance(operator, PolynomialTensor):
        return operator.n_qubits

    # Handle IsingOperator
    elif isinstance(operator, IsingOperator):
        num_qubits = 0
        for term in operator.terms:
            if term:
                if term[-1][0] + 1 > num_qubits:
                    num_qubits = term[-1][0] + 1
        return num_qubits

    # Raise for other classes.
    else:
        raise TypeError('Operator of invalid type.')