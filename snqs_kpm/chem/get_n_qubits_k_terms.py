def get_n_qubits_k_terms(qubit_hamiltonian):
    n_wires = (
    1 + max(max(i for i, _ in t) if t else 1 for t in qubit_hamiltonian.terms)
    if qubit_hamiltonian.terms
    else 1)

    k_terms = len(qubit_hamiltonian.terms.items())

    return n_wires, k_terms #n-qubits,n-terms