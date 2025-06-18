#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""
sNQS-KPM
"""
from snqs_kpm.chem import (
    atomic_numbers,
    STO3G,
    POPLE631G,
    POPLE6311G,
    CCPVDZ,
    basis_sets,
    load_basisset,
    run_pyscf,
    run_psi4,
    PyscfMolecularData,
    meanfield,
    one_particle,
    two_particle,
    MolecularData,
    active_space,
    dipole_moments,
    get_H2_chains,
    get_n_qubits_k_terms,
    geometry_from_pubchem,
    generate_mol_data,
    get_geometry,
    MOLECULE_LIST,
    H2_CHAINS_LIST
)

from snqs_kpm.ops import (
    FermionOperator,
    BosonOperator,
    QuadOperator,
    QubitOperator,
    MajoranaOperator,
    SymbolicOperator,
    IsingOperator,
    PolynomialTensor, 
    PolynomialTensorError, 
    general_basis_change,
    DiagonalCoulombHamiltonian,
    InteractionOperator,
    InteractionOperatorError,
    get_tensors_from_integrals,
    get_active_space_integrals,
    InteractionRDM, 
    InteractionRDMError,
    QuadraticHamiltonian,
    QuadraticHamiltonianError,
    antisymmetric_canonical_form,
    DOCIHamiltonian,
    PauliStrings,
    PauliStringsBase,
    number_operator,
    s_plus_operator,
    s_minus_operator,
    sx_operator,
    sy_operator,
    sz_operator,
    s_squared_operator
)

from snqs_kpm.transforms import (
    jordan_wigner,
    get_fermion_operator
)

from snqs_kpm.nqs import (
    run,
    run_moments
)

from snqs_kpm.kpm import (
    FCI_AS,
    KPM_AS,
    get_spe
)

from snqs_kpm.utils import (
    count_qubits,
    up_index, 
    down_index, 
    up_then_down
)
