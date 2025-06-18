import numpy as np
from sympy import symbols, binomial
from snqs_kpm.nqs.utils.sparse_math import sparse_dense_mv
import time
import scipy.sparse.linalg as spla
import scipy.sparse as sp

def get_Hn_dip_psi(n,ham,dip_X_sp,psi_gs,hbar=1):
    # 0 -> n 
    o = np.zeros((n+1,len(psi_gs)),dtype=np.complex128)
    o[0] = sparse_dense_mv(dip_X_sp, psi_gs)
    if n > 0:
        for i in range(n):
            o[i+1] = sparse_dense_mv(ham, o[i])
    return o


def get_Hn_psi(n,ham,psi_gs,hbar=1):
    # 0 -> n 
    o = np.zeros((n+1,len(psi_gs)),dtype=np.complex128)
    o[0] = psi_gs
    if n > 0:
        for i in range(n):
            o[i+1] = sparse_dense_mv(ham, o[i])
    return o

def get_psi_Hn_dip_Hn_psi(Hn_psi,Hn_dip_psi,dip_X_sp,coef,norm):
    o = np.zeros(Hn_psi.shape[0],dtype=np.complex128)
    for i in range(Hn_psi.shape[0]):
        o[i] = coef[i]*(Hn_psi[Hn_psi.shape[0]-1-i].conj().dot(sparse_dense_mv(dip_X_sp, Hn_dip_psi[i])))/norm
    return o

def get_Omega_n(n,ham_sp,psi_gs,dip_X_sp,norm,SCALE=0.01,hbar=1):
    coef = np.zeros(n+1,dtype=np.complex128)
    for k in range(n + 1):
        coefficient = (-1) ** k * binomial(n, k)
        # term = f"{coefficient} * {H}^{n-k} * {mu} * {H}^{k}"
        # terms.append(term)
        coef[k]=coefficient
    coef = coef * (SCALE * 1j / hbar)**n
    Hn_psi = get_Hn_psi(n,ham_sp,psi_gs)
    Hn_dip_psi = get_Hn_dip_psi(n,ham_sp,dip_X_sp,psi_gs)
    pHddp = get_psi_Hn_dip_Hn_psi(Hn_psi,Hn_dip_psi,dip_X_sp,coef,norm)
    return np.sum(pHddp)

def prep_Hamiltonian(H,H_center,H_scale,alpha=0.4):
        if H_center is not None:
            H_tilde = H - sp.diags([H_center], shape=H.shape)
        else:
            H_tilde = H.copy()
            H_center = 0.
        if H_scale is None:
            s = time.time()
            Emax, _ = spla.eigsh(H_tilde, k=1)
            H_scale = np.abs(Emax[0])
            print('maximum eigenvalue = {:.4g}, computation time = {:.4g} [s]'.format(H_scale, time.time()-s))

        H_scale *= 1. + alpha

        return H_tilde /H_scale