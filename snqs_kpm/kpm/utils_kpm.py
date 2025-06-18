import numpy as np
from snqs_kpm.kpm import KPM
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import h5py
import pickle
import netket as nk
from snqs_kpm.chem import get_n_qubits_k_terms
import scipy.sparse.linalg as spla
import time
import math
from snqs_kpm.ops import PauliStrings
# parameters = {
#     'figure.figsize': (7,5),
#     'font.family': 'Times New Roman',
#     'mathtext.fontset': 'stix',
#     'mathtext.default': 'it',
#     'axes.labelsize': 24,
#     'axes.titlesize': 26,
#     'xtick.labelsize': 20,
#     'ytick.labelsize': 20
# }
# plt.rcParams.update(parameters)

tps = ["vmc_energy","ccsd_energy", "fci_energy"]
hartree_to_eV = 27.211386245988

def get_from_file(property_name,path):
    """Helper routine to re-open HDF5 file and pull out single property

    Args:
        property_name: Property name to load from self.filename

    Returns:
        The data located at file[property_name] for the HDF5 file at
            self.filename. Returns None if the key is not found in the
            file.
    """
    try:
        with h5py.File(path+".hdf5", "r") as f:
            data = f[property_name][...]
    except KeyError:
        data = None
    except IOError:
        data = None
    return data

def get_absorption_spectrum(ham,dip,psi0,N_random=20,H_scale=None, N_moments=4000, N_division=8000):
    kpm = KPM(ham, N_random=N_random, N_moments=N_moments, N_division=N_division, kernel='Jackson',H_scale=H_scale)
    energy = np.linspace(-kpm.H_scale*0.95,kpm.H_scale*0.95,kpm.N_division)
    dE = energy[1] - energy[0]
    dos = kpm.get_corr_DOS(dip, psi0.ravel(), energy)
    dos /= np.sum(dos*dE)
    return energy, dos


def load_data(mol_name,use_ham_sp=False,lanczos=True):
    main_path = f"./molecules/{mol_name}/{mol_name}"
    
    # # e_gs = get_from_file(tps[2], main_path)
    # HSC = np.load(main_path+"_HSC.npy")

    with open(main_path+"_qubit_hamiltonian.pkl",'rb') as f:
        qubit_ham = pickle.load(f)
    with open(main_path+"_qubit_dipole_series.pkl",'rb') as f:
        qubit_dip = pickle.load(f)


    if use_ham_sp:
        with open(main_path+"_ham_sp.pkl",'rb') as f:
            ham_sp = pickle.load(f)

        with open(main_path+"_dip_sp_XYZ.pkl",'rb') as f:
            dip_sp_XYZ = pickle.load(f)
    else:
        n_qubits, n_terms = get_n_qubits_k_terms(qubit_ham)
        hi = nk.hilbert.Fock(n_max=1, N=n_qubits)
        ham = PauliStrings.from_snqs_kpm(hi,qubit_ham)

        ham_sp = ham.to_sparse()

        dip_sp_XYZ =[]
        for i in range(len(qubit_dip)):
            dip_sp_XYZ.append(PauliStrings.from_snqs_kpm(hi,qubit_dip[i]).to_sparse())

    if lanczos:
        e_gs, psi0 = spla.eigsh(ham_sp,k=1,which="SA")
        return ham_sp, dip_sp_XYZ, psi0, e_gs
    else:
        return ham_sp, dip_sp_XYZ
    
    # with open(main_path+f"_moments_XYZ.pkl", 'rb') as f:
    #     moments_XYZ = pickle.load(f)

def KPM_AS(mol_name,XLIM=50, PLOT=False, SAVE=True):
    ham_sp, dip_sp_XYZ, psi0, e_gs = load_data(mol_name)
    N_moments = 400*math.ceil(np.abs(e_gs))
    N_division = 2*N_moments

    energy = []
    AS = []
    for i in range(len(dip_sp_XYZ)):
        energy_dum,AS_dum = get_absorption_spectrum(ham_sp, dip_sp_XYZ[i], psi0, H_scale=None, N_random=1, N_moments=N_moments, N_division=N_division)
        energy.append(energy_dum)
        AS.append(AS_dum)
    
    x_energy = (energy[0]-e_gs)*hartree_to_eV
    dum = 0
    for i in range(len(AS)):
        dum+=np.abs(AS[i])
    y_intensity = dum/np.max(dum)

    if SAVE:
        np.save(f"./molecules/{mol_name}/{mol_name}_energy.npy", x_energy)
        np.save(f"./molecules/{mol_name}/{mol_name}_intensity.npy", y_intensity)

    return x_energy, y_intensity


    



# def get_spe_exact(mol_name,ham_sp,dip_sp,psi0,e_gs,qubit_dip, XLIM = 3*hartree_to_eV, PLOT=False, SAVE=True):
#     epsilon = []
#     intensity = []
#     for i in range(len(qubit_dip)):
#         e, i = get_absorption_spectrum(ham_sp,dip_sp,psi0)
#         epsilon.append(e)
#         intensity.append(i)
    
#     x_energy = (epsilon[0]-e_gs)*hartree_to_eV
#     dum = 0
#     for i in range(len(intensity)):
#         dum+=np.abs(intensity[i])
#     y_intensity = dum/np.max(dum)

#     if PLOT:
#         plt.figure(dpi=350)
#         plt.plot(x_energy, y_intensity,'-',c='k',lw=2)
#         plt.xlim(0,XLIM)
#         plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=10, width=1, labelsize=20)
#         plt.xlabel(r"Energy[eV]")
#         plt.ylabel(r"Intensity")
#         plt.title(f"{mol_name}")
    
#     if SAVE:
#         np.save(f"../molecules/{mol_name}/{mol_name}_energy_exact.npy", x_energy)
#         np.save(f"../molecules/{mol_name}/{mol_name}_instity_exact.npy", y_intensity)

#     return x_energy, y_intensity


def get_e_dos_input_moments(ham,moments,HSC_value,N_moments=2000, N_division=8000):

    kpm = KPM(ham, N_random=20, N_moments=N_moments, N_division=N_division, kernel='Jackson', H_scale=HSC_value)
    energy = np.linspace(-kpm.H_scale*0.95,kpm.H_scale*0.95,kpm.N_division)
    dE = energy[1] - energy[0]
    dos = kpm.get_corr_DOS_input_moments(moments, energy)
    dos /= np.sum(dos*dE)
    return energy, dos


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def multi_gaussian(x, peaks, sigma):
    return sum(gaussian(x, A, mu, sigma) for mu, A in peaks)

def gaussian_delta(x, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x / sigma) ** 2)

# 定义主函数

def spectral_density(e_input, e_gs, psi_gs, dip_X, sigma):
    """
    计算光谱密度函数 2π ∑_n ⟨n | μ_x | o⟩^2 δ(ω_n - ω_o)
    
    参数:
    omega_o (float): 基态频率

    matrix_elements (list of float): 矩阵元 ⟨n | μ_x | o⟩

    omega_ns (list of float): 激发态频率

    sigma (float): 高斯函数的标准差，用于近似δ函数

    返回:
    float: 光谱密度

    """
    idx = np.argmin(e_gs)
    spectral_density = 0.0

    for n in range(e_gs.shape[0]):
        if n != idx:
            spectral_density += np.abs(psi_gs[:,n].conj().T@(dip_X@psi_gs[:,idx]))**2 * gaussian_delta(e_gs[n]-e_gs[idx] - e_input, sigma)
    return 2 * np.pi * spectral_density

def FCI_AS(mol_name,k=1001,XLIM=50,PLOT=False,SAVE=True):
    ham_sp, dip_sp_XYZ = load_data(mol_name,lanczos=False)
    e_gs_s, psi_gs_s = spla.eigsh(ham_sp,k=k,which="SA")
    np.save(f"./molecules/{mol_name}/{mol_name}_e_g_excited.npy",e_gs_s)
    omega = np.linspace(0,50,1000)/hartree_to_eV
    x_energy = omega*hartree_to_eV
    intensity_1 = spectral_density(omega, e_gs_s, psi_gs_s, dip_sp_XYZ[0],sigma=0.005)
    intensity_2 = spectral_density(omega, e_gs_s, psi_gs_s, dip_sp_XYZ[1],sigma=0.005)
    intensity_3 = spectral_density(omega, e_gs_s, psi_gs_s, dip_sp_XYZ[2],sigma=0.005)
    intensity=np.abs(intensity_1)+np.abs(intensity_2)+np.abs(intensity_3)
    if PLOT:
        plt.plot(x_energy,intensity_1.reshape(-1),label="X")
        plt.plot(x_energy,intensity_2.reshape(-1),label="Y")
        plt.plot(x_energy,intensity_3.reshape(-1),label="Z")
        plt.plot(x_energy,intensity.reshape(-1),label="tot.")
        plt.xlim(0,XLIM)
    intensity = intensity/np.max(intensity)
    if SAVE:
        np.save(f"./molecules/{mol_name}/{mol_name}_energy_fci.npy", x_energy)
        np.save(f"./molecules/{mol_name}/{mol_name}_intensity_fci.npy", intensity)
    return x_energy, intensity

def load_data_traditional(mol_name, N_moments=2000, N_division=8000,use_ham_sp=True):
    main_path = f"./molecules/{mol_name}/{mol_name}"
    e_gs = get_from_file(tps[2], main_path)
    HSC = np.load(main_path+"_HSC.npy")

    with open(main_path+"_qubit_hamiltonian.pkl",'rb') as f:
        qubit_ham = pickle.load(f)
    with open(main_path+"_qubit_dipole_series.pkl",'rb') as f:
        qubit_dip = pickle.load(f)

    if use_ham_sp:
        with open(main_path+"_ham_sp.pkl",'rb') as f:
            ham_sp = pickle.load(f)

        with open(main_path+"_dip_sp_XYZ.pkl",'rb') as f:
            dip_sp_XYZ = pickle.load(f)
    else:
        n_qubits, n_terms = get_n_qubits_k_terms(qubit_ham)
        hi = nk.hilbert.Fock(n_max=1, N=n_qubits)
        ham = PauliStrings.from_snqs_kpm(hi,qubit_ham)

        ham_sp = ham.to_sparse()

        dip_sp_XYZ =[]
        for i in range(len(qubit_dip)):
            dip_sp_XYZ.append(PauliStrings.from_snqs_kpm(hi,qubit_dip[i]))
    
    with open(main_path+f"_moments_XYZ.pkl", 'rb') as f:
        moments_XYZ = pickle.load(f)

    return ham_sp, dip_sp_XYZ, moments_XYZ, e_gs, HSC



    

def get_spe_exact(mol_name,e_gs,energy,dos,XLIM=50,sigma_initial=0.1,height=0.001,distance=5,PLOT_exact=False,PLOT_kpm=False,SAVE_exact=True,SAVE_kpm=True):
    epsilon = []
    intensity = []
    for i in range(len(energy)):
        epsilon.append(energy[i])
        intensity.append(dos[i])

    hartree_to_eV = 27.211386245988

    x_energy = (epsilon[0]-e_gs)*hartree_to_eV
    dum = 0
    for i in range(len(intensity)):
        dum+=np.abs(intensity[i])
    y_intensity = dum/np.max(dum)

    if PLOT_exact:
        plt.figure(dpi=350)
        plt.plot(x_energy, y_intensity,'-',c='k',lw=2)
        plt.xlim(0,XLIM)
        # plt.xlim(0,10)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=10, width=1, labelsize=20)
        plt.xlabel(r"Energy[eV]")
        plt.ylabel(r"Intensity")
        plt.title(f"{mol_name}")
        plt.tight_layout()
        plt.savefig(f"./molecules/{mol_name}/{mol_name}_absorption_spectrum_exact.pdf", format='pdf')
        plt.savefig(f"../molecules/{mol_name}/{mol_name}_absorption_spectrum_exact.png", format='png')

    if SAVE_exact:
        np.save(f"./molecules/{mol_name}/{mol_name}_energy_exact.npy", x_energy)
        np.save(f"./molecules/{mol_name}/{mol_name}_intensity_exact.npy", y_intensity)


    peaks, _ = find_peaks(y_intensity, height=height, distance=distance)

    peak_x_dum = x_energy[peaks]
    peak_y_dum = y_intensity[peaks]
    indices = np.where(peak_x_dum > 0.5)
    peak_x = peak_x_dum[indices]
    peak_y = peak_y_dum[indices]

    peaks = np.array([peak_x,peak_y]).T
    x_range = np.linspace(0, XLIM, 1000)
    y_smoothed = multi_gaussian(x_range, peaks, sigma_initial)

    if PLOT_kpm:
        plt.figure(dpi=350)
        plt.plot(x_range, y_smoothed/np.max(y_smoothed),'-',c='k',lw=2)
        plt.xlim(0,XLIM)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=10, width=1, labelsize=20)
        plt.xlabel(r"Energy[eV]")
        plt.ylabel(r"Intensity")
        plt.title(f"{mol_name}")
        plt.tight_layout()
        plt.savefig(f"./molecules/{mol_name}/{mol_name}_absorption_spectrum_kpm.pdf", format='pdf')
        plt.savefig(f"./molecules/{mol_name}/{mol_name}_absorption_spectrum_kpm.png", format='png')


    if SAVE_kpm:
        np.save(f"./molecules/{mol_name}/{mol_name}_energy_kpm.npy", x_range)
        np.save(f"./molecules/{mol_name}/{mol_name}_intensity_kpm.npy", y_smoothed/np.max(y_smoothed))


def get_spe(mol_name, ham_sp, moments_XYZ,e_gs,HSC,N_moments, N_division=8000, XLIM=3*27.211386245988, height=0.001, PLOT=False, SAVE=False):
    N_division = 2*N_moments
    epsilon = []
    intensity = []
    HSC_value1 = HSC
    HSC_value2 = HSC
    HSC_value3 = HSC
    # for i in range(len(qubit_dip)):
    #     e, i = get_e_dos_input_moments(ham_sp,moments_XYZ[i],HSC_value,N_moments,N_division)
    #     epsilon.append(e)
    #     intensity.append(i)

    e, i = get_e_dos_input_moments(ham_sp,moments_XYZ[0],HSC_value1,N_moments,N_division)
    epsilon.append(e)
    intensity.append(i)

    e, i = get_e_dos_input_moments(ham_sp,moments_XYZ[1],HSC_value2,N_moments,N_division)
    epsilon.append(e)
    intensity.append(i)

    e, i = get_e_dos_input_moments(ham_sp,moments_XYZ[2],HSC_value3,N_moments,N_division)
    epsilon.append(e)
    intensity.append(i)

    x_energy = (epsilon[0]-e_gs)*27.211386245988
    dum = 0
    for i in range(len(intensity)):
        dum+=np.abs(intensity[i])
    y_intensity = dum/np.max(dum)

    peaks, _ = find_peaks(y_intensity, height=height, distance=5)

    peak_x_dum = x_energy[peaks]
    peak_y_dum = y_intensity[peaks]
    indices = np.where(peak_x_dum > 0.5)
    peak_x = peak_x_dum[indices]
    peak_y = peak_y_dum[indices]

    peaks = np.array([peak_x,peak_y]).T
    sigma_initial = 1 
    PS = 0
    x_range = np.linspace(0, XLIM, 1000)

    # 计算 y 值
    y_smoothed = multi_gaussian(x_range, peaks[PS:PS+15], sigma_initial)

    if PLOT:
        plt.figure(dpi=350)
        plt.plot(x_range, y_smoothed/np.max(y_smoothed),'-',c='k',lw=2)
        plt.xlim(0,XLIM)
        plt.tick_params(axis='both', which='both', direction='in', top=True, right=True, length=10, width=1, labelsize=20)
        plt.xlabel(r"Energy[eV]")
        plt.ylabel(r"Intensity")
        plt.title(f"{mol_name}")
        plt.tight_layout()
        plt.savefig(f"./molecules/{mol_name}/{mol_name}_spe.pdf", format='pdf')
        plt.savefig(f"./molecules/{mol_name}/{mol_name}_spe.png", format='png')
    
    if SAVE:
        np.save(f"./molecules/{mol_name}/{mol_name}_eps_nqs.npy", x_range)
        np.save(f"./molecules/{mol_name}/{mol_name}_inst_nqs.npy", y_smoothed/np.max(y_smoothed))

    return x_range, y_smoothed/np.max(y_smoothed)