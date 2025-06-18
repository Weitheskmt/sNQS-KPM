import numpy as np

def create_h_chain(n_atoms, bond_length=1.0):
    """
    创建一维 H 链的结构。

    参数:
    n_atoms (int): 氢原子的数量。
    bond_length (float): 相邻氢原子之间的键长，默认为 1.0。

    返回:
    numpy.ndarray: 包含氢原子坐标的数组，形状为 (n_atoms, 3)。
    """
    # 初始化坐标数组

    coordinates = np.zeros((n_atoms, 3))

    # 设置每个氢原子的坐标

    for i in range(n_atoms):
        coordinates[i, 0] = i * bond_length

    return coordinates

def get_H2_chains(n_atoms):
    # 创建包含 10 个氢原子的 H 链

    h_chain_geometry = create_h_chain(n_atoms)

    # 打印每个氢原子的坐标

    geometry_H_chain = []

    for i, pos in enumerate(h_chain_geometry):
        geometry_H_chain.append(('H',tuple(pos)))
        # print(f"Atom {i}: {pos}")
    return geometry_H_chain