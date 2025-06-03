from copy import deepcopy
import numpy as np
from molsym.salcs.function_set import FunctionSet

class IC():
    def __init__(self, atom_list):
        if len(atom_list) != self.num_centers:
            raise ValueError(f"Incorrect number of centers ({len(atom_list)}) for internal coordinate type {self.__class__}")
        self.atom_list = atom_list
        self.phase_on_permute = 1
        self.phase_on_inversion = 1
        self.exchange_atoms = None
        self.perm_symmetry = None

    def is_equiv(self, ic):
        if isinstance(ic, self.__class__):
            if self.atom_list == ic.atom_list:
                return True, 1
            elif self.perm_symmetry(self.atom_list) == ic.atom_list:
                return True, self.phase_on_permute
        return False, 1

    def __eq__(self, value):
        if isinstance(value, self.__class__):
            if self.atom_list == value.atom_list:
                return True
        return False

    def __repr__(self):
        cls_name = str(self.__class__).split(".")[-1][:-2]
        return f"{cls_name}" + " [" + ",".join([f"{a}" for a in self.atom_list]) + "]"

    def reversal(self, atom_list):
        return list(reversed(atom_list))

    def exchange(self, atom_list):
        return [atom_list[i] for i in self.exchange_atoms]

class Stretch(IC):
    def __init__(self, atom_list):
        self.num_centers = 2
        super().__init__(atom_list)
        self.perm_symmetry = self.reversal

class Bend(IC):
    def __init__(self, atom_list):
        self.num_centers = 3
        super().__init__(atom_list)
        self.perm_symmetry = self.reversal

class Torsion(IC):
    def __init__(self, atom_list):
        self.num_centers = 4
        super().__init__(atom_list)
        self.phase_on_inversion = -1
        self.perm_symmetry = self.reversal

class OutOfPlane(IC):
    def __init__(self, atom_list):
        self.num_centers = 4
        super().__init__(atom_list)
        self.phase_on_permute = -1
        self.phase_on_inversion = -1
        self.exchange_atoms = [0,1,3,2]
        self.perm_symmetry = self.exchange

class Linear(IC):
    def __init__(self, atom_list):
        self.num_centers = 4
        super().__init__(atom_list)
        self.phase_on_permute = -1
        self.exchange_atoms = [2,1,0,3]
        self.perm_symmetry = self.exchange

class LinX(IC):
    def __init__(self, atom_list):
        self.num_centers = 4
        super().__init__(atom_list)

class LinY(IC):
    def __init__(self, atom_list):
        self.num_centers = 4
        super().__init__(atom_list)

def user_to_IC(ic_list):
    atom_list = ic_list[0]
    name = ic_list[1]
    if name[0] == "R":
        return Stretch(atom_list)
    elif name[0] == "A":
        return Bend(atom_list)
    elif name[0] == "D":
        return Torsion(atom_list)
    elif name[0] == "O":
        return OutOfPlane(atom_list)
    elif name[0:3] == "Lin":
        return Linear(atom_list)
    elif name[0:2] == "Lx":
        return LinX(atom_list)
    elif name[0:2] == "Ly":
        return LinY(atom_list)
    else:
        raise ValueError(f"Unrecogonized internal coordinate name {name}")

class InternalCoordinates(FunctionSet):
    def __init__(self, symtext, fxn_list):
        self.ic_list = [user_to_IC(i) for i in fxn_list]
        super().__init__(symtext, fxn_list)

    def find_equiv_ic(self, ic):
        for idx, ic_i in enumerate(self.ic_list):
            ic_equiv = ic.is_equiv(ic_i)
            if ic_equiv[0]:
                return idx, ic_equiv[1]

    def operate_on_ic(self, ic_idx, sidx):
        """
        Maps an internal coordinate to a new internal coordinate under a symmetry operation.
        The phase can be 1 or -1 depending on the effect of the symmetry element.
        :type ic_idx: int
        :type symop: molsym.Symel
        :return: New internal coordinate index and phase
        :rtype: int, float
        """
        mapped_ic_atom_list = [self.symtext.atom_map[a, sidx] for a in self.ic_list[ic_idx].atom_list]
        mapped_ic = deepcopy(self.ic_list[ic_idx])
        mapped_ic.atom_list = mapped_ic_atom_list
        index, phase = self.find_equiv_ic(mapped_ic)
        symbol = self.symtext.symels[sidx].symbol
        if symbol[0] == "i" or symbol[0] == "S" or symbol[0] == "s": # s is for sigma
            phase *= mapped_ic.phase_on_inversion
        return index, phase

    def get_fxn_map(self):
        """
        Builds the function map for all of the internal coordinates under each symmetry element.

        :rtype: NumPy array of shape (number of internal coordinates, nsymels)
        """
        ic_map = np.zeros((len(self.ic_list), len(self.symtext)), dtype=np.int32)
        S = (len(self.ic_list), len(self.symtext))
        self.phase_map = np.ones(S)
        for ic_idx in range(len(self)):
            for sidx, symel in enumerate(self.symtext.symels):
                index, phase = self.operate_on_ic(ic_idx, sidx)
                ic_map[ic_idx, sidx] = index
                self.phase_map[ic_idx, sidx] *= phase
        return ic_map

    def get_symmetry_equiv_functions(self):
        """
        Finds the sets of functions that are invariant under all of the symmetry elements.

        :rtype: List[List[int]]
        """
        SEICs = []
        done = []
        for ic_idx in range(len(self.ic_list)):
            if ic_idx in done:
                continue
            else:
                seics = []
                for symel_idx in range(len(self.symtext)):
                    seics.append(self.fxn_map[ic_idx, symel_idx])
                reduced_seics = list(set(seics))
                done += reduced_seics
                SEICs.append(reduced_seics)
        
        return SEICs

    def span(self, SE_fxn):
        chorker_loaf = np.zeros(self.fxn_map.shape)
        for i in range(self.fxn_map.shape[0]):
            for j in range(self.fxn_map.shape[1]):
                if self.fxn_map[i,j] == i:
                    chorker_loaf[i,j] = 1
        rhorker_loaf = chorker_loaf[SE_fxn, :]
        shorker_loaf = np.sum(rhorker_loaf, axis=0)
        rshorker_loaf = np.zeros(self.symtext.chartable.characters.shape[1])
        for i in range(len(self.symtext)):
            rshorker_loaf[self.symtext.class_map[i]] = shorker_loaf[i]
        span = np.zeros(len(self.symtext.chartable.irreps), dtype=np.int32)
        for idx, irrep in enumerate(self.symtext.chartable.irreps):
            n = round(np.sum(rshorker_loaf * self.symtext.chartable.class_orders * self.symtext.chartable.characters[idx,:]) / self.symtext.order)
            span[idx] = n
        return span

    def special_function(self, salc, coord, sidx, irrmat):
        """
        Defines how to map an internal coordinate under a symmetry operation for the ProjectionOp function.
        
        :type salc: NumPy array of shape (number of internal coordinates,)
        :type coord: int
        :type sidx: int
        :type irrmat: NumPy array of shape (nsymel, irrep.d, irrep.d)
        """
        ic2 = self.fxn_map[coord, sidx]
        p = self.phase_map[coord, sidx]
        salc[:,:,ic2] += (irrmat[sidx, :, :]) * p
        return salc

