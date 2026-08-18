import numpy as np
from .function_set import FunctionSet
from molsym.molecule import global_tol

class CartesianCoordinates(FunctionSet):
    """
    FunctionSet for Cartesian coordinates
    """
    def __init__(self, symtext) -> None:
        # xyz on each atom in molecule
        # fxn map is ncart (natom sets of xyz) by nsymel
        fxn_list = [i for i in range(3*len(symtext.mol))]
        super().__init__(symtext, fxn_list)
        # get_symmetry_equiv_functions() (called above via FunctionSet.__init__)
        # assumes symel.rrep is sparse, which only holds in the standard
        # orientation -- see get_symmetry_equiv_functions_nonstandard for why
        # that assumption silently undercounts SALCs once symel.rrep is dense
        # (e.g. symtext.is_nonstandard).
        if getattr(symtext, "is_nonstandard", False):
            self.SE_fxns = self.get_symmetry_equiv_functions_nonstandard()

    def get_fxn_map(self):
        """
        Builds the function map for all of the Cartesian coordinates under each symmetry element.
        
        :rtype: NumPy array of shape (nsymels, 3, 3)
        """
        # Symel by xyz by xyz, maps xyz to xyz under symels
        fxn_map = np.zeros((len(self.symtext), 3, 3))
        #phase_map = None
        for s in range(len(self.symtext)):
            fxn_map[s, :, :] = self.symtext.symels[s].rrep.T
        return fxn_map#, phase_map
 
    def get_symmetry_equiv_functions(self):
        """
        Finds the sets of functions that are invariant under all of the symmetry elements.

        :rtype: List[List[int]]
        """
        symm_equiv = []
        done = []
        xyz = np.array([0,1,2], dtype=int)
        for atom_i in range(len(self.symtext.mol)):
            for xyz_idx in range(3):
                fidx = 3*atom_i + xyz_idx
                if fidx in done:
                    continue
                equiv_set = []
                for sidx in range(len(self.symtext)):
                    newatom = self.symtext.atom_map[atom_i, sidx]
                    notzero = xyz[~np.isclose(self.symtext.symels[sidx].rrep[:,xyz_idx], 0.0, atol=1e-10)]
                    r = [3*newatom+i for i in notzero]
                    equiv_set += r
                reduced_equiv_set = list(set(equiv_set))
                symm_equiv.append(reduced_equiv_set)
                done += reduced_equiv_set
        return symm_equiv

    def get_symmetry_equiv_functions_nonstandard(self):
        """
        Finds the sets of functions that are invariant under all of the symmetry elements.

        :rtype: List[List[int]]
        """
        # get_symmetry_equiv_functions above assumes one seed (ProjectionOp
        # uses min() of the set) always spans the set it returns for a given
        # coordinate. That's only true when symel.rrep is sparse enough that
        # "touched by an operation" already matches "reachable from one
        # seed" -- true in the standard orientation but not once symel.rrep
        # is dense (e.g. a rotated/nonstandard frame), where a merged set
        # may need more than one seed to actually span it. This verifies by
        # rank instead of assuming, and expands the frontier from every
        # newly touched coordinate (not just the original one) first,
        # since two coordinates can be genuinely different orbits that
        # still overlap in support through a third coordinate.
        symm_equiv = []
        done = []
        xyz = np.array([0,1,2], dtype=int)

        def touched_by_one_hop(atom_i, xyz_idx):
            equiv_set = []
            for sidx in range(len(self.symtext)):
                newatom = self.symtext.atom_map[atom_i, sidx]
                notzero = xyz[~np.isclose(self.symtext.symels[sidx].rrep[:,xyz_idx], 0.0, atol=1e-10)]
                equiv_set += [3*newatom+i for i in notzero]
            return equiv_set

        for atom_i in range(len(self.symtext.mol)):
            for xyz_idx in range(3):
                fidx = 3*atom_i + xyz_idx
                if fidx in done:
                    continue

                component = set([fidx])
                frontier = [fidx]
                while frontier:
                    coord = frontier.pop()
                    c_atom_i, c_xyz_idx = divmod(coord, 3)
                    for idx in touched_by_one_hop(c_atom_i, c_xyz_idx):
                        if idx not in component:
                            component.add(idx)
                            frontier.append(idx)
                component = sorted(component)
                component_pos = {idx: pos for pos, idx in enumerate(component)}
                current_basis = np.zeros((len(component), 0))
                for seed in component:
                    s_atom_i, s_xyz_idx = divmod(seed, 3)
                    orbit_vectors = []
                    for sidx in range(len(self.symtext)):
                        newatom = self.symtext.atom_map[s_atom_i, sidx]
                        coeffs = self.fxn_map[sidx, s_xyz_idx, :]
                        v = np.zeros(len(component))
                        for i, c in enumerate(coeffs):
                            out_idx = 3 * newatom + i
                            if out_idx in component_pos:
                                v[component_pos[out_idx]] = c
                        orbit_vectors.append(v)
                    orbit_matrix = np.column_stack(orbit_vectors)

                    if current_basis.shape[1] == 0:
                        old_rank = 0
                        trial = orbit_matrix
                    else:
                        old_rank = np.linalg.matrix_rank(current_basis, tol=global_tol)
                        trial = np.column_stack((current_basis, orbit_matrix))

                    new_rank = np.linalg.matrix_rank(trial, tol=global_tol)
                    if new_rank > old_rank:
                        symm_equiv.append([seed])
                        current_basis = trial
                    if new_rank == len(component):
                        break
                done += component
        return symm_equiv

    def special_function(self, salc, coord, sidx, irrmat):
        """
        Defines how to map an internal coordinate under a symmetry operation for the ProjectionOp function.
        
        :type salc: NumPy array of shape (number of internal coordinates,)
        :type coord: int
        :type sidx: int
        :type irrmat: NumPy array of shape (nsymel, irrep.d, irrep.d)
        """
        atom_idx = self.symtext.atom_map[coord//3, sidx]
        cfxn = coord % 3
        xyz = self.fxn_map[sidx,cfxn,:]
        for i in range(3):
            if self.symtext.complex:
                salc[:,:,3*atom_idx+i] += np.conj(irrmat[sidx, :, :]) * xyz[i]
            else:
                salc[:,:,3*atom_idx+i] += irrmat[sidx, :, :] * xyz[i]
        return salc
