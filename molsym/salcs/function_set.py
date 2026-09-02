import numpy as np
from abc import ABC, abstractmethod
from molsym.molecule import global_tol

class FunctionSet(ABC):
    """
    Base class for sets of functions to be symmetrized.
    """
    def __init__(self, symtext, fxn_list) -> None:
        # fxn_list: List of coordinate objects. Concrete classes will deal with operations on the coordinates
        self.fxns = fxn_list
        self.symtext = symtext
        self.fxn_map = self.get_fxn_map()
        self.SE_fxns = self.get_symmetry_equiv_functions()

    def __len__(self):
        return len(self.fxns)

    @abstractmethod
    def get_fxn_map(self):
        pass

    @abstractmethod
    def get_symmetry_equiv_functions(self):
        pass

    @abstractmethod
    def special_function(self, salc, coord, sidx, irrmat):
        pass

# FunctionSet whose symmetry action is fully described by a single dense
# fxn_map[sidx, input_idx, output_idx] covering every function in self.fxns,
# as opposed to one distributed across atoms/shells (e.g. CartesianCoordinates,
# SphericalHarmonics). Concrete subclasses only need to implement get_fxn_map().
class DenseFunctionSet(FunctionSet):
    """
    FunctionSet with a dense fxn_map[sidx, input_idx, output_idx].
    """

    def get_symmetry_equiv_functions(self):
        """
        Finds seed functions whose group orbits span every symmetry-invariant
        mixing component of the basis.

        :rtype: list[list[int]]
        """
        # A connected component (functions reachable from one another under
        # the group action) is not always a single orbit: one seed's own
        # orbit can fail to span its whole component (e.g. a non-cyclic block
        # like {x^3, x^2y, xy^2, y^3} in C3v, where x^3 alone cannot reach the
        # A2 combination). So walk the component and only keep a new seed
        # when it actually increases the spanned rank, so the returned seeds
        # fully cover every component.
        nfxn = len(self.fxns)
        done = set()
        seed_sets = []

        for start in range(nfxn):
            if start in done:
                continue

            component = set([start])
            frontier = [start]

            while frontier:
                coord = frontier.pop()

                for sidx in range(len(self.symtext)):
                    coeffs = self.fxn_map[sidx, coord, :]
                    mapped = [i for i, coeff in enumerate(coeffs) if abs(coeff) > global_tol]

                    for idx in mapped:
                        if idx not in component:
                            component.add(idx)
                            frontier.append(idx)

            component = sorted(component)
            done.update(component)

            component_pos = {idx: pos for pos, idx in enumerate(component)}
            current_basis = np.zeros((len(component), 0))

            for seed in component:
                orbit_vectors = []

                for sidx in range(len(self.symtext)):
                    coeffs = self.fxn_map[sidx, seed, :]
                    v = np.zeros(len(component))

                    for idx in component:
                        v[component_pos[idx]] = coeffs[idx]

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
                    # ProjectionOp only uses min(se_fxn_set), so return
                    # this seed as its own set.
                    seed_sets.append([seed])
                    current_basis = trial

                if new_rank == len(component):
                    break

        return seed_sets

    def special_function(self, salc, coord, sidx, irrmat):
        # Expand the coord-th basis function, as transformed by symel sidx,
        # in the same basis via fxn_map, and accumulate into salc.
        coeffs = self.fxn_map[sidx, coord, :]

        for out_idx, coeff in enumerate(coeffs):
            if abs(coeff) < global_tol:
                continue

            if self.symtext.complex:
                salc[:, :, out_idx] += np.conj(irrmat[sidx, :, :]) * coeff
            else:
                salc[:, :, out_idx] += irrmat[sidx, :, :] * coeff

        return salc
