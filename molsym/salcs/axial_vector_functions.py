import numpy as np
from .function_set import FunctionSet


class AxialVectorFunctions(FunctionSet):
    """
    FunctionSet for axial vectors: Rx, Ry, Rz.
    Axial vectors transform as:
        D_axial(g) = det(R_g) R_g
    where R_g is the ordinary Cartesian representation.
    """

    def __init__(self, symtext):
        self.labels = ["Rx", "Ry", "Rz"]
        fxn_list = [0, 1, 2]
        super().__init__(symtext, fxn_list)

    def print_salcs(self, salcs):
        from molsym.salcs.salc_tools import format_salcs
        return str(format_salcs(salcs))
   
    def salc_to_string(self, salc):
        from molsym.salcs.salc_tools import axial_vector_salc_to_string
        return axial_vector_salc_to_string(salc, self)


    def get_fxn_map(self):
        """
        Shape:
            (nsymel, 3, 3)

        Convention:
            fxn_map[sidx, input_idx, output_idx] = coefficient
        """
        from molsym.salcs.salc_tools import axial_matrix

        fxn_map = np.zeros((len(self.symtext), 3, 3))

        for sidx, symel in enumerate(self.symtext.symels):
            A = np.array(symel.rrep, dtype=float)
            D = axial_matrix(A, self.symtext.mol.tol)

            # D[row, col] maps input col -> output row.
            # Store input_idx, output_idx for special_function.
            fxn_map[sidx, :, :] = D.T

        return fxn_map

    def get_symmetry_equiv_functions(self):
        """
        Group axial-vector components by actual symmetry mixing.
        """
        nfxn = 3
        done = set()
        equiv_sets = []

        for start in range(nfxn):
            if start in done:
                continue

            orbit = set([start])
            frontier = [start]

            while frontier:
                coord = frontier.pop()

                for sidx in range(len(self.symtext)):
                    coeffs = self.fxn_map[sidx, coord, :]

                    mapped = [i for i, coeff in enumerate(coeffs) if abs(coeff) > self.symtext.mol.tol]

                    for idx in mapped:
                        if idx not in orbit:
                            orbit.add(idx)
                            frontier.append(idx)

            equiv_set = sorted(orbit)
            equiv_sets.append(equiv_set)
            done.update(equiv_set)

        return equiv_sets

    def special_function(self, salc, coord, sidx, irrmat):
        """
        Project one axial-vector component under symmetry operation sidx.
        """
        coeffs = self.fxn_map[sidx, coord, :]

        for out_idx, coeff in enumerate(coeffs):
            if abs(coeff) < self.symtext.mol.tol:
                continue

            if self.symtext.complex:
                salc[:, :, out_idx] += np.conj(irrmat[sidx, :, :]) * coeff
            else:
                salc[:, :, out_idx] += irrmat[sidx, :, :] * coeff

        return salc
