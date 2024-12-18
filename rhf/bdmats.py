import numpy as np
from scipy.linalg import fractional_matrix_power

class BDMatrix():
    """
    Block diagonal matrix object.
    Defines functions to speed up computation by ignoring zero blocks.
    Can be tempermental so be careful when using these bad boys.
    Also, probably not optimized, but I'm not optimizing things in Python...
    """
    def __init__(self, blocks):
        #print("did we init?")
        self.blocks = blocks

    def __add__(self, A):
        if self.check_size(A):
            B = []
            for i, block in enumerate(self.blocks):
                B.append(block + A.blocks[i])
        return BDMatrix(B)

    def __sub__(self, A):
        return self+(-1*A)

    def __mul__(self, n):
        B = []
        for i, block in enumerate(self.blocks):
            B.append(np.multiply(block,n.blocks[i]))
        return BDMatrix(B)

    def __rmul__(self, n):
        return self.__mul__(n)

    def sum(self):
        suum = 0
        for i, block in enumerate(self.blocks):
            if np.size(block) == 0:
                continue
            else:
                suum += sum(sum(block))
        return suum

    def dot(self, A):
        if self.check_size(A):
            B = []
            for i, block in enumerate(self.blocks):
                if block.size == 0:
                    B.append(block)
                elif block.size == 1:
                    B.append(np.array([block[0]*A.blocks[i][0]]))
                else:
                    B.append(np.dot(block, A.blocks[i]))
        return BDMatrix(B)

    def transpose(self):
        B = []
        for i, block in enumerate(self.blocks):
            B.append(block.transpose())
        return BDMatrix(B)
    
    def eigh(self):
        eigval = []
        eigvec = []
        for i, block in enumerate(self.blocks):
            if np.size(block) < 1:
                eigvec.append(np.array([]))
                eigval.append(np.empty((0,)))
            else:
                e,v = np.linalg.eigh(block)
                eigval.append(e)
                eigvec.append(v)
        return eigval, BDMatrix(eigvec)

    def __pow__(self, n):
        B = []
        for i, block in enumerate(self.blocks):
            if block.size == 0:
                B.append(block)
            elif block.size == 1:
                B.append(np.array([block[0]**n]))
            else:
                B.append(fractional_matrix_power(block, n))
        return BDMatrix(B)
    
    def check_size(self, A):
        for i,block in enumerate(self.blocks):
            if np.shape(A.blocks[i])[0] != np.shape(block)[0]:
                raise ValueError(": Arrays do not have same shape")
        return True

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.blocks)

    #def sblock(self, n):
    #    return self.blocks[n]

    def full_mat(self):
        fshape = 0
        for i, block in enumerate(self.blocks):
            fshape += np.shape(block)[0]
        fullmat = np.zeros((fshape,fshape))
        c = 0
        for i, block in enumerate(self.blocks):
            s = np.shape(block)[0]
            fullmat[c:c+s, c:c+s] = block
            c += s
        return fullmat
if __name__ == "__main__":
    #print('lol where are we')
    A = np.array([[1,2],[3,4]])
    B = np.array([[1,2,3],[4,5,6],[7,8,9]])
    C = np.array([1])
    D = np.array([])
    bdmat = BDMatrix((A,B,C))
    bdmat2 = BDMatrix((B,D,C,D,C,A,B,C))
    #print(bdmat2.full_mat())

