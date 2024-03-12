import numpy as np

def epsilon(gamma):
    return np.exp(2 * np.pi * 1j / gamma)

def rename_irreps(irrmat, name_map):
    new_irrmat = {}
    for name_set in name_map:
        new_irrmat[name_set[1]] = irrmat[name_set[0]]
    return new_irrmat

def irrmat_gen_Cn(n):
    irrmat = {}
    irrmat["A"] = [[[1]] for m in range(n)]
    if n % 2 == 0:
        irrmat["B"] = [[[(-1)**m]] for m in range(n)]
    
    eps = epsilon(n)
    if 2 < n <= 4:
        irrmat["E_1"] = [[[eps**(m)]] for m in range(n)]
        irrmat["E_2"] = [[[eps**(-1*m)]] for m in range(n)]
    else:
        for a in range((n-1)>>1):
            irrmat[f"E{a+1}_1"] = [[[eps**((a+1)*m)]] for m in range(n)]
            irrmat[f"E{a+1}_2"] = [[[eps**(-1*(a+1)*m)]] for m in range(n)]
    return irrmat

# Define base irreps and regenerate from multiplications. Use multiplication table to place generated elements
from .multiplication_table import multiply
def pibbis(mult_table, m, Cn, prod):
    out = []
    for i in range(m): # Cycle through m
        out.append(multiply(mult_table, *prod, *[Cn for j in range(i)]))
    return out

def irrmat_gen_Cnh(n, mult_table):
    # Cn and i generate for even, Cn and sigma_h for odd
    irrmat = {}
    if n % 2 == 0:
        prs = pibbis(mult_table, n, 3, [0])
        #print(prs)
        iprs = pibbis(mult_table, n, 3, [2])
        #print(iprs)
        irrmat["Ag"] = np.ones((2*n,1,1))
        irrmat["Au"] = np.ones((2*n,1,1))
        irrmat["Au"][iprs] *= -1
        irrmat["Bg"] = np.ones((2*n,1,1))
        irrmat["Bg"][[prs[i] for i in range(n) if i % 2 == 1]] *= -1
        irrmat["Bg"][[iprs[i] for i in range(n) if i % 2 == 1]] *= -1
        irrmat["Bu"] = np.ones((2*n,1,1))
        irrmat["Bu"][[prs[i] for i in range(n) if i % 2 == 1]] *= -1
        irrmat["Bu"][[iprs[i] for i in range(n) if i % 2 == 0]] *= -1

        if 2 < n <=5:
            pass
        else:
            for a in range((n-1)>>1):
                pass
    else:
        pass
    return irrmat

