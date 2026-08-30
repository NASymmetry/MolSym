<p align="center">
  <img src="molsym_logo_v3.svg" width="400" alt=""/>
</p>
<table align="center">
  <tr>
    <th>Documentation</th>
    <th>CI</th>
    <th>Coverage</th>
    <th>Citation</th>
  </tr>
  <tr>
    <td align="center">
      <a href='https://molsym.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/molsym/badge/?version=latest' alt='Documentation Status' />
      </a>
    </td>
    <td align="center">
      <a href=https://github.com/NASymmetry/MolSym/actions/workflows/workflow.yml>
      <img src=https://github.com/NASymmetry/MolSym/actions/workflows/workflow.yml/badge.svg>
      </a>
    </td>
    <td align="center">
      <a href=https://codecov.io/gh/NASymmetry/MolSym>
      <img src=https://codecov.io/gh/NASymmetry/MolSym/branch/main/graph/badge.svg?token=NQDJ0QYLB0>
      </a> 
    </td>
     <td align="center">
      <a href=https://doi.org/10.1063/5.0216738>
      <img src=https://img.shields.io/badge/JCP-10.1063/5.0216738-purple.svg>
      </a>
    </td>
  </tr>
</table>

# MolSym
A python package for handling molecular symmetry.


## Capabilities
- [Point group detection](https://github.com/NASymmetry/MolSym/wiki/Point-group-detection)
- [Molecule symmetrization](https://github.com/NASymmetry/MolSym/wiki/Symmetrizing-a-molecule)
- Symmetry element generation
- Character table generation
- SALC generation for [atomic basis functions](https://github.com/NASymmetry/MolSym/wiki/SALCs#spherical-harmonics), [internal coordinates](https://github.com/NASymmetry/MolSym/wiki/SALCs#internal-coordinates), and [cartesian coordinates](https://github.com/NASymmetry/MolSym/wiki/SALCs#cartesian-coordinates)

## Installing
We recommend installing MolSym into its own environment. For example, with conda:

  `conda create -n "NameYourEnvironment" python=3.X`

  `conda activate "NameYourEnvironment"`

MolSym is available on [PyPI](https://pypi.org/project/molsym/) and can be installed with `pip`:

  `pip install molsym`

MolSym is also available on conda-forge and can be installed with `conda`:

  `conda install -c conda-forge molsym`

MolSym is tested with Python 3.10–3.14. Some features rely on the optional `QCElemental` dependency, which can be pulled in with:

  `pip install "molsym[qcel]"`

### Development installation
To work on MolSym itself, clone the repository and install it in editable mode:

  `git clone https://github.com/NASymmetry/MolSym.git # or git@github.com:NASymmetry/MolSym.git`

  `cd MolSym`

  `pip install -e .`

Include the optional `QCElemental`-backed features with:

  `pip install -e ".[qcel]"`
