<p align="center">
  <img src="molsym_logo_v3.svg" width="400" alt=""/>
</p>
<table align="center">
  <tr>
    <th>Documentation</th>
    <th>CI</th>
    <th>Coverage</th>
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
      <a href=https://codecov.io/gh/CCQC/MolSym>
      <img src=https://codecov.io/gh/CCQC/MolSym/branch/main/graph/badge.svg?token=NQDJ0QYLB0>
      </a> 
    </td>
  </tr>
</table>

# MolSym
A python package for handling molecular symmetry.


## Capabilities
- Point group detection
- Symmetry element generation
- Character table generation
- SALC generation for atomic basis functions, internal coordinates, and cartesian coordinates

## Installing
As of now we do not have a better way to install the code other than cloning from GitHub.
Create a new conda environment with:

  `conda create -n "NameYourEnvironment" python=3.X`

MolSym is tested with Python 3.9-3.13, but should work for more recent versions and some older versions as well.
  
  `git clone git@github.com:NASymmetry/MolSym.git`

Install the necessary dependencies using `pip`.
  
  `pip install -r <Path to MolSym directory>/requirements.tx`

Alternatively, most Python environments come equipped with all but one dependency, so if `pip` is not desired, installing `QCElemental` is all that should be required.

  `conda install -c conda-forge qcelemental`

Finally append the MolSym directory to your `PYTHONPATH`.
  
 `export PYTHONPATH=$PYTHONPATH:<Path to MolSym directory>`
