'''
This module is used to fit elastic incoherent neutron scatting (EINS) data. 
The main class is EINSfit. Most of the other functions are used by this class and are usually not needed.

Needs packages:
    - python >=3.6 (tested with 3.7)
    - numpy        (tested with 1.16)
    - matplotlib   (tested with 2.0 and 3.0)
    - cycler       (tested with 0.10)
    - lmfit >= 0.9 (tested with 0.9.13)

Module created by Dominik Zeller.
'''

from .definitions import EINSfit
from .definitions import take_closest_value

from .version import __all__
__version__=__all__['__version__']

