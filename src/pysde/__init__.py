import warnings

from beartype.claw import beartype_this_package
from numba import NumbaPerformanceWarning

beartype_this_package()

warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
