"""Linear operators for the tomography experiments.

Operator implementations by Jonathan Lindbloom.
"""
from compute_backend import DEVICE

CUPY_INSTALLED = DEVICE == "gpu"

from .util import *
from .base import *
from .base import _CustomLinearOperator, _AdjointLinearOperator
from .diagonal import *
from .derivatives import *
from .linalg.dct import *
from .pseudoinverse import QRPinvOperator, CGPreconditionedPinvModOperator
