"""OpenJSpace: interactive Jacobian-lens and J-space visualizer for
open-weight language models and VLMs.

Methodology follows the reference implementation at
https://github.com/anthropics/jacobian-lens (Apache-2.0); see NOTICE.
"""

from openjspace.core.applying import inspect_prompt
from openjspace.core.decomposition import nonnegative_omp
from openjspace.core.fitting import fit, jacobian_for_prompt
from openjspace.core.lens import JacobianLens
from openjspace.models.protocol import LensModelAdapter
from openjspace.models.registry import load_model

__version__ = "0.1.0"

__all__ = [
    "JacobianLens",
    "LensModelAdapter",
    "__version__",
    "fit",
    "inspect_prompt",
    "jacobian_for_prompt",
    "load_model",
    "nonnegative_omp",
]
