"""Compatibility import for the retired pre-PRL module path.

The strict PKDD action is owned by :mod:`dpl_rhf.functionals.pkdd_action`.
Legacy training entrypoints remain outside the supported PRL-RMF path.
"""

from dpl_rhf.functionals.pkdd_action import *  # noqa: F401,F403
