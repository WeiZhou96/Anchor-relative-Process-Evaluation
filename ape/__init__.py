"""APE: anchor-relative process evaluation of collision-type determination.

This package is the *instrument*, not a method. It defines the evaluation
protocol (anchor-relative nested prefixes, a fixed observation horizon and a
fixed eligibility cohort), computes the process metrics on cached answer
matrices, characterises the protocol's response to its own knobs, and reports
phenomena. It proposes no model, trains nothing, and never ranks a system as
better than another.

Module map
----------
``classes``    closed-set class codes and the ``BOT`` symbol
``protocol``   protocol vector, ``pi_hash``, anchor perturbation, prefix lists
``cohort``     fixed eligibility cohorts and source-video-cluster utilities
``metrics``    answer matrices, ``S_H``, ``RMSCD@H``, flips, commitment triple
``stats``      cluster-level paired bootstrap, the pair set ``P``, Holm
``calib``      ``b_M``, ``s_M``, ``R_M``, comparability region, ``MRD_M``
``blocks``     synthetic gauge blocks and their analytic expectations
``phenomena``  phenomenon statistics and the K1-K6 checks (report only)
``cli``        ``make-prefixes / blocks / eval / scan / phenomena / report-json``
"""

from __future__ import annotations

__version__ = "0.1.0"

from .classes import BOT, CLASS_NAMES, N_CLASSES  # noqa: F401
from .protocol import (  # noqa: F401
    ProtocolConfig,
    ProtocolVector,
    load_manifest,
    load_protocol,
    make_prefixes,
    perturb_manifest,
)
from .cohort import cohort, cohort_summary, eligible_mask  # noqa: F401
from .metrics import AnswerTable, EvalResult, evaluate_system  # noqa: F401

__all__ = [
    "__version__",
    "BOT",
    "CLASS_NAMES",
    "N_CLASSES",
    "ProtocolConfig",
    "ProtocolVector",
    "load_manifest",
    "load_protocol",
    "make_prefixes",
    "perturb_manifest",
    "cohort",
    "cohort_summary",
    "eligible_mask",
    "AnswerTable",
    "EvalResult",
    "evaluate_system",
]
