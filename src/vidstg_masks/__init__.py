"""vidstg_masks: SAM 3.1 mask tracks for the subject and object of every VidSTG relation.

Inputs are VidOR videos + VidOR annotation JSONs (boxes) and VidSTG annotation JSONs
(relations). Prompts are geometric: VidOR human-keyframe boxes as corner points.
torch and sam3 are imported lazily inside sam_session.py only, so every other module
works on a CPU-only login node.
"""

__version__ = "0.1.0"

MODEL_NAME = "sam3.1-object-multiplex"
SAM_VERSION = "3.1"
SAM3_COMMIT = "96914d2425f90a64f45ca977c2b5165418099543"
CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
