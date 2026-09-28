"""Read-only compatibility check of the exact requested CLIP weights."""
from systems.r2_features import load_clip_model
model=load_clip_model()
print('CLIP_LOAD_PROBE_OK',type(model).__name__,sum(p.numel() for p in model.parameters()),model.config._commit_hash,flush=True)
