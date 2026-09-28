"""Environment probe for the B (system library) track. Read-only."""
import importlib.util
import os
import sys
import urllib.request

print("python:", sys.version.replace("\n", " "))

for name in ("torch", "torchvision", "numpy", "pandas", "cv2", "yaml", "sklearn", "pytest", "pyarrow", "scipy"):
    spec = importlib.util.find_spec(name)
    print(f"pkg {name}: {'yes' if spec is not None else 'MISSING'}")

import torch  # noqa: E402
import torchvision  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import cv2  # noqa: E402

print("torch", torch.__version__, "tv", torchvision.__version__, "np", np.__version__,
      "pd", pd.__version__, "cv2", cv2.__version__)
print("cuda available:", torch.cuda.is_available(), "n_dev:", torch.cuda.device_count())

CSV = os.path.join(os.environ.get("APE_ACCIDENT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "external", "ACCIDENT_2026")), "metadata-real.csv")
df = pd.read_csv(CSV)
print("csv shape:", df.shape)
print("columns:", list(df.columns))
print("type unique:", sorted(df["type"].astype(str).unique().tolist()))
print("split_in_distribution counts:")
print(df["split_in_distribution"].value_counts().to_string())
print("fps stats:")
fps = df["no_frames"] / df["duration"]
print("fps min/med/max:", float(fps.min()), float(fps.median()), float(fps.max()))
post = df["duration"] - df["accident_time"]
print("post-anchor length quantiles (25/50/75):",
      float(post.quantile(0.25)), float(post.quantile(0.50)), float(post.quantile(0.75)))
print("post-anchor min/max:", float(post.min()), float(post.max()))
print("n post<0:", int((post < 0).sum()))
print("sample paths:", df["path"].head(3).tolist())

# network probe for torchvision pretrained weights
URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
try:
    req = urllib.request.Request(URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=25) as r:
        print("NETWORK_OK", r.status, r.headers.get("Content-Length"))
except Exception as exc:  # noqa: BLE001
    print("NETWORK_FAIL", type(exc).__name__, exc)

print("PROBE_DONE")
