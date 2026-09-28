# -*- coding: utf-8 -*-
"""Both freeze files must parse, and v1 must be byte-identical to what was committed."""
import yaml, hashlib, subprocess, os
ROOT = os.environ.get("APE_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P = os.path.join(ROOT, "prereg")
for name in ("S1_freeze_2026-09-03.yaml", "S1_freeze_2026-09-03_v2.yaml",
             "S1_freeze_template.yaml"):
    p = os.path.join(P, name)
    d = yaml.safe_load(open(p, encoding="utf-8"))
    print("%-32s parses ok; top keys: %s" % (name, sorted(d.keys())))
    if "amendments" in d:
        print("   amendments:", [a["id"] for a in d["amendments"]])
        print("   frozen:", d["meta"]["frozen"], "| supersedes:", d["meta"]["supersedes"],
              "| v1 commit:", d["meta"]["supersedes_git_commit"])
    print("   sha256:", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16])

out = subprocess.run(["git", "-C", ROOT,
                      "diff", "--stat", "HEAD", "--", "prereg/S1_freeze_2026-09-03.yaml"],
                     capture_output=True, text=True)
print("v1 diff vs HEAD (empty means untouched):", repr(out.stdout.strip()))
