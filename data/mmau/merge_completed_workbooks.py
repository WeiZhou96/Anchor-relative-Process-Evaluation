"""Read submitted workbooks; preserve originals and write a new JSON receipt."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from mmau_mapping import merge_forms


def read_form(path: Path) -> tuple[str, str, list[dict]]:
    identity = pd.read_excel(path, sheet_name="独立判定", header=None, nrows=3)
    identity = identity.reindex(index=range(3), columns=range(5)).fillna("")
    name, date = str(identity.iloc[2, 1]).strip(), str(identity.iloc[2, 3]).strip()
    if not name or not date:
        raise ValueError("Both mapper name and completion date are required")
    rows = pd.read_excel(path, sheet_name="独立判定", header=5).fillna("").to_dict("records")
    return name, date, rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)
    p.add_argument("--definitions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    one, two = read_form(args.first), read_form(args.second)
    if one[0] == two[0] or args.first.resolve() == args.second.resolve():
        raise ValueError("Two different human mappers and original files are required")
    result = merge_forms(one[2], two[2], json.loads(args.definitions.read_text(encoding="utf-8")))
    receipt = dict(
        rows=result,
        submissions=[
            dict(
                path=str(path), mapper=data[0], completed=data[1], sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            for path, data in [(args.first, one), (args.second, two)]
        ],
        independence_note="Names and hashes record provenance; software cannot establish human independence.",
    )
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
