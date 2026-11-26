#!/usr/bin/env python3
"""Fetch the permissive upstream injection datasets into assets/armor/upstream.

The corpus build reads these cached JSONL snapshots so that a rebuild is
deterministic and works without a network. Each snapshot is accompanied by a
`<slug>.meta.json` that records the dataset id, the exact revision (commit sha)
and the licence taken from the dataset card.

No dataset is fetched unless its card declares one of the licences in
`ALLOWED_LICENCES`. A dataset without a licence is skipped and reported.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ALLOWED_LICENCES = {"apache-2.0", "mit", "cc-by-4.0", "cc-by-3.0", "cc0-1.0", "odc-by"}

# dataset id -> list of (config, split) to concatenate, plus row adapters.
DATASETS = {
    "deepset/prompt-injections": {
        "splits": [("default", "train"), ("default", "test")],
        "text": "text",
        "label": lambda row: "injection" if int(row["label"]) == 1 else "safe",
    },
    "jackhhao/jailbreak-classification": {
        "splits": [("default", "train"), ("default", "test")],
        "text": "prompt",
        "label": lambda row: "injection" if str(row["type"]).lower().startswith("jailbreak") else "safe",
    },
    "Lakera/gandalf_ignore_instructions": {
        "splits": [("default", "train"), ("default", "validation"), ("default", "test")],
        "text": "text",
        "label": lambda row: "injection",
    },
}

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "assets", "armor", "upstream"))


def get_json(url: str, attempts: int = 4) -> dict:
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "armor-corpus-build/1.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            if i == attempts - 1:
                raise
            print(f"  retry {i + 1}: {exc}", file=sys.stderr)
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def fetch_rows(dataset: str, config: str, split: str) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while True:
        url = (
            "https://datasets-server.huggingface.co/rows"
            f"?dataset={urllib.parse.quote(dataset)}&config={urllib.parse.quote(config)}"
            f"&split={urllib.parse.quote(split)}&offset={offset}&length=100"
        )
        page = get_json(url)
        batch = [r["row"] for r in page.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        total = int(page.get("num_rows_total", 0))
        if not batch or offset >= total:
            break
    return rows


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    missing: list[str] = []
    for dataset, spec in DATASETS.items():
        slug = dataset.replace("/", "__")
        print(f"fetching {dataset}")
        info = get_json(f"https://huggingface.co/api/datasets/{dataset}")
        licence = ((info.get("cardData") or {}).get("license") or "").lower()
        if licence not in ALLOWED_LICENCES:
            print(f"  REFUSED: licence={licence!r} not in {sorted(ALLOWED_LICENCES)}")
            missing.append(f"{dataset} (licence={licence or 'unknown'})")
            continue
        out = []
        for config, split in spec["splits"]:
            for row in fetch_rows(dataset, config, split):
                text = str(row.get(spec["text"], "")).strip()
                if not text:
                    continue
                out.append(
                    {
                        "text": text,
                        "label": spec["label"](row),
                        "dataset": dataset,
                        "split": split,
                    }
                )
        path = os.path.join(OUT_DIR, f"{slug}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for item in out:
                fh.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        meta = {
            "dataset": dataset,
            "licence": licence,
            "revision": info.get("sha"),
            "splits": [f"{c}/{s}" for c, s in spec["splits"]],
            "n_examples": len(out),
        }
        with open(os.path.join(OUT_DIR, f"{slug}.meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"  wrote {len(out)} rows -> {path}")

    if missing:
        print("\nnot fetched:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
