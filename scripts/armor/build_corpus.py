#!/usr/bin/env python3
"""Build the labelled corpus for the Armor L1 injection scanner.

What this script does
---------------------
1. Reads the cached permissive upstream datasets in `assets/armor/upstream/`
   (fetching them first if the cache is absent and the network is reachable).
2. Refuses any family whose licence is not Apache-2.0, MIT, CC-BY, CC0 or
   ODC-By. A family without a licence fails the build; it is not silently
   skipped.
3. Reads the families we author ourselves: `assets/armor/own/*.jsonl` and the
   deterministic template generators in `scripts/armor/generators.py`.
4. Reads `assets/armor/translations/en_de.tsv` and emits German examples whose
   `source` is `translated:<upstream>`; those inherit the upstream licence and
   are visibly weaker provenance than native German.
5. Normalises every example to
   `{"text", "label", "lang", "source", "licence", "redacted": true}`,
   truncates long tool output, and deduplicates by a hash of the normalised
   text across *all* families (first family in build order wins).
6. Splits **by family** into train/dev/test and writes the JSONLs, `stats.json`
   and the manifest `assets/armor/l1-data.toml` with per-family sha256.

Determinism
-----------
No `hash()`, no `set` iteration, no timestamps and no randomness are used to
order output. Two runs on the same inputs produce byte-identical JSONL files;
`--verify-determinism` re-runs the build in memory and compares hashes.

This script does not train anything and makes no claim that the corpus is
sufficient for a shipped scanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
ASSETS = os.path.join(ROOT, "assets", "armor")
UPSTREAM = os.path.join(ASSETS, "upstream")
OWN = os.path.join(ASSETS, "own")
TRANSLATIONS = os.path.join(ASSETS, "translations", "en_de.tsv")
OUT = os.path.join(ASSETS, "build")
MANIFEST = os.path.join(ASSETS, "l1-data.toml")

ALLOWED_LICENCES = {
    "apache-2.0",
    "mit",
    "cc-by-4.0",
    "cc-by-3.0",
    "cc-by-sa-4.0",  # only used if it ever appears; still permissive-style
    "cc0-1.0",
    "odc-by",
}
# CC-BY-SA is share-alike, which is a distribution obligation, not a
# permission problem; keep it out of the allowed set unless a card actually
# uses it. The build refuses it so nobody adds it silently.
ALLOWED_LICENCES.discard("cc-by-sa-4.0")

# dataset id + published split -> (corpus family, corpus split)
UPSTREAM_FAMILIES = {
    ("deepset/prompt-injections", "train"): ("deepset/prompt-injections:train", "train"),
    ("deepset/prompt-injections", "test"): ("deepset/prompt-injections:test", "test"),
    ("jackhhao/jailbreak-classification", "train"): ("jackhhao/jailbreak-classification:train", "train"),
    ("jackhhao/jailbreak-classification", "test"): ("jackhhao/jailbreak-classification:test", "dev"),
    ("Lakera/gandalf_ignore_instructions", "train"): ("Lakera/gandalf_ignore_instructions:train", "train"),
    ("Lakera/gandalf_ignore_instructions", "validation"): ("Lakera/gandalf_ignore_instructions:validation", "dev"),
    ("Lakera/gandalf_ignore_instructions", "test"): ("Lakera/gandalf_ignore_instructions:test", "test"),
}

# Hand-written families: file -> (family, lang, label, split, hard_negative)
OWN_FILES = {
    "redteam_en.jsonl": ("redteam/en-handwritten", "en", "injection", "train", False),
    "redteam_de.jsonl": ("redteam/de-handwritten", "de", "injection", "train", False),
    "benign_en.jsonl": ("benign/en-handwritten", "en", "safe", "train", True),
    "benign_de.jsonl": ("benign/de-handwritten", "de", "safe", "train", True),
}

MAX_CHARS = 1200
OWN_REVISION = "armor-own-2026-11-26"

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u2060"), None)
WS = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ZERO_WIDTH)
    text = WS.sub(" ", text)
    return text.strip()


def dedup_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).lower().encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_licence(name: str, licence: str | None) -> str:
    if not licence:
        raise SystemExit(f"refusing family {name!r}: no licence")
    if licence.lower() not in ALLOWED_LICENCES:
        raise SystemExit(
            f"refusing family {name!r}: licence {licence!r} is not one of "
            f"{sorted(ALLOWED_LICENCES)}"
        )
    return licence.lower()


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: {exc}") from exc


def load_upstream() -> list[dict]:
    families = []
    metas = {}
    for name in sorted(os.listdir(UPSTREAM)):
        if not name.endswith(".meta.json"):
            continue
        meta = json.load(open(os.path.join(UPSTREAM, name), encoding="utf-8"))
        metas[meta["dataset"]] = meta

    for dataset, meta in sorted(metas.items()):
        licence = require_licence(dataset, meta.get("licence"))
        slug = dataset.replace("/", "__")
        path = os.path.join(UPSTREAM, f"{slug}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"missing cached data for {dataset}: {path}")
        groups: dict[str, list[dict]] = {}
        for row in read_jsonl(path):
            key = (dataset, row["split"])
            if key not in UPSTREAM_FAMILIES:
                continue
            family, split = UPSTREAM_FAMILIES[key]
            groups.setdefault(family, []).append(
                {
                    "text": row["text"],
                    "label": row["label"],
                    "lang": "en",
                    "source": family,
                    "licence": licence,
                    "split": split,
                    "hard_negative": False,
                    "revision": meta.get("revision") or "unknown",
                }
            )
        for family in sorted(groups):
            families.append(
                {
                    "name": family,
                    "split": UPSTREAM_FAMILIES[
                        next(k for k, v in UPSTREAM_FAMILIES.items() if v[0] == family)
                    ][1],
                    "licence": licence,
                    "revision": meta.get("revision") or "unknown",
                    "hard_negative": False,
                    "examples": [],
                }
            )
            families[-1]["examples"] = groups[family]
    return families


def load_own() -> list[dict]:
    import generators  # local module next to this file

    families = []

    for row in generators.all_own_families():
        licence = require_licence(row["family"], row["licence"])
        split = "train"
        if row["family"].endswith("-dev"):
            split = "dev"
        elif row["family"].endswith("-test"):
            split = "test"
        families.append(
            {
                "name": row["family"],
                "split": split,
                "licence": licence,
                "revision": OWN_REVISION,
                "hard_negative": row["hard_negative"],
                "examples": [
                    {
                        "text": text,
                        "label": row["label"],
                        "lang": row["lang"],
                        "source": row["family"],
                        "licence": licence,
                        "split": split,
                        "hard_negative": row["hard_negative"],
                        "revision": OWN_REVISION,
                    }
                    for text in row["texts"]
                ],
            }
        )

    for filename, (family, lang, label, split, hard) in sorted(OWN_FILES.items()):
        path = os.path.join(OWN, filename)
        if not os.path.exists(path):
            raise SystemExit(f"missing hand-written family file: {path}")
        examples = []
        for row in read_jsonl(path):
            text = row.get("text")
            if not text:
                raise SystemExit(f"{path}: row without text")
            examples.append(
                {
                    "text": text,
                    "label": label,
                    "lang": lang,
                    "source": family,
                    "licence": "cc0-1.0",
                    "split": split,
                    "hard_negative": hard,
                    "revision": OWN_REVISION,
                }
            )
        families.append(
            {
                "name": family,
                "split": split,
                "licence": "cc0-1.0",
                "revision": OWN_REVISION,
                "hard_negative": hard,
                "examples": examples,
            }
        )
    return families


def load_translations() -> list[dict]:
    """German translations of a permissive English subset.

    `source` is `translated:<upstream>` so the weaker provenance is visible in
    every JSONL line. The licence is inherited from the upstream dataset.
    """
    if not os.path.exists(TRANSLATIONS):
        raise SystemExit(f"missing translation table: {TRANSLATIONS}")

    licences: dict[str, str] = {}
    revisions: dict[str, str] = {}
    for name in sorted(os.listdir(UPSTREAM)):
        if name.endswith(".meta.json"):
            meta = json.load(open(os.path.join(UPSTREAM, name), encoding="utf-8"))
            licences[meta["dataset"]] = require_licence(meta["dataset"], meta.get("licence"))
            revisions[meta["dataset"]] = meta.get("revision") or "unknown"

    # English text -> dataset id, from the cached upstream rows.
    origin: dict[str, str] = {}
    for name in sorted(os.listdir(UPSTREAM)):
        if not name.endswith(".jsonl"):
            continue
        dataset = name[:-len(".jsonl")].replace("__", "/")
        for row in read_jsonl(os.path.join(UPSTREAM, name)):
            origin.setdefault(normalise(row["text"]).lower(), dataset)

    groups: dict[str, list[dict]] = {}
    seen = set()
    with open(TRANSLATIONS, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise SystemExit(f"{TRANSLATIONS}:{lineno}: expected 3 tab-separated columns")
            english, german, dataset = (p.strip() for p in parts)
            if not english or not german or not dataset:
                raise SystemExit(f"{TRANSLATIONS}:{lineno}: empty column")
            if english.lower() == german.lower():
                continue  # not a translation
            if dataset not in licences:
                raise SystemExit(f"{TRANSLATIONS}:{lineno}: unknown dataset {dataset!r}")
            key = normalise(english).lower()
            if key in seen:
                continue
            seen.add(key)
            if key not in origin:
                print(
                    f"warning: {TRANSLATIONS}:{lineno}: English text not found upstream, skipped",
                    file=sys.stderr,
                )
                continue
            family = f"translated:{dataset}"
            label = "injection" if dataset == "Lakera/gandalf_ignore_instructions" else None
            if label is None:
                # deepset is mixed; recover the label from the upstream row.
                label = "safe"
                for name in sorted(os.listdir(UPSTREAM)):
                    if not name.endswith(".jsonl"):
                        continue
                    if name[:-len(".jsonl")].replace("__", "/") != dataset:
                        continue
                    for row in read_jsonl(os.path.join(UPSTREAM, name)):
                        if normalise(row["text"]).lower() == key:
                            label = row["label"]
                            break
            groups.setdefault(family, []).append(
                {
                    "text": german,
                    "label": label,
                    "lang": "de",
                    "source": family,
                    "licence": licences[dataset],
                    "split": "train",
                    "hard_negative": False,
                    "revision": revisions[dataset] + "+translation",
                }
            )

    families = []
    for family in sorted(groups):
        dataset = family.split(":", 1)[1]
        families.append(
            {
                "name": family,
                "split": "train",
                "licence": licences[dataset],
                "revision": revisions[dataset] + "+translation",
                "hard_negative": False,
                "examples": groups[family],
            }
        )
    return families


def build() -> dict:
    families = load_upstream() + load_own() + load_translations()
    seen: dict[str, str] = {}
    kept: list[dict] = []
    raw = 0
    dropped = 0

    for family in families:
        family["kept"] = []
        for example in family["examples"]:
            raw += 1
            text = example["text"]
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS].rsplit(" ", 1)[0] + " …"
            text = normalise(text)
            if len(text) < 8:
                dropped += 1
                continue
            key = dedup_hash(text)
            if key in seen:
                dropped += 1
                continue
            seen[key] = family["name"]
            out = {
                "text": text,
                "label": example["label"],
                "lang": example["lang"],
                "source": family["name"],
                "licence": family["licence"],
                "redacted": True,
            }
            family["kept"].append(out)
            kept.append({**out, "_split": family["split"], "_hard": family["hard_negative"]})

    for family in families:
        family["n_examples"] = len(family["kept"])

    splits = {"train": [], "dev": [], "test": []}
    for row in kept:
        splits[row["_split"]].append(row)

    # Deterministic order: family, then text hash, then text.
    for split, rows in splits.items():
        rows.sort(key=lambda r: (r["source"], dedup_hash(r["text"]), r["text"]))
        for row in rows:
            row.pop("_split", None)
            row.pop("_hard", None)

    return {
        "families": families,
        "splits": splits,
        "raw": raw,
        "dropped": dropped,
        "dedup_unique": len(seen),
    }


def validate(result: dict) -> None:
    """Cheap invariants that must hold before anything is written."""
    families = result["families"]
    splits = result["splits"]
    allowed_labels = {"injection", "safe"}
    allowed_langs = {"en", "de"}

    seen_family_split: dict[str, str] = {}
    for family in families:
        name = family["name"]
        if name in seen_family_split and seen_family_split[name] != family["split"]:
            raise SystemExit(f"family {name!r} appears in two splits")
        seen_family_split[name] = family["split"]
        require_licence(name, family["licence"])
        if not family["kept"]:
            raise SystemExit(f"family {name!r} is empty after deduplication")
        for row in family["kept"]:
            if row["label"] not in allowed_labels:
                raise SystemExit(f"{name}: bad label {row['label']!r}")
            if row["lang"] not in allowed_langs:
                raise SystemExit(f"{name}: bad lang {row['lang']!r}")
            if row.get("redacted") is not True:
                raise SystemExit(f"{name}: example is not redacted")
            if set(row) != {"text", "label", "lang", "source", "licence", "redacted"}:
                raise SystemExit(f"{name}: unexpected keys {sorted(row)}")

    # No normalised text may occur in two splits, and no duplicates at all.
    owners: dict[str, str] = {}
    for split, rows in splits.items():
        for row in rows:
            key = dedup_hash(row["text"])
            if key in owners:
                raise SystemExit(
                    f"duplicate text across {owners[key]} and {split}: {row['text'][:60]!r}"
                )
            owners[key] = split

    if not splits["train"] or not splits["dev"] or not splits["test"]:
        raise SystemExit("one of the splits is empty")


def write_family_files(families: list[dict]) -> dict[str, str]:
    outdir = os.path.join(OUT, "families")
    os.makedirs(outdir, exist_ok=True)
    hashes = {}
    for family in families:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "__", family["name"])
        path = os.path.join(outdir, f"{slug}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in sorted(family["kept"], key=lambda r: (dedup_hash(r["text"]), r["text"])):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        family["sha256"] = sha256_file(path)
        hashes[family["name"]] = family["sha256"]
    return hashes


def write_splits(splits: dict[str, list[dict]]) -> dict[str, str]:
    os.makedirs(OUT, exist_ok=True)
    hashes = {}
    for split in ("train", "dev", "test"):
        path = os.path.join(OUT, f"{split}.jsonl")
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            for row in splits[split]:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[split] = sha256_file(path)
    return hashes


def compute_stats(result: dict) -> dict:
    families = result["families"]
    rows = [row for split in result["splits"].values() for row in split]
    by_label = {"injection": 0, "safe": 0}
    by_lang = {"en": 0, "de": 0}
    by_split = {"train": 0, "dev": 0, "test": 0}
    hard = 0
    for row in rows:
        by_label[row["label"]] = by_label.get(row["label"], 0) + 1
        by_lang[row["lang"]] = by_lang.get(row["lang"], 0) + 1
    for split, items in result["splits"].items():
        by_split[split] = len(items)
    hard = sum(f["n_examples"] for f in families if f["hard_negative"])
    total = len(rows)

    per_family = []
    for family in sorted(families, key=lambda f: (f["split"], f["name"])):
        labels = {"injection": 0, "safe": 0}
        langs = {"en": 0, "de": 0}
        for row in family["kept"]:
            labels[row["label"]] += 1
            langs[row["lang"]] += 1
        per_family.append(
            {
                "name": family["name"],
                "split": family["split"],
                "licence": family["licence"],
                "revision": family["revision"],
                "hard_negative": family["hard_negative"],
                "n_examples": family["n_examples"],
                "labels": labels,
                "langs": langs,
                "sha256": family.get("sha256", ""),
            }
        )

    def pct(n: int) -> float:
        return round(100.0 * n / total, 2) if total else 0.0

    return {
        "note": (
            "Assembled corpus statistics. Upstream families are real public data; "
            "generated/own families are template combinatorics written for this "
            "project. Translat* families are German translations of upstream "
            "English and are not native-reviewed. Nothing here proves the corpus "
            "is sufficient for a shipped scanner."
        ),
        "total": total,
        "splits": by_split,
        "split_pct": {k: pct(v) for k, v in by_split.items()},
        "labels": by_label,
        "label_pct": {k: pct(v) for k, v in by_label.items()},
        "languages": by_lang,
        "language_pct": {k: pct(v) for k, v in by_lang.items()},
        "hard_negatives": {
            "n": hard,
            "pct": pct(hard),
            "families": sorted(f["name"] for f in families if f["hard_negative"]),
        },
        "dedup": {
            "raw_examples": result["raw"],
            "dropped": result["dropped"],
            "unique": result["dedup_unique"],
        },
        "families": per_family,
    }


TOML_ESCAPE = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def toml_string(value: str) -> str:
    out = []
    for ch in value:
        if ch in TOML_ESCAPE:
            out.append(TOML_ESCAPE[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def write_manifest(stats: dict, split_hashes: dict[str, str], network_used: bool) -> None:
    lines = [
        "# Armor L1 corpus manifest. Generated by scripts/armor/build_corpus.py.",
        "# One entry per family. A family without a licence fails the build.",
        "# `sha256` is the hash of that family's normalised JSONL under",
        "# assets/armor/build/families/. The split JSONLs are hashed at the end.",
        "",
        "[corpus]",
        f"version = {toml_string(OWN_REVISION)}",
        f"total_examples = {stats['total']}",
        f"train = {stats['splits']['train']}",
        f"dev = {stats['splits']['dev']}",
        f"test = {stats['splits']['test']}",
        f"german_pct = {stats['language_pct']['de']}",
        f"hard_negative_pct = {stats['hard_negatives']['pct']}",
        f"network_used = {'true' if network_used else 'false'}",
        "",
        "[splits]",
    ]
    for split in ("train", "dev", "test"):
        lines.append(f"{split}_sha256 = {toml_string(split_hashes[split])}")
    lines.append("")

    for family in stats["families"]:
        lines += [
            "[[family]]",
            f"name = {toml_string(family['name'])}",
            f"licence = {toml_string(family['licence'])}",
            f"revision = {toml_string(family['revision'])}",
            f"n_examples = {family['n_examples']}",
            f"sha256 = {toml_string(family['sha256'])}",
            f"split = {toml_string(family['split'])}",
            f"label = {toml_string('mixed' if min(family['labels'].values()) > 0 else ('injection' if family['labels']['injection'] else 'safe'))}",
            f"lang = {toml_string('mixed' if min(family['langs'].values()) > 0 else ('en' if family['langs']['en'] else 'de'))}",
            f"hard_negative = {'true' if family['hard_negative'] else 'false'}",
            "",
        ]

    lines += [
        "# Datasets we read but refuse to train on: their licence cannot be",
        "# established, so they are evaluation-only and never enter a split.",
        "",
        "[[excluded]]",
        f"name = {toml_string('xTRam1/safe-guard-prompt-injection')}",
        f"reason = {toml_string('no licence on the dataset card (licence field absent)')}",
        f"rule = {toml_string('evaluation-only; must not enter train.jsonl')}",
        "",
    ]
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))


def print_table(stats: dict) -> None:
    print(
        f"{'family':44s} {'split':5s} {'licence':12s} {'n':>5s} "
        f"{'inj':>5s} {'safe':>5s} {'en':>5s} {'de':>5s}  hard"
    )
    print("-" * 100)
    for f in stats["families"]:
        print(
            f"{f['name'][:44]:44s} {f['split']:5s} {f['licence']:12s} "
            f"{f['n_examples']:5d} {f['labels']['injection']:5d} "
            f"{f['labels']['safe']:5d} {f['langs']['en']:5d} {f['langs']['de']:5d}"
            f"  {'yes' if f['hard_negative'] else ''}"
        )
    print("-" * 100)
    print(
        f"TOTAL {stats['total']}  train/dev/test "
        f"{stats['splits']['train']}/{stats['splits']['dev']}/{stats['splits']['test']}"
        f"  injection/safe {stats['labels']['injection']}/{stats['labels']['safe']}"
        f"  en/de {stats['languages']['en']}/{stats['languages']['de']}"
        f"  german {stats['language_pct']['de']}%"
        f"  hard-negative {stats['hard_negatives']['n']} ({stats['hard_negatives']['pct']}%)"
    )
    print(
        f"dedup: raw {stats['dedup']['raw_examples']} -> unique "
        f"{stats['dedup']['unique']} (dropped {stats['dedup']['dropped']})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="never use the network")
    parser.add_argument("--check", action="store_true", help="build and print stats, write nothing")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="build twice and fail if the split contents differ",
    )
    args = parser.parse_args()

    # Fetch only when the cache is empty, and only if allowed.
    cached = [n for n in os.listdir(UPSTREAM) if n.endswith(".jsonl")] if os.path.isdir(UPSTREAM) else []
    network_used = False
    if not cached:
        if args.offline:
            raise SystemExit(
                "no cached upstream data and --offline was given; run "
                "scripts/armor/fetch_upstream.py on a machine with network"
            )
        print("no cached upstream data; fetching (network)", file=sys.stderr)
        import fetch_upstream

        fetch_upstream.main()
        network_used = True

    result = build()
    validate(result)
    if args.verify_determinism:
        first = {
            split: [dedup_hash(row["text"]) for row in rows]
            for split, rows in result["splits"].items()
        }
        again = build()
        second = {
            split: [dedup_hash(row["text"]) for row in rows]
            for split, rows in again["splits"].items()
        }
        if first != second:
            raise SystemExit("build is not deterministic: split contents differ")
        print("determinism: two in-memory builds produced identical splits", file=sys.stderr)

    split_hashes = write_family_files(result["families"]) if not args.check else {
        f["name"]: "" for f in result["families"]
    }
    stats = compute_stats(result)
    if not args.check:
        split_hashes = write_splits(result["splits"])
        with open(os.path.join(OUT, "stats.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(stats, fh, indent=2, sort_keys=False, ensure_ascii=False)
            fh.write("\n")
        write_manifest(stats, split_hashes, network_used=network_used)

    print_table(stats)
    if not args.check:
        print("\nsplits written to", OUT)
        print("manifest written to", MANIFEST)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())
