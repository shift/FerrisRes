#!/usr/bin/env python3
import pymupdf4llm
import os

pdfs = [
    "learnable_tokenization.pdf",
    "smiles_pair_encoding.pdf", 
    "merge_dna.pdf",
    "related/spacebyte.pdf",
    "related/mambabyte.pdf"
]

for pdf in pdfs:
    if os.path.exists(pdf):
        md = pymupdf4llm.to_markdown(pdf)
        out = pdf.replace(".pdf", "_extracted.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"{pdf}: {len(md)} chars -> {out}")
    else:
        print(f"Skipping {pdf} (not found)")

# List output files
print("\n=== Extracted files ===")
for f in sorted(os.listdir(".")):
    if f.endswith("_extracted.md"):
        print(f"{f}: {os.path.getsize(f)} bytes")