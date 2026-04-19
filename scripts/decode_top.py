"""Decode top predicted tokens from the model output."""
import json, sys

# Load tokenizer
with open("/home/shift/code/ferrisres/tokenizer.json", "r") as f:
    tok_data = json.load(f)

model = tok_data.get("model", {})
vocab = model.get("vocab", {})

# Reverse: id -> token
id_to_token = {v: k for k, v in vocab.items()}

prompt_tokens = [2, 3689, 563, 236743, 236778, 236862, 236778, 236881]
print("\n=== Prompt tokens ===")
for tid in prompt_tokens:
    token_str = id_to_token.get(tid, f"<UNKNOWN:{tid}>")
    print(f"  {tid}: '{token_str}'")

bos_top_tokens = [9622, 50921, 231990, 2795, 27978]
print("\n=== BOS single token top predictions (no PLE) ===")
for tid in bos_top_tokens:
    token_str = id_to_token.get(tid, f"<UNKNOWN:{tid}>")
    print(f"  {tid}: '{token_str}'")

# Top tokens from the Rust output
top_tokens = [207641, 99747, 241550, 208634, 220628]

for tid in top_tokens:
    token_str = id_to_token.get(tid, f"<UNKNOWN:{tid}>")
    # Try to decode the token bytes
    try:
        if token_str.startswith("<0x") and token_str.endswith(">"):
            byte_val = int(token_str[3:-1], 16)
            decoded = chr(byte_val)
        else:
            decoded = token_str.replace("▁", " ").replace("Ġ", " ")
        print(f"Token {tid}: '{token_str}' -> '{decoded}'")
    except:
        print(f"Token {tid}: '{token_str}'")

# Also check: what would token "4" or "four" be?
for target in ["4", "▁4", "four", "▁four", "2", "▁2", "two"]:
    if target in vocab:
        print(f"'{target}' -> token {vocab[target]}")
