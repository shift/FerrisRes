"""Check if lm_head exists in safetensors and its shape."""
import struct, json
path = "/home/shift/model.safetensors"
with open(path, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hdr_len))
for k in sorted(header.keys()):
    if "lm_head" in k or "embed_tokens" in k:
        print(k, header[k]["shape"], header[k]["dtype"])
