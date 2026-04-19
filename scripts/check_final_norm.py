"""Find the final norm tensor name."""
import struct, json
path = "/home/shift/model.safetensors"
with open(path, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hdr_len))
for k in sorted(header.keys()):
    if "norm.weight" in k and "layer" not in k:
        print(k, header[k]["shape"])
