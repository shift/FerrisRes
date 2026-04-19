"""Check actual Q/K/V tensor shapes for full attention layers."""
import struct, json

path = "/home/shift/model.safetensors"
with open(path, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hdr_len))

# Full attention layers: 4, 9, 14, 19, 24, 29, 34
for layer_idx in [0, 4, 9, 14, 15, 19, 34]:
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        key = f"model.language_model.layers.{layer_idx}.self_attn.{name}.weight"
        info = header[key]
        shape = info["shape"]
        hd = 1536
        q_dim = shape[0]
        head_dim = q_dim // 8  # num_heads=8
        print(f"L{layer_idx} {name}: {shape} → head_dim={head_dim}")
    print()
