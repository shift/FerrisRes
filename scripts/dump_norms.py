"""Dump input_norm and final_norm weights for debugging."""
import struct, json, math, sys

def load_safetensors(path):
    with open(path, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hdr_len))
        data_start = 8 + hdr_len
    return header, data_start

def get_tensor(path, header, data_start, name):
    info = header[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_start + s)
        raw = f.read(e - s)
    shape = info["shape"]
    vals = []
    for i in range(0, len(raw), 2):
        h = struct.unpack("<H", raw[i:i+2])[0]
        bits = h << 16
        vals.append(struct.unpack("<f", struct.pack("<I", bits))[0])
    return vals, shape

path = sys.argv[1] if len(sys.argv) > 1 else "/home/shift/model.safetensors"
header, data_start = load_safetensors(path)
def get(name): return get_tensor(path, header, data_start, name)

# Check norm weights for layer 0
for name in [
    "model.language_model.layers.0.input_layernorm.weight",
    "model.language_model.layers.0.post_attention_layernorm.weight",
    "model.language_model.layers.0.pre_feedforward_layernorm.weight",
    "model.language_model.layers.0.post_feedforward_layernorm.weight",
    "model.language_model.layers.0.post_per_layer_input_norm.weight",
    "model.language_model.model.norm.weight",  # final norm
]:
    try:
        vals, shape = get(name)
        mn = min(vals)
        mx = max(vals)
        avg = sum(abs(v) for v in vals) / len(vals)
        print(f"{name.split('.')[-2]}: shape={shape} min={mn:.4f} max={mx:.4f} avg_abs={avg:.4f} first5={[round(v,4) for v in vals[:5]]}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
