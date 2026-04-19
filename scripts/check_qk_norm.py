"""Check Q/K norm weight magnitudes for different layer types."""
import struct, json, math

path = "/home/shift/model.safetensors"
with open(path, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hdr_len))
    ds = 8 + hdr_len

def g(name):
    info = header[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f2:
        f2.seek(ds + s)
        raw = f2.read(e - s)
    shape = info["shape"]
    vals = []
    for i in range(0, len(raw), 2):
        h = struct.unpack("<H", raw[i:i+2])[0]
        vals.append(struct.unpack("<f", struct.pack("<I", h << 16))[0])
    return vals, shape

# Layer 0 (sliding, head_dim=256) and layer 4 (full, head_dim=512)
for layer_idx in [0, 4, 9, 14, 19]:
    qn, qns = g(f"model.language_model.layers.{layer_idx}.self_attn.q_norm.weight")
    kn, kns = g(f"model.language_model.layers.{layer_idx}.self_attn.k_norm.weight")
    qn_avg = sum(abs(v) for v in qn) / len(qn)
    kn_avg = sum(abs(v) for v in kn) / len(kn)
    print(f"Layer {layer_idx}: q_norm shape={qns} avg_abs={qn_avg:.4f} first3={[round(v,4) for v in qn[:3]]}")
    print(f"Layer {layer_idx}: k_norm shape={kns} avg_abs={kn_avg:.4f} first3={[round(v,4) for v in kn[:3]]}")
