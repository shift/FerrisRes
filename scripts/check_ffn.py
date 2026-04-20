"""Check FFN weight shapes and first elements."""
import struct, json

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
    for i in range(0, min(20, len(raw)), 2):
        h = struct.unpack("<H", raw[i:i+2])[0]
        vals.append(struct.unpack("<f", struct.pack("<I", h << 16))[0])
    return vals, shape

gate, gs = g("model.language_model.layers.0.mlp.gate_proj.weight")
up, us = g("model.language_model.layers.0.mlp.up_proj.weight")
down, ds_ = g("model.language_model.layers.0.mlp.down_proj.weight")
print(f"gate: {gs} first5={[round(v,4) for v in gate[:5]]}")
print(f"up: {us} first5={[round(v,4) for v in up[:5]]}")
print(f"down: {ds_} first5={[round(v,4) for v in down[:5]]}")

# Check layer 5 (which has different head_dim)
q5, q5s = g("model.language_model.layers.5.self_attn.q_proj.weight")
print(f"layer5 q_proj: {q5s} first5={[round(v,4) for v in q5[:5]]}")

# Check layer_scalar for layer 0
ls, lss = g("model.language_model.layers.0.layer_scalar")
print(f"layer_scalar: {lss} val={ls[0]}")

# Check FFN intermediate dim for layers 5, 10, 20
for l in [0, 5, 10, 15, 20, 30]:
    gate_l, gls = g(f"model.language_model.layers.{l}.mlp.gate_proj.weight")
    print(f"layer {l} gate_proj: {gls}")
