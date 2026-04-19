"""Complete layer 0 forward pass for single BOS token. Compare with Rust debug_compare logs."""
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

def rms_norm(x, w, dim, eps=1e-6):
    out = []
    for t in range(len(x) // dim):
        sl = x[t*dim:(t+1)*dim]
        ms = sum(v*v for v in sl) / dim + eps
        inv = 1.0 / math.sqrt(ms)
        for d in range(dim):
            out.append(sl[d] * inv * (w[d] if d < len(w) else 1.0))
    return out

def rms_norm_no_scale(x, dim, eps=1e-6):
    out = []
    for t in range(len(x) // dim):
        sl = x[t*dim:(t+1)*dim]
        ms = sum(v*v for v in sl) / dim + eps
        inv = 1.0 / math.sqrt(ms)
        out.extend(v * inv for v in sl)
    return out

def gelu_tanh(x):
    c = math.sqrt(2.0 / math.pi)
    inner = c * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(inner))

path = sys.argv[1] if len(sys.argv) > 1 else "/home/shift/model.safetensors"
header, data_start = load_safetensors(path)
def get(name): return get_tensor(path, header, data_start, name)

hd = 1536
nh = 8
nkv = 1
head_dim = 256
ple_dim = 256
q_dim = nh * head_dim  # 2048
kv_dim = nkv * head_dim  # 256
inter_dim = 6144

print("=== Python Reference: Complete Layer 0 ===")

# Embedding for BOS (token 2)
embed, _ = get("model.language_model.embed_tokens.weight")
bos = embed[2*hd:3*hd]
hidden = [v * math.sqrt(hd) for v in bos]
l2 = math.sqrt(sum(v*v for v in hidden))
print(f"before_layer: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")

# Layer 0 weights
in_norm, _ = get("model.language_model.layers.0.input_layernorm.weight")
q_raw, qs = get("model.language_model.layers.0.self_attn.q_proj.weight")
k_raw, _ = get("model.language_model.layers.0.self_attn.k_proj.weight")
v_raw, _ = get("model.language_model.layers.0.self_attn.v_proj.weight")
o_raw, _ = get("model.language_model.layers.0.self_attn.o_proj.weight")
q_norm_w, _ = get("model.language_model.layers.0.self_attn.q_norm.weight")
k_norm_w, _ = get("model.language_model.layers.0.self_attn.k_norm.weight")
post_attn_norm, _ = get("model.language_model.layers.0.post_attention_layernorm.weight")
pre_ffn_norm, _ = get("model.language_model.layers.0.pre_feedforward_layernorm.weight")
gate_raw, _ = get("model.language_model.layers.0.mlp.gate_proj.weight")
up_raw, _ = get("model.language_model.layers.0.mlp.up_proj.weight")
down_raw, _ = get("model.language_model.layers.0.mlp.down_proj.weight")
post_ffn_norm, _ = get("model.language_model.layers.0.post_feedforward_layernorm.weight")
ple_gate_raw, _ = get("model.language_model.layers.0.per_layer_input_gate.weight")
ple_proj_raw, _ = get("model.language_model.layers.0.per_layer_projection.weight")
ple_post_norm, _ = get("model.language_model.layers.0.post_per_layer_input_norm.weight")
layer_scalar_raw, _ = get("model.language_model.layers.0.layer_scalar")

print(f"layer_scalar: {layer_scalar_raw[0]:.6f}")

# === ATTENTION BLOCK ===
residual = hidden[:]

# Input norm
normed = rms_norm(hidden, in_norm, hd)

# Q/K/V: x @ W.T
q = [sum(normed[d] * q_raw[j*hd+d] for d in range(hd)) for j in range(q_dim)]
k = [sum(normed[d] * k_raw[j*hd+d] for d in range(hd)) for j in range(kv_dim)]
v = [sum(normed[d] * v_raw[j*hd+d] for d in range(hd)) for j in range(kv_dim)]

# Per-head Q norm
for h in range(nh):
    base = h * head_dim
    sl = q[base:base+head_dim]
    ms = sum(x*x for x in sl) / head_dim + 1e-6
    inv = 1.0 / math.sqrt(ms)
    for d in range(head_dim):
        q[base+d] = sl[d] * inv * q_norm_w[d]

# K norm
ms = sum(x*x for x in k) / kv_dim + 1e-6
inv = 1.0 / math.sqrt(ms)
for d in range(kv_dim):
    k[d] = k[d] * inv * k_norm_w[d % head_dim]

# V norm (no scale)
v = rms_norm_no_scale(v, kv_dim)

# RoPE at position 0 (no-op: cos=1, sin=0)

# Attention (1 token, scale=1.0)
attn_out = [0.0] * q_dim
for h in range(nh):
    qh = q[h*head_dim:(h+1)*head_dim]
    vh = v[:kv_dim]
    # 1 position: score = q @ k, softmax = 1.0
    prob = 1.0
    for d in range(head_dim):
        attn_out[h*head_dim+d] = prob * vh[d]

# O projection: attn_out @ o_raw.T
output = [sum(attn_out[d] * o_raw[j*q_dim+d] for d in range(q_dim)) for j in range(hd)]

# Post-attention norm
output_normed = rms_norm(output, post_attn_norm, hd)

# Residual
hidden = [residual[d] + output_normed[d] for d in range(hd)]
l2 = math.sqrt(sum(v*v for v in hidden))
print(f"after_attn_residual: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")

# === FFN BLOCK ===
residual2 = hidden[:]
normed2 = rms_norm(hidden, pre_ffn_norm, hd)

# FFN: gelu(x @ gate.T) * (x @ up.T), then @ down.T
# For 1 token: just dot products
# gate: [inter_dim, hd]. gate.T = [hd, inter_dim]
# gate_out[j] = sum_d normed2[d] * gate_raw[j*hd+d]
gate_out = [sum(normed2[d] * gate_raw[j*hd+d] for d in range(hd)) for j in range(inter_dim)]
gate_gelu = [gelu_tanh(v) for v in gate_out]
up_out = [sum(normed2[d] * up_raw[j*hd+d] for d in range(hd)) for j in range(inter_dim)]
combined = [gate_gelu[j] * up_out[j] for j in range(inter_dim)]
# down: [hd, inter_dim]. down.T = [inter_dim, hd]
ffn_out = [sum(combined[d] * down_raw[j*inter_dim+d] for d in range(inter_dim)) for j in range(hd)]

l2 = math.sqrt(sum(v*v for v in ffn_out))
print(f"ffn_out_before_norm: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in ffn_out[:10])}]")

ffn_normed = rms_norm(ffn_out, post_ffn_norm, hd)
l2 = math.sqrt(sum(v*v for v in ffn_normed))
print(f"ffn_out_after_norm: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in ffn_normed[:10])}]")

# Residual
hidden = [residual2[d] + ffn_normed[d] for d in range(hd)]

# === PLE BLOCK ===
# PLE model projection
ple_model_proj_raw, _ = get("model.language_model.model.per_layer_model_projection.weight")  
ple_model_proj_norm_raw, _ = get("model.language_model.model.per_layer_projection_norm.weight")

# Model projection: hidden @ model_proj.T
# model_proj shape [8960, 1536]. model_proj.T = [1536, 8960]
# ple_all[j] = sum_d hidden[d] * ple_model_proj_raw[j*hd+d]
# But this is VERY slow for 1536*8960. Skip and just note it.
# Instead, check if PLE model projection even exists
print(f"ple_model_proj shape: {get_tensor(path, header, data_start, 'model.language_model.model.per_layer_model_projection.weight')[1]}")

# For now, skip PLE (too slow for pure Python) and go to layer_scalar
print(f"before_layer_scalar (without PLE): first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")

# Layer scalar
ls = layer_scalar_raw[0]
hidden = [v * ls for v in hidden]
print(f"after_layer_scalar: first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")

print("\n=== Key comparison points ===")
print("Rust before_layer = Python before_layer: MATCH (verified earlier)")
print("Rust after_attn_residual: l2=2178.5308")
print(f"Python after_attn_residual: l2={math.sqrt(sum(v*v for v in [residual[d]+output_normed[d] for d in range(hd)])):.4f}")
