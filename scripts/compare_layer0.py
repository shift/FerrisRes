"""Compare layer 0 forward pass between Python reference and our Rust implementation."""
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

def matmul(a, b, m, k, n):
    """C[m,n] = A[m,k] @ B[k,n]"""
    c = [0.0] * (m * n)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for l in range(k):
                s += a[i*k+l] * b[l*n+j]
            c[i*n+j] = s
    return c

def transpose(mat, rows, cols):
    out = [0.0] * (rows * cols)
    for r in range(rows):
        for c in range(cols):
            out[c*rows+r] = mat[r*cols+c]
    return out

path = sys.argv[1] if len(sys.argv) > 1 else "/home/shift/model.safetensors"
header, data_start = load_safetensors(path)

def get(name):
    return get_tensor(path, header, data_start, name)

hd = 1536
nh = 8
nkv = 1
head_dim = 256
ple_dim = 256

print("=== Python Reference: Layer 0 Forward ===")

# Embedding for BOS (token 2)
embed, es = get("model.language_model.embed_tokens.weight")
print(f"embed shape: {es}")
bos_emb = embed[2*hd:3*hd]
scale = math.sqrt(hd)
hidden = [v * scale for v in bos_emb]
l2 = math.sqrt(sum(v*v for v in hidden))
print(f"After embed+scale: l2={l2:.4f} first5=[{', '.join(f'{v:.4f}' for v in hidden[:5])}]")

# Save for Rust comparison
ref_after_embed = hidden[:10]

# Layer 0
in_norm, _ = get("model.language_model.layers.0.input_layernorm.weight")
normed = rms_norm(hidden, in_norm, hd)
l2 = math.sqrt(sum(v*v for v in normed))
print(f"After input_norm: l2={l2:.4f} first5=[{', '.join(f'{v:.4f}' for v in normed[:5])}]")

# Q/K/V projections
# PyTorch: y = x @ W.T where W is [out, in]
# Our code transposes to [in, out] and uses matmul(x, W_t, 1, hd, out)
# Both compute the same: c[j] = sum_d x[d] * W[j*hd+d]
q_raw, qs = get("model.language_model.layers.0.self_attn.q_proj.weight")
k_raw, ks = get("model.language_model.layers.0.self_attn.k_proj.weight")
v_raw, vs = get("model.language_model.layers.0.self_attn.v_proj.weight")
o_raw, os_ = get("model.language_model.layers.0.self_attn.o_proj.weight")
print(f"q_proj: {qs} k_proj: {ks} v_proj: {vs} o_proj: {os_}")

q_dim = qs[0]  # 2048
kv_dim = ks[0]  # 256

# Python: q = normed @ q_raw.T  (q_raw is [q_dim, hd])
# q[j] = sum_d normed[d] * q_raw[j * hd + d]
q = [0.0] * q_dim
for j in range(q_dim):
    s = 0.0
    for d in range(hd):
        s += normed[d] * q_raw[j * hd + d]
    q[j] = s
print(f"Q raw: l2={math.sqrt(sum(v*v for v in q)):.4f} first5=[{', '.join(f'{v:.4f}' for v in q[:5])}]")

k = [0.0] * kv_dim
for j in range(kv_dim):
    s = 0.0
    for d in range(hd):
        s += normed[d] * k_raw[j * hd + d]
    k[j] = s

v = [0.0] * kv_dim
for j in range(kv_dim):
    s = 0.0
    for d in range(hd):
        s += normed[d] * v_raw[j * hd + d]
    v[j] = s

# Per-head RMSNorm on Q
q_norm_w, _ = get("model.language_model.layers.0.self_attn.q_norm.weight")
for h in range(nh):
    base = h * head_dim
    sl = q[base:base+head_dim]
    ms = sum(x*x for x in sl) / head_dim + 1e-6
    inv = 1.0 / math.sqrt(ms)
    for d in range(head_dim):
        q[base+d] = sl[d] * inv * q_norm_w[d]
l2 = math.sqrt(sum(v*v for v in q))
print(f"Q after norm: l2={l2:.4f} first5=[{', '.join(f'{v:.4f}' for v in q[:5])}]")

# Per-head RMSNorm on K
k_norm_w, _ = get("model.language_model.layers.0.self_attn.k_norm.weight")
ms = sum(x*x for x in k) / kv_dim + 1e-6
inv = 1.0 / math.sqrt(ms)
for d in range(kv_dim):
    k[d] = k[d] * inv * k_norm_w[d % head_dim]

# V norm (no scale)
v = rms_norm_no_scale(v, kv_dim)

# RoPE (position 0, theta=10000, head_dim=256, factor=1.0)
# freq = 1/theta^(2d/head_dim)
for h in range(nh):
    base = h * head_dim
    half = head_dim // 2
    for d in range(half):
        freq = 1.0 / (10000.0 ** (2.0 * d / head_dim))
        angle = 0.0 * freq  # position 0 → all cos=1, sin=0 → no change
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x0 = q[base + d]
        x1 = q[base + d + half]
        q[base + d] = x0 * cos_a - x1 * sin_a
        q[base + d + half] = x0 * sin_a + x1 * cos_a

# Same for K
for d in range(head_dim // 2):
    freq = 1.0 / (10000.0 ** (2.0 * d / head_dim))
    angle = 0.0 * freq  # position 0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x0 = k[d]
    x1 = k[d + head_dim // 2]
    k[d] = x0 * cos_a - x1 * sin_a
    k[d + head_dim // 2] = x0 * sin_a + x1 * cos_a

# Attention: Q[8,256] @ K[1,256].T -> scores[8,1] -> softmax -> @ V[1,256]
# Scale = 1.0
attn_out = [0.0] * q_dim
for h in range(nh):
    qh = q[h*head_dim:(h+1)*head_dim]
    kh = k[:kv_dim]
    vh = v[:kv_dim]
    # Score
    score = sum(qh[d]*kh[d] for d in range(head_dim)) * 1.0
    # Softmax over 1 position = 1.0
    prob = 1.0
    for d in range(head_dim):
        attn_out[h*head_dim+d] = prob * vh[d]

# O projection: attn_out @ o_raw.T
output = [0.0] * hd
for j in range(hd):
    s = 0.0
    for d in range(q_dim):
        s += attn_out[d] * o_raw[j * q_dim + d]
    output[j] = s
l2 = math.sqrt(sum(v*v for v in output))
print(f"Attn output: l2={l2:.4f} first5=[{', '.join(f'{v:.4f}' for v in output[:5])}]")

# Post-attention norm
post_norm, _ = get("model.language_model.layers.0.post_attention_layernorm.weight")
output_normed = rms_norm(output, post_norm, hd)
l2 = math.sqrt(sum(v*v for v in output_normed))
print(f"After post_attn_norm: l2={l2:.4f} first5=[{', '.join(f'{v:.4f}' for v in output_normed[:5])}]")

# Residual
residual = [hidden[d] + output_normed[d] for d in range(hd)]
l2 = math.sqrt(sum(v*v for v in residual))
print(f"After attn residual: l2={l2:.4f} first5=[{', '.join(f'{v:.4f}' for v in residual[:5])}]")

print("\n=== Values for Rust comparison ===")
print(f"ref_after_embed = [{', '.join(f'{v:.6f}' for v in ref_after_embed)}]")
print(f"ref_after_input_norm = [{', '.join(f'{v:.6f}' for v in normed[:10])}]")
print(f"ref_Q_first10 = [{', '.join(f'{v:.6f}' for v in q[:10])}]")
print(f"ref_attn_output = [{', '.join(f'{v:.6f}' for v in output[:10])}]")
print(f"ref_after_residual = [{', '.join(f'{v:.6f}' for v in residual[:10])}]")
