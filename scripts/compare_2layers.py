"""Reference: Layer 0 + Layer 1 forward for single BOS token (token 0 only, ignoring other 7 tokens).
This gives us reference values at key intermediate points for comparison with Rust.
Since we only process 1 token, the matmul is just dot products (fast!)."""
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
    ms = sum(v*v for v in x) / dim + eps
    inv = 1.0 / math.sqrt(ms)
    return [x[d] * inv * w[d] for d in range(dim)]

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
q_dim = nh * head_dim  # 2048
kv_dim = nkv * head_dim  # 256

# Embedding for BOS
embed, _ = get("model.language_model.embed_tokens.weight")
hidden = [v * math.sqrt(hd) for v in embed[2*hd:3*hd]]
print(f"before_layer 0: first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")

# Layer types
layer_types = ['sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention',
               'sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention',
               'sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention',
               'sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention',
               'sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention',
               'sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention',
               'sliding_attention','sliding_attention','sliding_attention','sliding_attention','full_attention']

def forward_layer(layer_idx, hidden):
    """Forward pass for one layer, one token. Returns hidden after all operations."""
    is_full = layer_types[layer_idx] == "full_attention"
    layer_head_dim = 512 if is_full else 256
    layer_q_dim = nh * layer_head_dim
    layer_kv_dim = nkv * layer_head_dim
    rope_theta = 1000000.0 if is_full else 10000.0
    partial_factor = 0.25 if is_full else 1.0

    prefix = f"model.language_model.layers.{layer_idx}"
    in_norm, _ = get(f"{prefix}.input_layernorm.weight")
    q_raw, _ = get(f"{prefix}.self_attn.q_proj.weight")
    k_raw, _ = get(f"{prefix}.self_attn.k_proj.weight")
    v_raw, _ = get(f"{prefix}.self_attn.v_proj.weight")
    o_raw, _ = get(f"{prefix}.self_attn.o_proj.weight")
    q_norm_w, _ = get(f"{prefix}.self_attn.q_norm.weight")
    k_norm_w, _ = get(f"{prefix}.self_attn.k_norm.weight")
    post_attn_norm, _ = get(f"{prefix}.post_attention_layernorm.weight")
    pre_ffn_norm, _ = get(f"{prefix}.pre_feedforward_layernorm.weight")
    gate_raw, _ = get(f"{prefix}.mlp.gate_proj.weight")
    up_raw, _ = get(f"{prefix}.mlp.up_proj.weight")
    down_raw, _ = get(f"{prefix}.mlp.down_proj.weight")
    post_ffn_norm, _ = get(f"{prefix}.post_feedforward_layernorm.weight")
    layer_scalar_raw, _ = get(f"{prefix}.layer_scalar")
    ls = layer_scalar_raw[0]

    # --- Attention ---
    residual = hidden[:]
    normed = rms_norm(hidden, in_norm, hd)

    # Q/K/V: x @ W.T
    q = [sum(normed[d] * q_raw[j*hd+d] for d in range(hd)) for j in range(layer_q_dim)]
    k = [sum(normed[d] * k_raw[j*hd+d] for d in range(hd)) for j in range(layer_kv_dim)]
    v_raw_out = [sum(normed[d] * v_raw[j*hd+d] for d in range(hd)) for j in range(layer_kv_dim)]

    # Per-head Q norm
    for h in range(nh):
        base = h * layer_head_dim
        sl = q[base:base+layer_head_dim]
        ms = sum(x*x for x in sl) / layer_head_dim + 1e-6
        inv = 1.0 / math.sqrt(ms)
        for d in range(layer_head_dim):
            q[base+d] = sl[d] * inv * q_norm_w[d % len(q_norm_w)]

    # K norm
    ms = sum(x*x for x in k) / layer_kv_dim + 1e-6
    inv = 1.0 / math.sqrt(ms)
    for d in range(layer_kv_dim):
        k[d] = k[d] * inv * k_norm_w[d % len(k_norm_w)]

    # V norm (no scale)
    ms = sum(x*x for x in v_raw_out) / layer_kv_dim + 1e-6
    inv = 1.0 / math.sqrt(ms)
    v = [x * inv for x in v_raw_out]

    # RoPE at position 0 (no-op for position 0, but let's compute it anyway)
    rotary_dims = int(layer_head_dim * partial_factor)
    half_rot = rotary_dims // 2
    for h in range(nh):
        base = h * layer_head_dim
        for d in range(half_rot):
            freq = 1.0 / rope_theta ** (2.0 * d / layer_head_dim)
            angle = 0.0 * freq
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            x0 = q[base + d]
            x1 = q[base + d + half_rot]
            q[base + d] = x0 * cos_a - x1 * sin_a
            q[base + d + half_rot] = x0 * sin_a + x1 * cos_a
    for d in range(half_rot):
        freq = 1.0 / rope_theta ** (2.0 * d / layer_head_dim)
        angle = 0.0 * freq
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        x0 = k[d]
        x1 = k[d + half_rot]
        k[d] = x0 * cos_a - x1 * sin_a
        k[d + half_rot] = x0 * sin_a + x1 * cos_a

    # Attention (1 token, scale=1.0)
    attn_out = [0.0] * layer_q_dim
    for h in range(nh):
        qh = q[h*layer_head_dim:(h+1)*layer_head_dim]
        vh = v[:layer_kv_dim]
        for d in range(layer_head_dim):
            attn_out[h*layer_head_dim+d] = vh[d]

    # O projection: attn_out @ o_raw.T
    output = [sum(attn_out[d] * o_raw[j*layer_q_dim+d] for d in range(layer_q_dim)) for j in range(hd)]

    # Post-attention norm + residual
    output_normed = rms_norm(output, post_attn_norm, hd)
    hidden = [residual[d] + output_normed[d] for d in range(hd)]
    l2 = math.sqrt(sum(v*v for v in hidden))
    print(f"  layer {layer_idx} after_attn_residual: l2={l2:.4f}")

    # --- FFN ---
    residual2 = hidden[:]
    normed2 = rms_norm(hidden, pre_ffn_norm, hd)

    # Get intermediate_dim from gate shape
    inter_dim = len(gate_raw) // hd

    # gate/up projections
    gate_out = [sum(normed2[d] * gate_raw[j*hd+d] for d in range(hd)) for j in range(inter_dim)]
    gate_gelu = [gelu_tanh(v) for v in gate_out]
    up_out = [sum(normed2[d] * up_raw[j*hd+d] for d in range(hd)) for j in range(inter_dim)]
    combined = [gate_gelu[j] * up_out[j] for j in range(inter_dim)]
    ffn_out = [sum(combined[d] * down_raw[j*inter_dim+d] for d in range(inter_dim)) for j in range(hd)]

    l2 = math.sqrt(sum(v*v for v in ffn_out))
    print(f"  layer {layer_idx} ffn_out_before_norm: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in ffn_out[:10])}]")

    ffn_normed = rms_norm(ffn_out, post_ffn_norm, hd)
    l2 = math.sqrt(sum(v*v for v in ffn_normed))
    print(f"  layer {layer_idx} ffn_out_after_norm: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in ffn_normed[:10])}]")

    # Residual
    hidden = [residual2[d] + ffn_normed[d] for d in range(hd)]

    # PLE (skip — too slow and we want to compare with PLE-disabled Rust run)

    # Layer scalar
    l2 = math.sqrt(sum(v*v for v in hidden))
    print(f"  layer {layer_idx} before_layer_scalar: ls={ls:.6f} l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")
    hidden = [v * ls for v in hidden]
    l2 = math.sqrt(sum(v*v for v in hidden))
    print(f"  layer {layer_idx} after_layer_scalar: l2={l2:.4f} first10=[{', '.join(f'{v:.6f}' for v in hidden[:10])}]")

    return hidden

# Process 2 layers (skip PLE)
for li in range(2):
    hidden = forward_layer(li, hidden)
