"""Full 35-layer forward pass for single BOS token using numpy.
Compares final logits with Rust output to pinpoint where computation diverges."""
import struct, json, math, sys
import numpy as np

path = "/home/shift/model.safetensors"
with open(path, "rb") as f:
    hdr_len = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(hdr_len))
    ds = 8 + hdr_len

def get_tensor(name):
    info = header[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(ds + s)
        raw = f.read(e - s)
    shape = info["shape"]
    vals = np.empty(len(raw) // 2, dtype=np.float32)
    for i in range(len(raw) // 2):
        h = struct.unpack_from("<H", raw, i * 2)[0]
        vals[i] = struct.unpack("<f", struct.pack("<I", h << 16))[0]
    return vals.reshape(shape)

# Load config values
hd = 1536
nh = 8
nkv = 1
ple_dim = 256
num_layers = 35
vocab_size = 262144
final_logit_softcapping = 30.0

layer_types = ['sliding_attention']*4 + ['full_attention'] + ['sliding_attention']*4 + ['full_attention'] + \
              ['sliding_attention']*4 + ['full_attention'] + ['sliding_attention']*4 + ['full_attention'] + \
              ['sliding_attention']*4 + ['full_attention'] + ['sliding_attention']*4 + ['full_attention'] + \
              ['sliding_attention']*4 + ['full_attention'] + ['sliding_attention']*4 + ['full_attention']

def rms_norm(x, w, eps=1e-6):
    """x: array [..., dim], w: array [dim]"""
    ms = np.mean(x ** 2, axis=-1, keepdims=True) + eps
    return x / np.sqrt(ms) * w

def gelu_tanh(x):
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + np.tanh(inner))

# Embedding for BOS (token 2)
embed = get_tensor("model.language_model.embed_tokens.weight")  # [262144, 1536]
hidden = embed[2:3] * np.sqrt(hd)  # [1, 1536]
print(f"After embed+scale: l2={np.linalg.norm(hidden):.4f} first5={hidden[0,:5].tolist()}")

# Load PLE model weights
ple_model_proj = get_tensor("model.language_model.per_layer_model_projection.weight")  # [8960, 1536]
ple_proj_norm = get_tensor("model.language_model.per_layer_projection_norm.weight")  # [256]
embed_per_layer = get_tensor("model.language_model.embed_tokens_per_layer.weight")  # [262144, 8960]
ple_total = num_layers * ple_dim  # 8960
ple_inputs_precomputed = None  # Will be set on layer 0

# Load final norm + lm_head
final_norm = get_tensor("model.language_model.norm.weight")  # [1536]
lm_head = embed  # tied weights [262144, 1536]

token_id = 2  # BOS
initial_hidden = hidden.copy()  # Save for PLE pre-computation

for layer_idx in range(num_layers):
    is_full = layer_types[layer_idx] == "full_attention"
    layer_head_dim = 512 if is_full else 256
    layer_q_dim = nh * layer_head_dim
    layer_kv_dim = nkv * layer_head_dim
    rope_theta = 1000000.0 if is_full else 10000.0
    partial_factor = 0.25 if is_full else 1.0
    rotary_dims = int(layer_head_dim * partial_factor)

    prefix = f"model.language_model.layers.{layer_idx}"
    
    # Load layer weights
    in_norm_w = get_tensor(f"{prefix}.input_layernorm.weight")
    q_w = get_tensor(f"{prefix}.self_attn.q_proj.weight")
    k_w = get_tensor(f"{prefix}.self_attn.k_proj.weight")
    v_w = get_tensor(f"{prefix}.self_attn.v_proj.weight")
    o_w = get_tensor(f"{prefix}.self_attn.o_proj.weight")
    q_norm_w = get_tensor(f"{prefix}.self_attn.q_norm.weight")
    k_norm_w = get_tensor(f"{prefix}.self_attn.k_norm.weight")
    post_attn_norm_w = get_tensor(f"{prefix}.post_attention_layernorm.weight")
    pre_ffn_norm_w = get_tensor(f"{prefix}.pre_feedforward_layernorm.weight")
    gate_w = get_tensor(f"{prefix}.mlp.gate_proj.weight")
    up_w = get_tensor(f"{prefix}.mlp.up_proj.weight")
    down_w = get_tensor(f"{prefix}.mlp.down_proj.weight")
    post_ffn_norm_w = get_tensor(f"{prefix}.post_feedforward_layernorm.weight")
    ls_val = get_tensor(f"{prefix}.layer_scalar")[0]
    
    inter_dim = gate_w.shape[0]

    # === ATTENTION ===
    residual = hidden.copy()
    normed = rms_norm(hidden, in_norm_w)  # [1, hd]

    # Q/K/V projections: y = x @ W.T
    q = normed @ q_w.T  # [1, q_dim]
    k = normed @ k_w.T  # [1, kv_dim]
    v = normed @ v_w.T  # [1, kv_dim]

    # Per-head Q norm
    q = q.reshape(1, nh, layer_head_dim)
    ms = np.mean(q ** 2, axis=-1, keepdims=True) + 1e-6
    q = q / np.sqrt(ms) * q_norm_w[np.newaxis, np.newaxis, :]
    q = q.reshape(1, layer_q_dim)

    # K norm
    k = k.reshape(1, nkv, layer_head_dim)
    ms = np.mean(k ** 2, axis=-1, keepdims=True) + 1e-6
    k = k / np.sqrt(ms) * k_norm_w[np.newaxis, np.newaxis, :]
    k = k.reshape(1, layer_kv_dim)

    # V norm (no scale)
    v = v.reshape(1, nkv, layer_head_dim)
    ms = np.mean(v ** 2, axis=-1, keepdims=True) + 1e-6
    v = v / np.sqrt(ms)
    v = v.reshape(1, layer_kv_dim)

    # RoPE at position 0 (cos=1, sin=0 for all dims → no-op)
    # Skip for speed

    # Attention: 1 token, scale=1.0
    # For 1 token, softmax([score]) = [1.0], so attn_out = V
    attn_out = np.zeros((1, layer_q_dim))
    for h in range(nh):
        attn_out[0, h*layer_head_dim:(h+1)*layer_head_dim] = v[0, :layer_kv_dim]

    # O projection
    output = attn_out @ o_w.T  # [1, hd]

    # Post-attention norm + residual
    output = rms_norm(output, post_attn_norm_w)
    hidden = residual + output

    # === FFN ===
    residual2 = hidden.copy()
    normed2 = rms_norm(hidden, pre_ffn_norm_w)

    gate_out = gelu_tanh(normed2 @ gate_w.T)  # [1, inter_dim]
    up_out = normed2 @ up_w.T  # [1, inter_dim]
    ffn_out = (gate_out * up_out) @ down_w.T  # [1, hd]

    # Post-FFN norm + residual
    ffn_out = rms_norm(ffn_out, post_ffn_norm_w)
    hidden = residual2 + ffn_out

    # === PLE ===
    # IMPORTANT: PLE uses pre-computed inputs from the INITIAL hidden state.
    # project_per_layer_inputs runs ONCE before the loop in llama.cpp.
    if layer_idx == 0:
        ple_proj_all = (initial_hidden @ ple_model_proj.T) * (1.0 / np.sqrt(hd))
        ple_proj_all = ple_proj_all.reshape(1, num_layers, ple_dim)
        
        # Per-layer RMS norm on context projection
        ple_normed = np.zeros_like(ple_proj_all)
        for l in range(num_layers):
            sl = ple_proj_all[0, l, :]
            ms = np.mean(sl ** 2) + 1e-6
            ple_normed[0, l, :] = sl / np.sqrt(ms) * ple_proj_norm
        
        # Add token embeddings scaled by sqrt(ple_dim)
        tok_emb_all = embed_per_layer[token_id].reshape(num_layers, ple_dim) * np.sqrt(ple_dim)
        ple_normed[0, :, :] += tok_emb_all
        
        # Scale by 1/sqrt(2)
        ple_normed *= 1.0 / np.sqrt(2.0)
        
        ple_inputs_precomputed = ple_normed[0]  # [num_layers, ple_dim]
    
    ple_input = ple_inputs_precomputed[layer_idx]  # [ple_dim]
    
    # Gate: hidden @ gate_w → gelu
    ple_gate_w = get_tensor(f"{prefix}.per_layer_input_gate.weight")  # [ple_dim, hd]
    ple_proj_w = get_tensor(f"{prefix}.per_layer_projection.weight")  # [hd, ple_dim]... wait, need to check
    ple_post_norm_w = get_tensor(f"{prefix}.post_per_layer_input_norm.weight")  # [hd]
    
    gate_out_ple = gelu_tanh(hidden @ ple_gate_w.T)  # [1, ple_dim]
    gated = gate_out_ple[0] * ple_input  # [ple_dim]
    
    # Project back
    proj_out_ple = gated @ ple_proj_w.T  # [1, hd]... or need to check shape
    
    # Post norm + residual
    proj_out_ple = rms_norm(proj_out_ple.reshape(1, hd), ple_post_norm_w)
    hidden = hidden + proj_out_ple

    # Layer scalar
    hidden = hidden * ls_val

    if layer_idx <= 3 or layer_idx == 4 or layer_idx == 34:
        l2 = np.linalg.norm(hidden)
        print(f"Layer {layer_idx}: l2={l2:.4f} ls={ls_val:.6f} first5={hidden[0,:5].tolist()}")

# Final norm
hidden = rms_norm(hidden, final_norm)
l2 = np.linalg.norm(hidden)
print(f"\nAfter final norm: l2={l2:.4f} first5={hidden[0,:5].tolist()}")

# LM head (tied weights)
logits = hidden @ lm_head.T  # [1, 262144]

# Softcapping
logits = np.tanh(logits / final_logit_softcapping) * final_logit_softcapping

# Stats
last_logits = logits[0]
top_indices = np.argsort(last_logits)[::-1][:10]
print(f"\nTop 10 tokens after softcapping:")
for idx in top_indices:
    print(f"  Token {idx}: {last_logits[idx]:.4f}")

# Entropy
max_l = np.max(last_logits)
probs = np.exp(last_logits - max_l)
probs /= np.sum(probs)
entropy = -np.sum(probs * np.log(probs + 1e-30))
print(f"\nEntropy: {entropy:.4f}")
print(f"Logit range: [{np.min(last_logits):.4f}, {np.max(last_logits):.4f}]")

# Decode top tokens
import json as json_mod
try:
    with open("/home/shift/code/ferrisres/tokenizer.json") as tf:
        tok_data = json_mod.load(tf)
    vocab = tok_data["model"]["vocab"]
    id_to_token = {v: k for k, v in vocab.items()}
    print(f"\nTop tokens decoded:")
    for idx in top_indices[:5]:
        tok_str = id_to_token.get(idx, f"<UNK:{idx}>")
        print(f"  {idx}: '{tok_str}'")
except:
    pass
