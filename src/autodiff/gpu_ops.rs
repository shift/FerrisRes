//! GPU kernels for the graph ops the reference transformer needs.
//!
//! Every op here has a CPU-reference test (`tests/gpu_train_ops.rs`). The
//! backward shaders are the transpose of the forward data flow, not a copy of
//! an untested kernel: the parity test in `tests/gpu_train_parity.rs` checks
//! the whole graph against [`crate::training::f32_trainer`].
//!
//! Layout conventions match the CPU reference:
//! * activations are row-major `[rows, cols]`, weights `[in, out]`,
//! * Q/K/V heads are time-major: row `t * heads + h`, then `head_dim`,
//! * RoPE is the half-split rotation `(d, d + head_dim/2)`.

use std::sync::Arc;

use wgpu::{BindGroupLayout, BufferDescriptor, BufferUsages, ComputePipeline, Device, Queue};

use crate::compute::buffer::GpuBuffer;
use crate::error::Result;

struct Kernel {
    pipeline: ComputePipeline,
    layout: BindGroupLayout,
    /// Number of storage bindings; the uniform params buffer, when present,
    /// always takes the next binding.
    storage: usize,
}

fn kernel(
    device: &Device,
    label: &str,
    wgsl: &str,
    entry: &str,
    storage: &[bool],
    has_params: bool,
) -> Kernel {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });

    let mut entries = Vec::with_capacity(storage.len() + 1);
    for (binding, read_only) in storage.iter().enumerate() {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: *read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    if has_params {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: storage.len() as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some(entry),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    Kernel {
        pipeline,
        layout,
        storage: storage.len(),
    }
}

fn groups(n: u32, size: u32) -> u32 {
    if n == 0 {
        0
    } else {
        (n + size - 1) / size
    }
}

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------

const RMSNORM_FWD: &str = r#"
struct P { dim: u32, rows: u32, eps: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) row: vec3<u32>) {
    let c = lid.x;
    var ss = 0.0;
    for (var i = c; i < p.dim; i = i + 256u) {
        let v = x[row.x * p.dim + i];
        ss = ss + v * v;
    }
    sh[c] = ss;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (c < stride) { sh[c] = sh[c] + sh[c + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let inv = inverseSqrt(sh[0] / f32(p.dim) + p.eps);
    for (var i = c; i < p.dim; i = i + 256u) {
        let idx = row.x * p.dim + i;
        y[idx] = x[idx] * inv;
    }
}
"#;

const RMSNORM_BWD: &str = r#"
struct P { dim: u32, rows: u32, eps: f32, _pad: u32 }
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> ssq: array<f32, 256>;
var<workgroup> sdx: array<f32, 256>;
var<workgroup> inv_shared: f32;
var<workgroup> c_shared: f32;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) row: vec3<u32>) {
    let c = lid.x;
    var a = 0.0;
    var b = 0.0;
    for (var i = c; i < p.dim; i = i + 256u) {
        let idx = row.x * p.dim + i;
        let v = x[idx];
        a = a + v * v;
        b = b + dy[idx] * v;
    }
    ssq[c] = a;
    sdx[c] = b;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (c < stride) {
            ssq[c] = ssq[c] + ssq[c + stride];
            sdx[c] = sdx[c] + sdx[c + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (c == 0u) {
        let inv = inverseSqrt(ssq[0] / f32(p.dim) + p.eps);
        inv_shared = inv;
        // mean(dy * x_hat) = inv * mean(dy * x)
        c_shared = inv * inv * inv * sdx[0] / f32(p.dim);
    }
    workgroupBarrier();
    let inv = inv_shared;
    let cc = c_shared;
    for (var i = c; i < p.dim; i = i + 256u) {
        let idx = row.x * p.dim + i;
        dx[idx] = inv * dy[idx] - cc * x[idx];
    }
}
"#;

const MUL_BCAST_FWD: &str = r#"
struct P { numel: u32, cols: u32, rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    out[i] = a[i] * w[i % p.cols];
}
"#;

const MUL_BCAST_GRAD_A: &str = r#"
struct P { numel: u32, cols: u32, rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> da: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    da[i] = dout[i] * w[i % p.cols];
}
"#;

const MUL_BCAST_GRAD_W: &str = r#"
struct P { numel: u32, cols: u32, rows: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> dw: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= p.cols) { return; }
    var s = 0.0;
    for (var r = 0u; r < p.rows; r = r + 1u) {
        s = s + dout[r * p.cols + c] * a[r * p.cols + c];
    }
    dw[c] = s;
}
"#;

const SILU_MUL_FWD: &str = r#"
struct P { numel: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    let g = gate[i];
    out[i] = (g / (1.0 + exp(-g))) * up[i];
}
"#;

const SILU_MUL_BWD: &str = r#"
struct P { numel: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> dout: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgate: array<f32>;
@group(0) @binding(4) var<storage, read_write> dup: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    let g = gate[i];
    let s = 1.0 / (1.0 + exp(-g));
    let silu = g * s;
    // silu'(g) = s * (1 + g * (1 - s))
    dgate[i] = dout[i] * up[i] * s * (1.0 + g * (1.0 - s));
    dup[i] = dout[i] * silu;
}
"#;

const ROPE_FWD: &str = r#"
struct P { seq: u32, heads: u32, head_dim: u32, offset: u32, theta: f32, _a: f32, _b: f32, _c: f32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.head_dim / 2u;
    let total = p.seq * p.heads * half;
    let i = gid.x;
    if (i >= total) { return; }
    let d = i % half;
    let rest = i / half;
    let h = rest % p.heads;
    let t = rest / p.heads;
    let base = (t * p.heads + h) * p.head_dim;
    let freq = pow(p.theta, -2.0 * f32(d) / f32(p.head_dim));
    let angle = (f32(t) + f32(p.offset)) * freq;
    let c = cos(angle);
    let s = sin(angle);
    let a = x[base + d];
    let b = x[base + d + half];
    out[base + d] = a * c - b * s;
    out[base + d + half] = b * c + a * s;
}
"#;

const ROPE_BWD: &str = r#"
struct P { seq: u32, heads: u32, head_dim: u32, offset: u32, theta: f32, _a: f32, _b: f32, _c: f32 }
@group(0) @binding(0) var<storage, read> dx: array<f32>;
@group(0) @binding(1) var<storage, read_write> dout: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half = p.head_dim / 2u;
    let total = p.seq * p.heads * half;
    let i = gid.x;
    if (i >= total) { return; }
    let d = i % half;
    let rest = i / half;
    let h = rest % p.heads;
    let t = rest / p.heads;
    let base = (t * p.heads + h) * p.head_dim;
    let freq = pow(p.theta, -2.0 * f32(d) / f32(p.head_dim));
    let angle = (f32(t) + f32(p.offset)) * freq;
    let c = cos(angle);
    let s = sin(angle);
    let da = dx[base + d];
    let db = dx[base + d + half];
    dout[base + d] = da * c + db * s;
    dout[base + d + half] = -da * s + db * c;
}
"#;

const GQA_EXPAND_FWD: &str = r#"
struct P { seq: u32, heads: u32, kv_heads: u32, head_dim: u32 }
@group(0) @binding(0) var<storage, read> kv: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.seq * p.heads * p.head_dim;
    let i = gid.x;
    if (i >= total) { return; }
    let d = i % p.head_dim;
    let rest = i / p.head_dim;
    let h = rest % p.heads;
    let t = rest / p.heads;
    let group = p.heads / p.kv_heads;
    let kvh = h / group;
    out[i] = kv[(t * p.kv_heads + kvh) * p.head_dim + d];
}
"#;

const GQA_EXPAND_BWD: &str = r#"
struct P { seq: u32, heads: u32, kv_heads: u32, head_dim: u32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read_write> dkv: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.seq * p.kv_heads * p.head_dim;
    let i = gid.x;
    if (i >= total) { return; }
    let d = i % p.head_dim;
    let rest = i / p.head_dim;
    let kvh = rest % p.kv_heads;
    let t = rest / p.kv_heads;
    let group = p.heads / p.kv_heads;
    var s = 0.0;
    for (var g = 0u; g < group; g = g + 1u) {
        let h = kvh * group + g;
        s = s + dout[(t * p.heads + h) * p.head_dim + d];
    }
    dkv[i] = s;
}
"#;

const ATTENTION_FWD: &str = r#"
struct P { seq: u32, heads: u32, head_dim: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k: array<f32>;
@group(0) @binding(2) var<storage, read> v: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let h = gid.y;
    if (t >= p.seq || h >= p.heads) { return; }
    let base = (t * p.heads + h) * p.head_dim;
    var mx = -1.0e30;
    for (var s = 0u; s <= t; s = s + 1u) {
        let kb = (s * p.heads + h) * p.head_dim;
        var dot = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + q[base + d] * k[kb + d];
        }
        mx = max(mx, dot * p.scale);
    }
    var sum = 0.0;
    for (var s = 0u; s <= t; s = s + 1u) {
        let kb = (s * p.heads + h) * p.head_dim;
        var dot = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + q[base + d] * k[kb + d];
        }
        sum = sum + exp(dot * p.scale - mx);
    }
    for (var d = 0u; d < p.head_dim; d = d + 1u) {
        var acc = 0.0;
        for (var s = 0u; s <= t; s = s + 1u) {
            let kb = (s * p.heads + h) * p.head_dim;
            var dot = 0.0;
            for (var i = 0u; i < p.head_dim; i = i + 1u) {
                dot = dot + q[base + i] * k[kb + i];
            }
            let prob = exp(dot * p.scale - mx) / sum;
            acc = acc + prob * v[kb + d];
        }
        out[base + d] = acc;
    }
}
"#;

const ATTENTION_BWD_A: &str = r#"
struct P { seq: u32, heads: u32, head_dim: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> dq: array<f32>;
@group(0) @binding(5) var<storage, read_write> probs: array<f32>;
@group(0) @binding(6) var<storage, read_write> ds: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let h = gid.y;
    if (t >= p.seq || h >= p.heads) { return; }
    let base = (t * p.heads + h) * p.head_dim;
    let row = (h * p.seq + t) * p.seq;
    var mx = -1.0e30;
    for (var s = 0u; s <= t; s = s + 1u) {
        let kb = (s * p.heads + h) * p.head_dim;
        var dot = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + q[base + d] * k[kb + d];
        }
        mx = max(mx, dot * p.scale);
    }
    var sum = 0.0;
    for (var s = 0u; s <= t; s = s + 1u) {
        let kb = (s * p.heads + h) * p.head_dim;
        var dot = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + q[base + d] * k[kb + d];
        }
        sum = sum + exp(dot * p.scale - mx);
    }
    // First pass: write p and dp, and accumulate sum_s(p * dp) for the
    // softmax Jacobian. dp is recomputed in the second pass, so no per-row
    // scratch array is needed and the kernel has no sequence-length ceiling.
    var weighted = 0.0;
    for (var s = 0u; s <= t; s = s + 1u) {
        let kb = (s * p.heads + h) * p.head_dim;
        var dot = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dot = dot + q[base + d] * k[kb + d];
        }
        let prob = exp(dot * p.scale - mx) / sum;
        probs[row + s] = prob;
        var dpv = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dpv = dpv + dout[base + d] * v[kb + d];
        }
        weighted = weighted + prob * dpv;
    }
    for (var s = 0u; s <= t; s = s + 1u) {
        let kb = (s * p.heads + h) * p.head_dim;
        var dpv = 0.0;
        for (var d = 0u; d < p.head_dim; d = d + 1u) {
            dpv = dpv + dout[base + d] * v[kb + d];
        }
        ds[row + s] = probs[row + s] * (dpv - weighted);
    }
    for (var s = t + 1u; s < p.seq; s = s + 1u) {
        probs[row + s] = 0.0;
        ds[row + s] = 0.0;
    }
    for (var d = 0u; d < p.head_dim; d = d + 1u) {
        var acc = 0.0;
        for (var s = 0u; s <= t; s = s + 1u) {
            let kb = (s * p.heads + h) * p.head_dim;
            acc = acc + ds[row + s] * p.scale * k[kb + d];
        }
        dq[base + d] = acc;
    }
}
"#;

const ATTENTION_BWD_B: &str = r#"
struct P { seq: u32, heads: u32, head_dim: u32, scale: f32 }
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read> probs: array<f32>;
@group(0) @binding(3) var<storage, read> ds: array<f32>;
@group(0) @binding(4) var<storage, read_write> dk: array<f32>;
@group(0) @binding(5) var<storage, read_write> dv: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    let h = gid.y;
    if (s >= p.seq || h >= p.heads) { return; }
    let base = (s * p.heads + h) * p.head_dim;
    for (var d = 0u; d < p.head_dim; d = d + 1u) {
        var dk_acc = 0.0;
        var dv_acc = 0.0;
        for (var t = s; t < p.seq; t = t + 1u) {
            let tb = (t * p.heads + h) * p.head_dim;
            let row = (h * p.seq + t) * p.seq;
            dk_acc = dk_acc + ds[row + s] * p.scale * q[tb + d];
            dv_acc = dv_acc + probs[row + s] * dout[tb + d];
        }
        dk[base + d] = dk_acc;
        dv[base + d] = dv_acc;
    }
}
"#;

const EMBEDDING_FWD: &str = r#"
struct P { seq: u32, vocab: u32, hidden: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> table: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.seq * p.hidden;
    let i = gid.x;
    if (i >= total) { return; }
    let d = i % p.hidden;
    let t = i / p.hidden;
    out[i] = table[ids[t] * p.hidden + d];
}
"#;

const EMBEDDING_BWD: &str = r#"
struct P { seq: u32, vocab: u32, hidden: u32, _pad: u32 }
@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> dout: array<f32>;
@group(0) @binding(2) var<storage, read_write> dtable: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = p.vocab * p.hidden;
    let i = gid.x;
    if (i >= total) { return; }
    let v = i / p.hidden;
    let d = i % p.hidden;
    var s = 0.0;
    for (var t = 0u; t < p.seq; t = t + 1u) {
        if (ids[t] == v) { s = s + dout[t * p.hidden + d]; }
    }
    dtable[i] = s;
}
"#;

const LAYER_SCALE_FWD: &str = r#"
struct P { numel: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> scalar: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    out[i] = x[i] * scalar[0];
}
"#;

const LAYER_SCALE_BWD_X: &str = r#"
struct P { numel: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> scalar: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= p.numel) { return; }
    dx[i] = dout[i] * scalar[0];
}
"#;

const LAYER_SCALE_BWD_S_PARTIAL: &str = r#"
struct P { numel: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> dout: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let c = lid.x;
    var s = 0.0;
    for (var i = wg.x * 256u + c; i < p.numel; i = i + 256u) {
        s = s + dout[i] * x[i];
    }
    sh[c] = s;
    workgroupBarrier();
    var stride = 128u;
    while (stride > 0u) {
        if (c < stride) { sh[c] = sh[c] + sh[c + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (c == 0u) { partials[wg.x] = sh[0]; }
}
"#;

const LAYER_SCALE_BWD_S_FINAL: &str = r#"
struct P { n: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<storage, read> partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> dscalar: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(1)
fn main() {
    var s = 0.0;
    for (var i = 0u; i < p.n; i = i + 1u) { s = s + partials[i]; }
    dscalar[0] = s;
}
"#;

// ---------------------------------------------------------------------------
// TrainOps
// ---------------------------------------------------------------------------

/// Pipelines for the transformer-specific graph ops. Constructing once per
/// model keeps shader compilation out of the training step.
pub struct TrainOps {
    device: Arc<Device>,
    queue: Arc<Queue>,
    rmsnorm_fwd: Kernel,
    rmsnorm_bwd: Kernel,
    mul_bcast_fwd: Kernel,
    mul_bcast_grad_a: Kernel,
    mul_bcast_grad_w: Kernel,
    silu_mul_fwd: Kernel,
    silu_mul_bwd: Kernel,
    rope_fwd: Kernel,
    rope_bwd: Kernel,
    gqa_fwd: Kernel,
    gqa_bwd: Kernel,
    attention_fwd: Kernel,
    attention_bwd_a: Kernel,
    attention_bwd_b: Kernel,
    embedding_fwd: Kernel,
    embedding_bwd: Kernel,
    layer_scale_fwd: Kernel,
    layer_scale_bwd_x: Kernel,
    layer_scale_bwd_s_partial: Kernel,
    layer_scale_bwd_s_final: Kernel,
}

impl TrainOps {
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Self {
        let d = device.as_ref();
        Self {
            rmsnorm_fwd: kernel(d, "rmsnorm_no_scale_fwd", RMSNORM_FWD, "main", &[true, false], true),
            rmsnorm_bwd: kernel(d, "rmsnorm_no_scale_bwd", RMSNORM_BWD, "main", &[true, true, false], true),
            mul_bcast_fwd: kernel(d, "mul_bcast_fwd", MUL_BCAST_FWD, "main", &[true, true, false], true),
            mul_bcast_grad_a: kernel(d, "mul_bcast_grad_a", MUL_BCAST_GRAD_A, "main", &[true, true, false], true),
            mul_bcast_grad_w: kernel(d, "mul_bcast_grad_w", MUL_BCAST_GRAD_W, "main", &[true, true, false], true),
            silu_mul_fwd: kernel(d, "silu_mul_fwd", SILU_MUL_FWD, "main", &[true, true, false], true),
            silu_mul_bwd: kernel(d, "silu_mul_bwd", SILU_MUL_BWD, "main", &[true, true, true, false, false], true),
            rope_fwd: kernel(d, "rope_fwd", ROPE_FWD, "main", &[true, false], true),
            rope_bwd: kernel(d, "rope_bwd", ROPE_BWD, "main", &[true, false], true),
            gqa_fwd: kernel(d, "gqa_expand_fwd", GQA_EXPAND_FWD, "main", &[true, false], true),
            gqa_bwd: kernel(d, "gqa_expand_bwd", GQA_EXPAND_BWD, "main", &[true, false], true),
            attention_fwd: kernel(d, "attention_fwd", ATTENTION_FWD, "main", &[true, true, true, false], true),
            attention_bwd_a: kernel(d, "attention_bwd_a", ATTENTION_BWD_A, "main", &[true, true, true, true, false, false, false], true),
            attention_bwd_b: kernel(d, "attention_bwd_b", ATTENTION_BWD_B, "main", &[true, true, true, true, false, false], true),
            embedding_fwd: kernel(d, "embedding_fwd", EMBEDDING_FWD, "main", &[true, true, false], true),
            embedding_bwd: kernel(d, "embedding_bwd", EMBEDDING_BWD, "main", &[true, true, false], true),
            layer_scale_fwd: kernel(d, "layer_scale_fwd", LAYER_SCALE_FWD, "main", &[true, true, false], true),
            layer_scale_bwd_x: kernel(d, "layer_scale_bwd_x", LAYER_SCALE_BWD_X, "main", &[true, true, false], true),
            layer_scale_bwd_s_partial: kernel(d, "layer_scale_bwd_s_partial", LAYER_SCALE_BWD_S_PARTIAL, "main", &[true, true, false], true),
            layer_scale_bwd_s_final: kernel(d, "layer_scale_bwd_s_final", LAYER_SCALE_BWD_S_FINAL, "main", &[true, false], true),
            device,
            queue,
        }
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    fn params(&self, bytes: &[u8]) -> GpuBuffer {
        let size = bytes.len().max(16);
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("train_op_params"),
            size: size as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut padded = bytes.to_vec();
        padded.resize(size, 0);
        self.queue.write_buffer(&buffer, 0, &padded);
        GpuBuffer::from_existing(buffer, size)
    }

    fn dispatch(
        &self,
        kernel: &Kernel,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &[&GpuBuffer],
        params: &[u8],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        assert_eq!(buffers.len(), kernel.storage);
        let params_buf = self.params(params);
        let mut entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.buffer().as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: kernel.storage as u32,
            resource: params_buf.buffer().as_entire_binding(),
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("train_op_bind_group"),
            layout: &kernel.layout,
            entries: &entries,
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("train_op"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&kernel.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        drop(pass);
        Ok(())
    }

    pub fn rmsnorm_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        y: &GpuBuffer,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        let p = params_rms(dim, rows, eps);
        self.dispatch(&self.rmsnorm_fwd, enc, &[x, y], &p, (rows, 1, 1))
    }

    pub fn rmsnorm_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dy: &GpuBuffer,
        x: &GpuBuffer,
        dx: &GpuBuffer,
        rows: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        let p = params_rms(dim, rows, eps);
        self.dispatch(&self.rmsnorm_bwd, enc, &[dy, x, dx], &p, (rows, 1, 1))
    }

    pub fn mul_broadcast_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        a: &GpuBuffer,
        w: &GpuBuffer,
        out: &GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        let p = params_mul(rows * cols, cols, rows);
        self.dispatch(&self.mul_bcast_fwd, enc, &[a, w, out], &p, (groups(rows * cols, 256), 1, 1))
    }

    pub fn mul_broadcast_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dout: &GpuBuffer,
        a: &GpuBuffer,
        w: &GpuBuffer,
        da: &GpuBuffer,
        dw: &GpuBuffer,
        rows: u32,
        cols: u32,
    ) -> Result<()> {
        let p = params_mul(rows * cols, cols, rows);
        self.dispatch(&self.mul_bcast_grad_a, enc, &[dout, w, da], &p, (groups(rows * cols, 256), 1, 1))?;
        self.dispatch(&self.mul_bcast_grad_w, enc, &[dout, a, dw], &p, (groups(cols, 256), 1, 1))
    }

    pub fn silu_mul_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        out: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let p = params_n(numel);
        self.dispatch(&self.silu_mul_fwd, enc, &[gate, up, out], &p, (groups(numel, 256), 1, 1))
    }

    pub fn silu_mul_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        dout: &GpuBuffer,
        dgate: &GpuBuffer,
        dup: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let p = params_n(numel);
        self.dispatch(
            &self.silu_mul_bwd,
            enc,
            &[gate, up, dout, dgate, dup],
            &p,
            (groups(numel, 256), 1, 1),
        )
    }

    pub fn rope_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        out: &GpuBuffer,
        seq: u32,
        heads: u32,
        head_dim: u32,
        theta: f32,
        offset: u32,
    ) -> Result<()> {
        let p = params_rope(seq, heads, head_dim, theta, offset);
        let total = seq * heads * (head_dim / 2);
        self.dispatch(&self.rope_fwd, enc, &[x, out], &p, (groups(total, 64), 1, 1))
    }

    pub fn rope_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dx: &GpuBuffer,
        dout: &GpuBuffer,
        seq: u32,
        heads: u32,
        head_dim: u32,
        theta: f32,
        offset: u32,
    ) -> Result<()> {
        let p = params_rope(seq, heads, head_dim, theta, offset);
        let total = seq * heads * (head_dim / 2);
        self.dispatch(&self.rope_bwd, enc, &[dx, dout], &p, (groups(total, 64), 1, 1))
    }

    pub fn gqa_expand_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        kv: &GpuBuffer,
        out: &GpuBuffer,
        seq: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        let p = params_gqa(seq, heads, kv_heads, head_dim);
        let total = seq * heads * head_dim;
        self.dispatch(&self.gqa_fwd, enc, &[kv, out], &p, (groups(total, 256), 1, 1))
    }

    pub fn gqa_expand_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dout: &GpuBuffer,
        dkv: &GpuBuffer,
        seq: u32,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
    ) -> Result<()> {
        let p = params_gqa(seq, heads, kv_heads, head_dim);
        let total = seq * kv_heads * head_dim;
        self.dispatch(&self.gqa_bwd, enc, &[dout, dkv], &p, (groups(total, 256), 1, 1))
    }

    pub fn attention_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        out: &GpuBuffer,
        seq: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<()> {
        let p = params_attn(seq, heads, head_dim, scale);
        self.dispatch(&self.attention_fwd, enc, &[q, k, v, out], &p, (seq, heads, 1))
    }

    /// Backward over pre-expanded K/V. `probs` and `ds` are scratch buffers of
    /// `heads * seq * seq` floats; the caller owns them.
    pub fn attention_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dout: &GpuBuffer,
        q: &GpuBuffer,
        k: &GpuBuffer,
        v: &GpuBuffer,
        dq: &GpuBuffer,
        dk: &GpuBuffer,
        dv: &GpuBuffer,
        probs: &GpuBuffer,
        ds: &GpuBuffer,
        seq: u32,
        heads: u32,
        head_dim: u32,
        scale: f32,
    ) -> Result<()> {
        let p = params_attn(seq, heads, head_dim, scale);
        self.dispatch(&self.attention_bwd_a, enc, &[dout, q, k, v, dq, probs, ds], &p, (seq, heads, 1))?;
        self.dispatch(&self.attention_bwd_b, enc, &[q, dout, probs, ds, dk, dv], &p, (seq, heads, 1))
    }

    pub fn embedding_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        ids: &GpuBuffer,
        table: &GpuBuffer,
        out: &GpuBuffer,
        seq: u32,
        vocab: u32,
        hidden: u32,
    ) -> Result<()> {
        let p = params_emb(seq, vocab, hidden);
        let total = seq * hidden;
        self.dispatch(&self.embedding_fwd, enc, &[ids, table, out], &p, (groups(total, 256), 1, 1))
    }

    pub fn embedding_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        ids: &GpuBuffer,
        dout: &GpuBuffer,
        dtable: &GpuBuffer,
        seq: u32,
        vocab: u32,
        hidden: u32,
    ) -> Result<()> {
        let p = params_emb(seq, vocab, hidden);
        let total = vocab * hidden;
        self.dispatch(&self.embedding_bwd, enc, &[ids, dout, dtable], &p, (groups(total, 256), 1, 1))
    }

    pub fn layer_scale_forward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        x: &GpuBuffer,
        scalar: &GpuBuffer,
        out: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let p = params_n(numel);
        self.dispatch(&self.layer_scale_fwd, enc, &[x, scalar, out], &p, (groups(numel, 256), 1, 1))
    }

    pub fn layer_scale_backward(
        &self,
        enc: &mut wgpu::CommandEncoder,
        dout: &GpuBuffer,
        x: &GpuBuffer,
        scalar: &GpuBuffer,
        dx: &GpuBuffer,
        dscalar: &GpuBuffer,
        numel: u32,
    ) -> Result<()> {
        let p = params_n(numel);
        self.dispatch(&self.layer_scale_bwd_x, enc, &[dout, scalar, dx], &p, (groups(numel, 256), 1, 1))?;
        let n = groups(numel, 256);
        let partials = GpuBuffer::zeros(
            self.device.as_ref(),
            self.queue.as_ref(),
            n as usize * std::mem::size_of::<f32>(),
            Some("layer_scale_partials"),
        )?;
        self.dispatch(
            &self.layer_scale_bwd_s_partial,
            enc,
            &[dout, x, &partials],
            &p,
            (n, 1, 1),
        )?;
        self.dispatch(
            &self.layer_scale_bwd_s_final,
            enc,
            &[&partials, dscalar],
            &params_n(n),
            (1, 1, 1),
        )
    }
}

fn params_rms(dim: u32, rows: u32, eps: f32) -> Vec<u8> {
    [
        dim.to_le_bytes(),
        rows.to_le_bytes(),
        eps.to_le_bytes(),
        0u32.to_le_bytes(),
    ]
    .concat()
}

fn params_mul(numel: u32, cols: u32, rows: u32) -> Vec<u8> {
    [
        numel.to_le_bytes(),
        cols.to_le_bytes(),
        rows.to_le_bytes(),
        0u32.to_le_bytes(),
    ]
    .concat()
}

fn params_n(numel: u32) -> Vec<u8> {
    [
        numel.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
    ]
    .concat()
}

fn params_rope(seq: u32, heads: u32, head_dim: u32, theta: f32, offset: u32) -> Vec<u8> {
    [
        seq.to_le_bytes(),
        heads.to_le_bytes(),
        head_dim.to_le_bytes(),
        offset.to_le_bytes(),
        theta.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
        0u32.to_le_bytes(),
    ]
    .concat()
}

fn params_gqa(seq: u32, heads: u32, kv_heads: u32, head_dim: u32) -> Vec<u8> {
    [
        seq.to_le_bytes(),
        heads.to_le_bytes(),
        kv_heads.to_le_bytes(),
        head_dim.to_le_bytes(),
    ]
    .concat()
}

fn params_attn(seq: u32, heads: u32, head_dim: u32, scale: f32) -> Vec<u8> {
    [
        seq.to_le_bytes(),
        heads.to_le_bytes(),
        head_dim.to_le_bytes(),
        scale.to_le_bytes(),
    ]
    .concat()
}

fn params_emb(seq: u32, vocab: u32, hidden: u32) -> Vec<u8> {
    [
        seq.to_le_bytes(),
        vocab.to_le_bytes(),
        hidden.to_le_bytes(),
        0u32.to_le_bytes(),
    ]
    .concat()
}
