pub mod matmul;
pub mod softmax;
pub mod elementwise;
pub mod rmsnorm;
pub mod moe;
pub mod rope;
pub mod im2col;
pub mod fused_patch_embed;
pub mod tome_merge;
pub mod flash_decode;
pub mod causal_mask;
pub mod prefill_attn;
pub mod turboquant_kernels;
pub mod fft;
pub mod conv3d;
pub mod immediates;
pub mod ternary_matmul;
pub mod sparse_ternary;
pub mod optimizer_scale;
pub mod paged_attention;
pub mod hull_kv;
pub mod coop_matmul;

pub mod gpu_transformer;

pub use turboquant_kernels::TURBOQUANT_WGSL;
pub use fused_patch_embed::FusedPatchEmbedOp;

#[cfg(test)]
mod wgsl_validation_tests {
    /// Validate a WGSL shader source using naga (no GPU needed).
    /// Catches parse errors, undefined identifiers, type mismatches, etc.
    fn validate_wgsl(label: &str, source: &str) {
        let module = match naga::front::wgsl::parse_str(source) {
            Ok(m) => m,
            Err(e) => panic!(
                "WGSL parse error in '{}': {}",
                label, e.emit_to_string(source)
            ),
        };
        // Also run validation to catch semantic errors
        if let Err(e) = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        ).validate(&module) {
            panic!(
                "WGSL validation error in '{}': {}",
                label, e.emit_to_string(source)
            );
        }
    }

    // --- matmul.rs ---
    #[test] fn test_matmul_shader() { validate_wgsl("SHADER", super::matmul::SHADER); }
    #[test] fn test_matmul_double_buffer() { validate_wgsl("SHADER_DOUBLE_BUF", super::matmul::SHADER_DOUBLE_BUF); }
    #[test] fn test_matmul_transpose_b() { validate_wgsl("MATMUL_TRANSPOSE_B_WGSL", super::matmul::MATMUL_TRANSPOSE_B_WGSL); }
    #[test] fn test_matmul_transpose_a() { validate_wgsl("MATMUL_TRANSPOSE_A_WGSL", super::matmul::MATMUL_TRANSPOSE_A_WGSL); }

    // --- softmax.rs ---
    #[test] fn test_softmax() { validate_wgsl("SOFTMAX_WGSL", super::softmax::SOFTMAX_WGSL); }
    #[test] fn test_softmax_immediate() { validate_wgsl("SOFTMAX_IMMEDIATE_WGSL", super::softmax::SOFTMAX_IMMEDIATE_WGSL); }
    #[test] fn test_softmax_backward() { validate_wgsl("SOFTMAX_BACKWARD_WGSL", super::softmax::SOFTMAX_BACKWARD_WGSL); }

    // --- elementwise.rs ---
    #[test] fn test_elementwise() { validate_wgsl("ELEMENTWISE_WGSL", super::elementwise::ELEMENTWISE_WGSL); }

    // --- rmsnorm.rs ---
    #[test] fn test_rmsnorm() { validate_wgsl("RMSNORM_WGSL", super::rmsnorm::RMSNORM_WGSL); }
    #[test] fn test_rmsnorm_backward() { validate_wgsl("RMSNORM_BACKWARD_WGSL", super::rmsnorm::RMSNORM_BACKWARD_WGSL); }

    // --- rope.rs ---
    #[test] fn test_rope() { validate_wgsl("ROPE_WGSL", super::rope::ROPE_WGSL); }
    #[test] fn test_rope_inplace() { validate_wgsl("ROPE_INPLACE_WGSL", super::rope::ROPE_INPLACE_WGSL); }
    #[test] fn test_yarn_rope() { validate_wgsl("YARN_ROPE_WGSL", super::rope::YARN_ROPE_WGSL); }
    #[test] fn test_rope_backward() { validate_wgsl("ROPE_BACKWARD_WGSL", super::rope::ROPE_BACKWARD_WGSL); }

    // --- flash_decode.rs ---
    #[test] fn test_flash_decode() { validate_wgsl("FLASH_DECODE_WGSL", super::flash_decode::FLASH_DECODE_WGSL); }
    #[test] fn test_flash_decode_tiled() { validate_wgsl("FLASH_DECODE_TILED_WGSL", super::flash_decode::FLASH_DECODE_TILED_WGSL); }
    #[test] fn test_flash_decode_subgroup() { validate_wgsl("FLASH_DECODE_SUBGROUP_WGSL", super::flash_decode::FLASH_DECODE_SUBGROUP_WGSL); }

    // --- causal_mask.rs ---
    #[test] fn test_causal_mask() { validate_wgsl("CAUSAL_MASK_WGSL", super::causal_mask::CAUSAL_MASK_WGSL); }

    // --- hull_kv.rs ---
    #[test] fn test_hull_project_2d() { validate_wgsl("HULL_PROJECT_2D_WGSL", super::hull_kv::HULL_PROJECT_2D_WGSL); }
    #[test] fn test_hull_search() { validate_wgsl("HULL_SEARCH_WGSL", super::hull_kv::HULL_SEARCH_WGSL); }

    // --- ternary_matmul.rs ---
    #[test] fn test_ternary_matmul() { validate_wgsl("TERNARY_MATMUL_WGSL", super::ternary_matmul::TERNARY_MATMUL_WGSL); }

    // --- sparse_ternary.rs ---
    #[test] fn test_sparse_ternary() { validate_wgsl("SPARSE_TERNARY_MATMUL_WGSL", super::sparse_ternary::SPARSE_TERNARY_MATMUL_WGSL); }

    // --- paged_attention.rs ---
    #[test] fn test_paged_attention() { validate_wgsl("PAGED_ATTENTION_DECODE_WGSL", super::paged_attention::PAGED_ATTENTION_DECODE_WGSL); }

    // --- coop_matmul.rs ---
    #[test] fn test_tiled_matmul() { validate_wgsl("TILED_MATMUL_WGSL", super::coop_matmul::TILED_MATMUL_WGSL); }
    #[test] fn test_coop_matmul() { validate_wgsl("COOP_MATMUL_WGSL", super::coop_matmul::COOP_MATMUL_WGSL); }

    // --- optimizer_scale.rs ---
    #[test] fn test_column_norms() { validate_wgsl("COLUMN_NORMS_WGSL", super::optimizer_scale::COLUMN_NORMS_WGSL); }
    #[test] fn test_scale_update() { validate_wgsl("SCALE_UPDATE_WGSL", super::optimizer_scale::SCALE_UPDATE_WGSL); }
    #[test] fn test_scale_momentum() { validate_wgsl("SCALE_MOMENTUM_WGSL", super::optimizer_scale::SCALE_MOMENTUM_WGSL); }

    // --- gpu_transformer.rs ---
    #[test] fn test_rmsnorm_weighted() { validate_wgsl("RMSNORM_WEIGHTED_WGSL", super::gpu_transformer::RMSNORM_WEIGHTED_WGSL); }
    #[test] fn test_silu_multiply() { validate_wgsl("SILU_MULTIPLY_WGSL", super::gpu_transformer::SILU_MULTIPLY_WGSL); }
    #[test] fn test_residual_add() { validate_wgsl("RESIDUAL_ADD_WGSL", super::gpu_transformer::RESIDUAL_ADD_WGSL); }
    #[test] fn test_embedding_gather() { validate_wgsl("EMBEDDING_GATHER_WGSL", super::gpu_transformer::EMBEDDING_GATHER_WGSL); }
    #[test] fn test_causal_attention_gqa() { validate_wgsl("CAUSAL_ATTENTION_GQA_WGSL", super::gpu_transformer::CAUSAL_ATTENTION_GQA_WGSL); }
    #[test] fn test_gpu_rope() { validate_wgsl("ROPE_WGSL", super::gpu_transformer::ROPE_WGSL); }

    // --- moe.rs ---
    #[test] fn test_moe_gating() { validate_wgsl("MOE_GATING_WGSL", super::moe::MOE_GATING_WGSL); }
    #[test] fn test_moe_dispatch() { validate_wgsl("MOE_DISPATCH_WGSL", super::moe::MOE_DISPATCH_WGSL); }
    #[test] fn test_moe_gather() { validate_wgsl("MOE_GATHER_WGSL", super::moe::MOE_GATHER_WGSL); }

    // --- Other kernels ---
    #[test] fn test_fft() { validate_wgsl("FFT_WGSL", super::fft::FFT_WGSL); }
    #[test] fn test_im2col() { validate_wgsl("IM2COL_WGSL", super::im2col::IM2COL_WGSL); }
    #[test] fn test_fused_patch_embed() { validate_wgsl("FUSED_PATCH_EMBED_WGSL", super::fused_patch_embed::FUSED_PATCH_EMBED_WGSL); }
    #[test] fn test_tome_merge() { validate_wgsl("TOME_MERGE_WGSL", super::tome_merge::TOME_MERGE_WGSL); }
    #[test] fn test_prefill_attn() { validate_wgsl("PREFILL_ATTN_WGSL", super::prefill_attn::PREFILL_ATTN_WGSL); }
    // turboquant_kernels.rs uses pseudo-WGSL (not compilable, skip)
    // #[test] fn test_turboquant() { validate_wgsl("TURBOQUANT_WGSL", super::turboquant_kernels::TURBOQUANT_WGSL); }
    #[test] fn test_temporal_conv() { validate_wgsl("TEMPORAL_CONV_WGSL", super::conv3d::TEMPORAL_CONV_WGSL); }
    #[test] fn test_spatial_conv() { validate_wgsl("SPATIAL_CONV_WGSL", super::conv3d::SPATIAL_CONV_WGSL); }
}

