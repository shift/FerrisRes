use ferrisres::error::FerrisResError;
use ferrisres::model::attention_metadata::AttentionMetadata;
use ferrisres::model::gguf::{GgufFile, GgufValue};
use ferrisres::model::standard_transformer::GatedActivation;
use serde_json::json;

fn gguf() -> GgufFile {
    GgufFile {
        version: 3,
        tensor_count: 0,
        tensor_infos: Default::default(),
        data_offset: 0,
        alignment: 32,
        path: Default::default(),
        metadata: [
            (
                "general.architecture".into(),
                GgufValue::String("qwen2".into()),
            ),
            ("qwen2.embedding_length".into(), GgufValue::UInt32(4096)),
            ("qwen2.attention.head_count".into(), GgufValue::UInt32(32)),
            ("qwen2.attention.head_count_kv".into(), GgufValue::UInt32(8)),
        ]
        .into_iter()
        .collect(),
    }
}

#[test]
fn public_fields_are_revalidated_before_shape_checks_or_conversion() {
    let valid =
        AttentionMetadata::from_hf_json(&json!({"hidden_size":8,"num_attention_heads":2})).unwrap();
    for invalid in [
        AttentionMetadata {
            hidden_dim: 0,
            ..valid
        },
        AttentionMetadata {
            num_heads: 0,
            ..valid
        },
        AttentionMetadata {
            num_kv_heads: 0,
            ..valid
        },
        AttentionMetadata {
            head_dim: 0,
            ..valid
        },
        AttentionMetadata {
            head_dim: usize::MAX,
            ..valid
        },
    ] {
        assert!(invalid
            .validate_projection_shapes(&[8, 8], &[8, 8], &[8, 8])
            .is_err());
        assert!(invalid
            .standard_config(1, 16, 10, GatedActivation::Silu)
            .is_err());
    }
    assert!(valid
        .standard_config(0, 16, 10, GatedActivation::Silu)
        .is_err());
    let config = valid
        .standard_config(1, 16, 10, GatedActivation::GeluTanh)
        .unwrap();
    assert_eq!(config.ffn_activation, GatedActivation::GeluTanh);
}

#[test]
fn only_absent_optional_gguf_fields_receive_defaults() {
    let mut f = gguf();
    f.metadata.remove("qwen2.attention.head_count_kv");
    f.metadata
        .insert("qwen2.attention.head_count".into(), GgufValue::UInt64(32));
    assert_eq!(AttentionMetadata::from_gguf(&f).unwrap().num_kv_heads, 32);
    f.metadata
        .insert("qwen2.attention.head_count_kv".into(), GgufValue::UInt32(0));
    assert!(AttentionMetadata::from_gguf(&f).is_err());
}

#[test]
fn square_q_projection_does_not_determine_head_count() {
    let m = AttentionMetadata::from_hf_json(&json!({"hidden_size":4096,"num_attention_heads":32}))
        .unwrap();
    assert_eq!((m.num_heads, m.num_kv_heads, m.head_dim), (32, 32, 128));
    m.validate_projection_shapes(&[4096, 4096], &[4096, 4096], &[4096, 4096])
        .unwrap();
    let config = m
        .standard_config(2, 11008, 32000, GatedActivation::Silu)
        .unwrap();
    assert_eq!((config.num_heads, config.head_dim), (32, 128));
    assert!(AttentionMetadata::from_hf_json(
        &json!({"hidden_size":4096,"q_proj_shape":[4096,4096]})
    )
    .is_err());
}

#[test]
fn gqa_and_mqa_counts_are_preserved_but_not_silently_converted_to_mha() {
    for kv in [1, 8] {
        let m = AttentionMetadata::from_hf_json(
            &json!({"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":kv}),
        )
        .unwrap();
        assert_eq!(m.num_kv_heads, kv);
        m.validate_projection_shapes(&[4096, 4096], &[kv * 128, 4096], &[kv * 128, 4096])
            .unwrap();
        assert!(m
            .validate_projection_shapes(&[4096, 4096], &[4096, 4096], &[4096, 4096])
            .is_err());
        assert!(matches!(
            m.standard_config(2, 8192, 32000, GatedActivation::Silu),
            Err(FerrisResError::Unsupported(_))
        ));
    }
}

#[test]
fn explicit_head_dimension_is_not_overwritten_by_hidden_width() {
    let m = AttentionMetadata::from_hf_json(
        &json!({"hidden_size":2304,"num_attention_heads":8,"num_key_value_heads":4,"head_dim":256}),
    )
    .unwrap();
    assert_eq!(m.head_dim, 256);
    m.validate_projection_shapes(&[2048, 2304], &[1024, 2304], &[1024, 2304])
        .unwrap();
    assert!(matches!(
        m.standard_config(2, 8192, 32000, GatedActivation::GeluTanh),
        Err(FerrisResError::Unsupported(_))
    ));
}

#[test]
fn invalid_or_ambiguous_hf_dimensions_are_rejected() {
    for value in [
        json!(0),
        json!(-1),
        json!(1.5),
        json!("32"),
        json!(null),
        json!(u64::MAX),
    ] {
        assert!(AttentionMetadata::from_hf_json(
            &json!({"hidden_size":4096,"num_attention_heads":value})
        )
        .is_err());
    }
    for config in [
        json!({}),
        json!({"hidden_size":0,"num_attention_heads":32}),
        json!({"hidden_size":4097,"num_attention_heads":32}),
        json!({"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":3}),
        json!({"hidden_size":4096,"num_attention_heads":32,"num_key_value_heads":64}),
        json!({"hidden_size":4096,"num_attention_heads":32,"head_dim":0}),
        json!({"hidden_size":4096,"num_attention_heads":32,"head_dim":u32::MAX}),
    ] {
        assert!(
            AttentionMetadata::from_hf_json(&config).is_err(),
            "{config}"
        );
    }
}

#[test]
fn gguf_uses_architecture_prefix_and_explicit_head_lengths() {
    let mut f = gguf();
    let m = AttentionMetadata::from_gguf(&f).unwrap();
    assert_eq!((m.num_heads, m.num_kv_heads, m.head_dim), (32, 8, 128));
    f.metadata
        .insert("qwen2.attention.key_length".into(), GgufValue::UInt32(64));
    f.metadata
        .insert("qwen2.attention.value_length".into(), GgufValue::UInt32(64));
    assert_eq!(AttentionMetadata::from_gguf(&f).unwrap().head_dim, 64);
    f.metadata.insert(
        "qwen2.attention.value_length".into(),
        GgufValue::UInt32(128),
    );
    assert!(matches!(
        AttentionMetadata::from_gguf(&f),
        Err(FerrisResError::Unsupported(_))
    ));
}

#[test]
fn malformed_gguf_metadata_is_not_defaulted_or_truncated() {
    for value in [
        GgufValue::UInt32(0),
        GgufValue::UInt64(u64::MAX),
        GgufValue::Int32(-1),
        GgufValue::Float32(32.0),
        GgufValue::String("32".into()),
    ] {
        let mut f = gguf();
        f.metadata
            .insert("qwen2.attention.head_count".into(), value);
        assert!(AttentionMetadata::from_gguf(&f).is_err());
    }
    for key in [
        "general.architecture",
        "qwen2.attention.head_count",
        "qwen2.embedding_length",
    ] {
        let mut f = gguf();
        f.metadata.remove(key);
        assert!(AttentionMetadata::from_gguf(&f).is_err());
    }
}

#[test]
fn hf_config_file_is_read_and_projection_layout_is_explicit() {
    let path =
        std::env::temp_dir().join(format!("ferrisres-head-config-{}.json", std::process::id()));
    std::fs::write(
        &path,
        br#"{"hidden_size":8,"num_attention_heads":2,"head_dim":2}"#,
    )
    .unwrap();
    let result = AttentionMetadata::from_hf_config(&path);
    std::fs::remove_file(&path).unwrap();
    let m = result.unwrap();
    m.validate_projection_shapes(&[4, 8], &[4, 8], &[4, 8])
        .unwrap();
    assert!(m
        .validate_projection_shapes(&[8, 4], &[4, 8], &[4, 8])
        .is_err());
    assert!(matches!(
        m.standard_config(1, 16, 10, GatedActivation::Silu),
        Err(FerrisResError::Unsupported(_))
    ));
}
