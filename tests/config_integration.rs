use magnetar_component_olala::{LayerKind, LayerLayout, attention_profile::AttentionProfile};

#[test]
fn published_olala_mini_layout_preserves_every_mixer_in_order() {
    let config = include_bytes!("fixtures/olala-mini.config.json");
    let layout = LayerLayout::from_huggingface_config(config).unwrap();
    assert_eq!(layout.layers().len(), 36);
    assert_eq!(
        layout
            .layers()
            .iter()
            .filter(|layer| **layer == LayerKind::Mamba3)
            .count(),
        29
    );
    for (index, &layer) in layout.layers().iter().enumerate() {
        let expected = if (index + 1) % 5 == 0 {
            LayerKind::DifferentialTpaAttention
        } else {
            LayerKind::Mamba3
        };
        assert_eq!(layer, expected, "layer {index}");
    }
}

#[test]
fn a_custom_layout_is_not_replaced_by_the_published_default() {
    let config = br#"{"model_type":"olala","num_hidden_layers":3,"layers_config":"VMV"}"#;
    assert_eq!(
        LayerLayout::from_huggingface_config(config)
            .unwrap()
            .layers(),
        &[
            LayerKind::DifferentialTpaAttention,
            LayerKind::Mamba3,
            LayerKind::DifferentialTpaAttention,
        ]
    );
}

#[test]
fn rejects_unknown_or_inconsistent_layouts() {
    let cases: &[&[u8]] = &[
        br#"{"model_type":"llama","num_hidden_layers":1,"layers_config":"M"}"#,
        br#"{"model_type":"olala","num_hidden_layers":0,"layers_config":""}"#,
        br#"{"model_type":"olala","num_hidden_layers":4097,"layers_config":"M"}"#,
        br#"{"model_type":"olala","num_hidden_layers":2,"layers_config":"M"}"#,
        br#"{"model_type":"olala","num_hidden_layers":2,"layers_config":"MX"}"#,
        br#"{"model_type":"olala","num_hidden_layers":2}"#,
        b"not-json",
    ];
    for config in cases {
        assert!(
            LayerLayout::from_huggingface_config(config).is_err(),
            "invalid config accepted: {}",
            String::from_utf8_lossy(config)
        );
    }
}

#[test]
fn published_attention_profile_preserves_geometry_and_numeric_limits() {
    let config = include_bytes!("fixtures/olala-mini.config.json");
    let profile = AttentionProfile::from_huggingface_config(config).unwrap();
    assert_eq!(profile.attention_layers, vec![4, 9, 14, 19, 24, 29, 34]);
    assert_eq!(
        (
            profile.hidden_size,
            profile.query_heads,
            profile.signal_heads,
            profile.kv_heads
        ),
        (1536, 48, 36, 12)
    );
    assert_eq!((profile.head_dim, profile.rank), (128, 4));
    assert_eq!(profile.epsilon, 1e-6);
    assert!(profile.zero_centered_norm);
    assert_eq!(profile.score_scale, 1.0 / 128.0);
    assert_eq!(profile.softcap, 150.0);
    assert_eq!(profile.window_left, Some(32768));
    assert_eq!(profile.max_query_tokens, 67_108_864 / (48 * 128));
    assert_eq!(profile.max_history_tokens, 67_108_864 / (12 * 128));
}

#[test]
fn attention_weights_match_the_pinned_safetensors_header() {
    let profile = AttentionProfile::from_huggingface_config(include_bytes!(
        "fixtures/olala-mini.config.json"
    ))
    .unwrap();
    let header: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olala-attention-header.json")).unwrap();
    let weights = profile.weight_specs(4).unwrap();
    assert_eq!(weights.len(), 11);
    assert_eq!(header.as_object().unwrap().len(), weights.len());
    for (index, weight) in weights.iter().enumerate() {
        let actual = &header[&weight.checkpoint_name];
        let shape: Vec<usize> = serde_json::from_value(actual["shape"].clone()).unwrap();
        assert_eq!(weight.checkpoint_shape, shape, "{}", weight.checkpoint_name);
        assert_eq!(actual["dtype"], if index == 10 { "F32" } else { "BF16" });
        if weight.transpose {
            assert_eq!(weight.operator_shape, vec![shape[1], shape[0]]);
        } else {
            assert_eq!(weight.operator_shape, shape);
        }
    }
    assert!(profile.weight_specs(0).is_err());
    assert!(profile.weight_specs(36).is_err());
    let mut invalid = profile;
    invalid.hidden_size = usize::MAX;
    assert!(invalid.weight_specs(4).is_err());
}

#[test]
fn attention_profile_rejects_missing_and_unsupported_settings() {
    let original: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olala-mini.config.json")).unwrap();
    let cases = [
        ("rope_theta", serde_json::json!(10000.0)),
        ("rope_type", serde_json::json!("rope")),
        ("xsa", serde_json::json!(true)),
        ("cosnet", serde_json::json!(true)),
        ("use_value_embedding", serde_json::json!(true)),
        ("token_conv1d_attn", serde_json::json!(true)),
        ("intra_doc_masking", serde_json::json!(true)),
        ("token_shift_attn", serde_json::json!(false)),
        ("qk_norm", serde_json::json!(false)),
        ("scalable_softmax", serde_json::json!(false)),
        ("normalization_type", serde_json::json!("layernorm")),
        ("norm_epsilon", serde_json::json!(0.0)),
        ("softcap_attn", serde_json::json!(-1.0)),
        ("slw_wsize", serde_json::json!(-2)),
        ("num_signal_heads_diff", serde_json::json!(35)),
        ("head_dim", serde_json::json!(0)),
        ("tpa_rank", serde_json::json!(0)),
        ("max_position_embeddings", serde_json::json!(0)),
    ];
    for (name, value) in cases {
        let mut config = original.clone();
        config[name] = value;
        assert!(
            AttentionProfile::from_huggingface_config(&serde_json::to_vec(&config).unwrap())
                .is_err(),
            "accepted {name}"
        );
    }
    let mut missing = original.clone();
    missing.as_object_mut().unwrap().remove("token_shift_attn");
    assert!(
        AttentionProfile::from_huggingface_config(&serde_json::to_vec(&missing).unwrap()).is_err()
    );
}

#[test]
fn attention_profile_preserves_other_supported_scaling_and_window() {
    let mut config: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olala-mini.config.json")).unwrap();
    config["num_signal_heads_diff"] = serde_json::json!(0);
    config["use_completed_p"] = serde_json::json!(false);
    config["slw_wsize"] = serde_json::json!(-1);
    config["max_position_embeddings"] = serde_json::json!(100);
    let profile =
        AttentionProfile::from_huggingface_config(&serde_json::to_vec(&config).unwrap()).unwrap();
    assert_eq!((profile.signal_heads, profile.kv_heads), (24, 24));
    assert_eq!(profile.score_scale, 1.0 / 128.0_f64.sqrt());
    assert_eq!(profile.window_left, None);
    assert_eq!(profile.max_query_tokens, 100);
    assert_eq!(profile.max_history_tokens, 100);
}
