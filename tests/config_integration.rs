use magnetar_component_olala::{LayerKind, LayerLayout};

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
