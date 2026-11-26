use magnetar_component_olala::mamba3_profile::{Mamba3Profile, WeightSpec, WeightTransform};

fn profile() -> Mamba3Profile {
    Mamba3Profile::from_huggingface_config(include_bytes!("fixtures/olala-mini.config.json"))
        .unwrap()
}

#[test]
fn pinned_mamba3_layers_and_weights_match_the_checkpoint_header() {
    let profile = profile();
    assert_eq!(profile.layers().len(), 29);
    let header: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olala-mamba3-header.json")).unwrap();
    assert_eq!(header.as_object().unwrap().len(), 29 * 11);
    for &layer in profile.layers() {
        let plan = profile.plan_sequence(layer, 1).unwrap();
        assert_eq!(plan.weights.len(), 11);
        for weight in plan.weights {
            let metadata = &header[&weight.checkpoint_name];
            let shape: Vec<usize> = serde_json::from_value(metadata["shape"].clone()).unwrap();
            assert_eq!(shape, weight.checkpoint_shape, "{}", weight.checkpoint_name);
            assert_eq!(
                metadata["dtype"], weight.checkpoint_dtype,
                "{}",
                weight.checkpoint_name
            );
        }
    }
}

#[test]
fn mamba3_invocation_has_distinct_state_and_mixer_shapes() {
    let profile = profile();
    let plan = profile.plan_sequence(0, 3).unwrap();
    assert_eq!(plan.operator, "mamba3-mixer");
    assert_eq!((plan.heads, plan.groups, plan.rank), (48, 1, 4));
    assert_eq!((plan.head_dim, plan.state_dim, plan.angles), (64, 128, 32));
    assert_eq!((plan.epsilon, plan.decay_floor), (1e-6, 1e-4));
    assert!(plan.zero_centered_norm);
    assert_eq!(plan.input_shapes.len(), 16);
    assert_eq!(plan.output_shapes.len(), 5);
    assert_eq!(plan.input_shapes[0], vec![3, 1536]);
    assert_eq!(
        plan.input_shapes[1..5],
        [
            vec![48, 32],
            vec![48, 64, 128],
            vec![4, 48, 128],
            vec![48, 64]
        ]
    );
    assert_eq!(plan.input_shapes[5], vec![1536, 6288]);
    assert_eq!(plan.input_shapes[6], vec![1536, 1056]);
    assert_eq!(plan.output_shapes[0], vec![3, 3072]);
    assert_eq!(plan.output_shapes[1..], plan.input_shapes[1..5]);
    let decode = profile.plan_sequence(0, 1).unwrap();
    assert_eq!(decode.input_shapes[1..5], plan.input_shapes[1..5]);
    assert_eq!(decode.output_shapes[1..], plan.output_shapes[1..]);
    assert_eq!(profile.max_tokens(), 67_108_864 / 6288);
    assert!(profile.plan_sequence(4, 1).is_err());
    assert!(profile.plan_sequence(36, 1).is_err());
    assert!(profile.plan_sequence(0, 0).is_err());
    assert!(profile.plan_sequence(0, profile.max_tokens() + 1).is_err());
}

#[test]
fn checkpoint_reordering_transposes_matrices_and_swaps_head_rank() {
    let mut spec = WeightSpec {
        checkpoint_name: "test".into(),
        checkpoint_dtype: "F32",
        checkpoint_shape: vec![2, 3],
        operator_shape: vec![3, 2],
        transform: WeightTransform::TransposeMatrix,
    };
    assert_eq!(
        spec.reorder_f32(&[0., 1., 2., 3., 4., 5.]).unwrap(),
        vec![0., 3., 1., 4., 2., 5.]
    );
    spec.checkpoint_shape = vec![2, 3, 2];
    spec.operator_shape = vec![3, 2, 2];
    spec.transform = WeightTransform::SwapHeadAndRank;
    let input: Vec<_> = (0..12).map(|i| i as f32).collect();
    assert_eq!(
        spec.reorder_f32(&input).unwrap(),
        vec![0., 1., 6., 7., 2., 3., 8., 9., 4., 5., 10., 11.]
    );
    assert!(spec.reorder_f32(&input[..11]).is_err());
    let mut invalid = input;
    invalid[0] = f32::NAN;
    assert!(spec.reorder_f32(&invalid).is_err());
    spec.operator_shape = vec![2, 3, 2];
    assert!(
        spec.reorder_f32(&(0..12).map(|i| i as f32).collect::<Vec<_>>())
            .is_err()
    );
}

#[test]
fn unsupported_mamba3_configuration_is_rejected() {
    let config: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/olala-mini.config.json")).unwrap();
    for (field, value) in [
        ("hidden_size", serde_json::json!(1024)),
        ("mamba_ngroups", serde_json::json!(2)),
        ("mamba_headdim", serde_json::json!(32)),
        ("mamba_mimo_dim", serde_json::json!(2)),
        ("mamba_d_state", serde_json::json!(64)),
        ("mamba3_postgate_norm", serde_json::json!(true)),
        ("mamba3_remove_BC_bias", serde_json::json!(true)),
        ("mamba3_remove_conv", serde_json::json!(false)),
        ("mamba3_rope", serde_json::json!(false)),
        ("complete_slw", serde_json::json!(true)),
        ("cosnet", serde_json::json!(true)),
        ("zero_centered_gamma", serde_json::json!(false)),
        ("norm_epsilon", serde_json::json!(0.0)),
        ("dtype", serde_json::json!("float32")),
    ] {
        let mut changed = config.clone();
        changed[field] = value;
        assert!(
            Mamba3Profile::from_huggingface_config(&serde_json::to_vec(&changed).unwrap()).is_err(),
            "accepted {field}"
        );
    }
    let mut missing = config;
    missing.as_object_mut().unwrap().remove("mamba3_rope");
    assert!(
        Mamba3Profile::from_huggingface_config(&serde_json::to_vec(&missing).unwrap()).is_err()
    );
}
