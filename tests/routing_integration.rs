use magnetar_component_olala::routing::{RoutingError, route};
use serde::Deserialize;

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    logits: Vec<f32>,
    selection_bias: Vec<f32>,
    top_k: usize,
    routed_scaling: f32,
    indices: Vec<usize>,
    weights: Vec<f32>,
}

#[test]
fn router_matches_pinned_upstream_pytorch_forward() {
    let cases: Vec<ReferenceCase> =
        serde_json::from_slice(include_bytes!("fixtures/olala-routing-reference.json")).unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let actual = route(
            &case.logits,
            &case.selection_bias,
            case.top_k,
            case.routed_scaling,
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(actual.len(), case.indices.len(), "{}", case.name);
        for ((expert, expected_index), expected_weight) in actual
            .iter()
            .zip(case.indices.iter())
            .zip(case.weights.iter())
        {
            assert_eq!(expert.index, *expected_index, "{}", case.name);
            assert!(
                (expert.weight - expected_weight).abs() < 1e-6,
                "{}: expert {} weight {} != {}",
                case.name,
                expert.index,
                expert.weight,
                expected_weight
            );
        }
    }
}

#[test]
fn top_one_keeps_its_sigmoid_weight_instead_of_normalizing_to_one() {
    let selected = route(&[0.0, -1.0], &[0.0, -1.0], 1, 2.0).unwrap();
    assert_eq!(selected[0].index, 0);
    assert_eq!(selected[0].weight, 1.0);
}

#[test]
fn rejects_invalid_inputs_and_an_ambiguous_selection_boundary() {
    assert_eq!(
        route(&[], &[], 1, 1.0),
        Err(RoutingError::InvalidDimensions)
    );
    assert_eq!(
        route(&[1.0], &[], 1, 1.0),
        Err(RoutingError::InvalidDimensions)
    );
    assert_eq!(
        route(&[1.0], &[0.0], 0, 1.0),
        Err(RoutingError::InvalidDimensions)
    );
    assert_eq!(
        route(&[1.0], &[0.0], 2, 1.0),
        Err(RoutingError::InvalidDimensions)
    );
    assert_eq!(
        route(&[f32::NAN], &[0.0], 1, 1.0),
        Err(RoutingError::NonFinite)
    );
    assert_eq!(
        route(&[1.0], &[f32::INFINITY], 1, 1.0),
        Err(RoutingError::NonFinite)
    );
    assert_eq!(
        route(&[1.0], &[0.0], 1, f32::NAN),
        Err(RoutingError::NonFinite)
    );
    assert_eq!(
        route(&[0.0, 0.0], &[0.0, 0.0], 1, 1.0),
        Err(RoutingError::AmbiguousBoundaryTie)
    );
}

#[test]
fn ties_within_the_selected_set_have_deterministic_order() {
    let selected = route(&[0.0, 0.0, -3.0], &[0.0; 3], 2, 1.0).unwrap();
    assert_eq!(
        selected.iter().map(|entry| entry.index).collect::<Vec<_>>(),
        [0, 1]
    );
    assert!((selected[0].weight - 0.5).abs() < 1e-6);
    assert!((selected[1].weight - 0.5).abs() < 1e-6);
}
