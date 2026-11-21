use magnetar_component_olala::geodesic::{GeodesicError, update};
use serde::Deserialize;

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    residual: Vec<f32>,
    branch: Vec<f32>,
    scale: f32,
    bias: f32,
    layer_index: usize,
    output: Vec<f32>,
}

#[test]
fn geodesic_update_matches_pinned_upstream_pytorch() {
    let cases: Vec<ReferenceCase> =
        serde_json::from_slice(include_bytes!("fixtures/olala-geodesic-reference.json")).unwrap();
    assert_eq!(cases.len(), 8);
    for case in cases {
        let actual = update(
            &case.residual,
            &case.branch,
            case.scale,
            case.bias,
            case.layer_index,
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(actual.len(), case.output.len());
        for (index, (value, expected)) in actual.iter().zip(&case.output).enumerate() {
            assert!(
                (value - expected).abs() <= 1e-6,
                "{}[{index}]: {value} != {expected}",
                case.name
            );
        }
    }
}

#[test]
fn invalid_dimensions_layers_and_nonfinite_values_are_rejected() {
    assert_eq!(
        update(&[], &[], 1.0, 0.0, 0),
        Err(GeodesicError::InvalidDimensions)
    );
    assert_eq!(
        update(&[1.0], &[], 1.0, 0.0, 0),
        Err(GeodesicError::InvalidDimensions)
    );
    assert_eq!(
        update(&[1.0], &[1.0], 1.0, 0.0, usize::MAX),
        Err(GeodesicError::InvalidLayer)
    );
    assert_eq!(
        update(&[f32::NAN], &[0.0], 1.0, 0.0, 0),
        Err(GeodesicError::NonFinite)
    );
    assert_eq!(
        update(&[1.0], &[0.0], f32::INFINITY, 0.0, 0),
        Err(GeodesicError::NonFinite)
    );
    assert_eq!(
        update(&[f32::MAX], &[f32::MAX], 1.0, 0.0, 0),
        Err(GeodesicError::NonFinite)
    );
}
