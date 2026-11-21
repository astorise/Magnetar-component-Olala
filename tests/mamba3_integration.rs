use magnetar_component_olala::mamba3::{Geometry, Parameters, State, Token, step};
use serde::Deserialize;

#[derive(Deserialize)]
struct GeometryFixture {
    heads: usize,
    groups: usize,
    rank: usize,
    head_dim: usize,
    state_dim: usize,
    angles: usize,
}

impl From<&GeometryFixture> for Geometry {
    fn from(value: &GeometryFixture) -> Self {
        Self {
            heads: value.heads,
            groups: value.groups,
            rank: value.rank,
            head_dim: value.head_dim,
            state_dim: value.state_dim,
            angles: value.angles,
        }
    }
}

#[derive(Deserialize)]
struct ParameterFixture {
    query_bias: Vec<f32>,
    key_bias: Vec<f32>,
    input_projection: Vec<f32>,
    gate_projection: Vec<f32>,
    output_projection: Vec<f32>,
    skip: Vec<f32>,
}

impl From<&ParameterFixture> for Parameters {
    fn from(value: &ParameterFixture) -> Self {
        Self {
            query_bias: value.query_bias.clone(),
            key_bias: value.key_bias.clone(),
            input_projection: value.input_projection.clone(),
            gate_projection: value.gate_projection.clone(),
            output_projection: value.output_projection.clone(),
            skip: value.skip.clone(),
        }
    }
}

#[derive(Deserialize)]
struct TokenFixture {
    query: Vec<f32>,
    key: Vec<f32>,
    input: Vec<f32>,
    gate: Vec<f32>,
    decay: Vec<f32>,
    dt: Vec<f32>,
    trap: Vec<f32>,
    angle_projection: Vec<f32>,
}

impl TokenFixture {
    fn token(&self) -> Token<'_> {
        Token {
            query: &self.query,
            key: &self.key,
            input: &self.input,
            gate: &self.gate,
            decay: &self.decay,
            dt: &self.dt,
            trap: &self.trap,
            angle_projection: &self.angle_projection,
        }
    }
}

#[derive(Deserialize)]
struct StepFixture {
    output: Vec<f32>,
    angle: Vec<f32>,
    ssm: Vec<f32>,
    key: Vec<f32>,
    input: Vec<f32>,
}

#[derive(Deserialize)]
struct Reference {
    geometry: GeometryFixture,
    parameters: ParameterFixture,
    tokens: Vec<TokenFixture>,
    steps: Vec<StepFixture>,
    prefill_output: Vec<f32>,
    prefill_angle: Vec<f32>,
    prefill_ssm: Vec<f32>,
    prefill_key: Vec<f32>,
    prefill_input: Vec<f32>,
}

fn cases() -> Vec<Reference> {
    serde_json::from_slice(include_bytes!("fixtures/olala-mamba3-reference.json")).unwrap()
}

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name}");
    for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - b).abs() <= 3e-6 + 1e-6 * b.abs(),
            "{name}[{index}]: {a} != {b}"
        );
    }
}

#[test]
fn recurrence_matches_pinned_pytorch_prefill_and_cached_steps() {
    for case in cases() {
        let geometry = Geometry::from(&case.geometry);
        let parameters = Parameters::from(&case.parameters);
        let zero = State::zero(geometry).unwrap();
        let mut state = zero.clone();
        let mut full = Vec::new();
        let mut prefix = None;
        for (index, (token, expected)) in case.tokens.iter().zip(&case.steps).enumerate() {
            let result = step(geometry, &parameters, &state, &token.token()).unwrap();
            close(&result.output, &expected.output, "step output");
            close(&result.state.angle, &expected.angle, "angle state");
            close(&result.state.ssm, &expected.ssm, "SSM state");
            close(&result.state.key, &expected.key, "key state");
            close(&result.state.input, &expected.input, "input state");
            full.extend_from_slice(&result.output);
            state = result.state;
            if index == 1 {
                prefix = Some(state.clone());
            }
        }
        close(&full, &case.prefill_output, "full prefill");
        close(&state.ssm, &case.prefill_ssm, "prefill SSM");
        close(&state.key, &case.prefill_key, "prefill key");
        close(&state.input, &case.prefill_input, "prefill input");
        let wrapped: Vec<_> = state
            .angle
            .iter()
            .map(|value| value.rem_euclid(std::f32::consts::TAU))
            .collect();
        close(&wrapped, &case.prefill_angle, "prefill angle modulo 2pi");

        // Chunk continuation and a new document reset do not share state.
        let mut continued = prefix.unwrap();
        // Upstream returns angles modulo 2*pi from prefill but leaves them
        // unwrapped during cached steps. Starting decode from the prefill
        // representation must preserve the same recurrence.
        for angle in &mut continued.angle {
            *angle = angle.rem_euclid(std::f32::consts::TAU);
        }
        for (token, expected) in case.tokens[2..].iter().zip(&case.steps[2..]) {
            let result = step(geometry, &parameters, &continued, &token.token()).unwrap();
            close(&result.output, &expected.output, "prefill-to-decode output");
            continued = result.state;
        }
        close(&continued.ssm, &state.ssm, "chunked continuation");
        let reset = step(geometry, &parameters, &zero, &case.tokens[0].token()).unwrap();
        close(&reset.output, &case.steps[0].output, "reset output");
        close(&reset.state.ssm, &case.steps[0].ssm, "reset state");
        assert_eq!(zero, State::zero(geometry).unwrap());
    }
}

#[test]
fn invalid_geometry_and_tensors_fail_without_modifying_the_prior_state() {
    let case = cases().remove(0);
    let geometry = Geometry::from(&case.geometry);
    let parameters = Parameters::from(&case.parameters);
    let state = State::zero(geometry).unwrap();
    assert!(
        State::zero(Geometry {
            groups: 3,
            ..geometry
        })
        .is_err()
    );
    assert!(
        State::zero(Geometry {
            state_dim: 3,
            ..geometry
        })
        .is_err()
    );
    assert!(
        State::zero(Geometry {
            angles: geometry.state_dim,
            ..geometry
        })
        .is_err()
    );
    let mut bad = case.tokens[0].token();
    bad.query = &[];
    assert!(step(geometry, &parameters, &state, &bad).is_err());
    bad = case.tokens[0].token();
    bad.dt = &[-1.0, 0.5];
    assert!(step(geometry, &parameters, &state, &bad).is_err());
    bad = case.tokens[0].token();
    bad.trap = &[f32::NAN, 0.5];
    assert!(step(geometry, &parameters, &state, &bad).is_err());
    assert_eq!(state, State::zero(geometry).unwrap());
}
