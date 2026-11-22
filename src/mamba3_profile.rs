//! Checkpoint mapping for the pinned Olala mini Mamba-3 mixer.
//!
//! This describes f32 CPU operator inputs. It does not implement BF16 model
//! execution, graph production, checkpoint loading or recurrent sessions.

use serde::Deserialize;

use crate::{ConfigError, LayerKind, LayerLayout};

const LAYOUT: &str = "MMMMVMMMMVMMMMVMMMMVMMMMVMMMMVMMMMVM";
const MAX_ELEMENTS: usize = 67_108_864;
const HIDDEN: usize = 1536;
const HEADS: usize = 48;
const HEAD_DIM: usize = 64;
const GROUPS: usize = 1;
const RANK: usize = 4;
const STATE_DIM: usize = 128;
const ANGLES: usize = 32;
const PROJECTION: usize = HEADS * (2 * HEAD_DIM + 3);
const DYNAMIC: usize = 2 * GROUPS * RANK * STATE_DIM + ANGLES;

/// A checkpoint tensor's conversion after its storage bytes are decoded to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightTransform {
    Identity,
    TransposeMatrix,
    SwapHeadAndRank,
}

/// Checkpoint and portable-operator shapes for one learned input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSpec {
    pub checkpoint_name: String,
    pub checkpoint_dtype: &'static str,
    pub checkpoint_shape: Vec<usize>,
    pub operator_shape: Vec<usize>,
    pub transform: WeightTransform,
}

impl WeightSpec {
    /// Reorder an already-decoded, contiguous checkpoint tensor. The caller
    /// must first verify the checkpoint name, dtype, shape and payload bytes.
    pub fn reorder_f32(&self, values: &[f32]) -> Result<Vec<f32>, ConfigError> {
        if self.checkpoint_shape.is_empty() {
            return Err(ConfigError("invalid Mamba-3 weight shape".into()));
        }
        let count = self
            .checkpoint_shape
            .iter()
            .try_fold(1usize, |count, &dimension| {
                count
                    .checked_mul(dimension)
                    .filter(|&value| dimension > 0 && value <= MAX_ELEMENTS)
            })
            .ok_or_else(|| ConfigError("invalid Mamba-3 weight shape".into()))?;
        if values.len() != count || values.iter().any(|value| !value.is_finite()) {
            return Err(ConfigError("invalid Mamba-3 weight values".into()));
        }
        match (self.transform, self.checkpoint_shape.as_slice()) {
            (WeightTransform::Identity, shape) if self.operator_shape == shape => {
                Ok(values.to_vec())
            }
            (WeightTransform::TransposeMatrix, [rows, columns])
                if self.operator_shape == [*columns, *rows] =>
            {
                let mut output = vec![0.0; count];
                for row in 0..*rows {
                    for column in 0..*columns {
                        output[column * rows + row] = values[row * columns + column];
                    }
                }
                Ok(output)
            }
            (WeightTransform::SwapHeadAndRank, [heads, rank, width])
                if self.operator_shape == [*rank, *heads, *width] =>
            {
                let mut output = vec![0.0; count];
                for head in 0..*heads {
                    for index in 0..*rank {
                        let source = (head * rank + index) * width;
                        let target = (index * heads + head) * width;
                        output[target..target + width]
                            .copy_from_slice(&values[source..source + width]);
                    }
                }
                Ok(output)
            }
            _ => Err(ConfigError("invalid Mamba-3 weight transform".into())),
        }
    }
}

/// An explicit single-sequence invocation of the f32 reference operator.
#[derive(Clone, Debug, PartialEq)]
pub struct InvocationPlan {
    pub operator: &'static str,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub weights: Vec<WeightSpec>,
    pub heads: usize,
    pub groups: usize,
    pub rank: usize,
    pub head_dim: usize,
    pub state_dim: usize,
    pub angles: usize,
    pub epsilon: f64,
    pub decay_floor: f64,
    pub zero_centered_norm: bool,
}

/// Validated layer locations for the exact pinned mini checkpoint profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mamba3Profile {
    layers: Vec<usize>,
}

#[derive(Deserialize)]
struct RawConfig {
    layers_config: String,
    hidden_size: usize,
    expand_factor: usize,
    mamba_headdim: usize,
    mamba_ngroups: usize,
    mamba_mimo_dim: usize,
    mamba_d_state: usize,
    max_position_embeddings: usize,
    normalization_type: String,
    norm_epsilon: f64,
    zero_centered_gamma: bool,
    dtype: String,
    cosnet: bool,
    complete_slw: bool,
    use_value_embedding: bool,
    mamba3_add_trapezoid: bool,
    #[serde(rename = "mamba3_is_A_dd")]
    mamba3_is_a_dd: bool,
    mamba3_is_id_rms: bool,
    mamba3_postgate_norm: bool,
    #[serde(rename = "mamba3_remove_BC_bias")]
    mamba3_remove_bc_bias: bool,
    mamba3_remove_conv: bool,
    mamba3_rope: bool,
}

impl Mamba3Profile {
    /// Accept the pinned checkpoint's configuration rather than guessing
    /// support for other Olala mixer variants.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, ConfigError> {
        let layout = LayerLayout::from_huggingface_config(bytes)?;
        let raw: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| ConfigError(format!("invalid Mamba-3 configuration: {error}")))?;
        if raw.layers_config != LAYOUT
            || raw.hidden_size != HIDDEN
            || raw.expand_factor != 2
            || raw.mamba_headdim != HEAD_DIM
            || raw.mamba_ngroups != GROUPS
            || raw.mamba_mimo_dim != RANK
            || raw.mamba_d_state != STATE_DIM
            || raw.max_position_embeddings != 100_000
            || raw.normalization_type != "rmsnorm"
            || raw.norm_epsilon != 1e-6
            || !raw.zero_centered_gamma
            || raw.dtype != "bfloat16"
            || raw.cosnet
            || raw.complete_slw
            || raw.use_value_embedding
            || !raw.mamba3_add_trapezoid
            || !raw.mamba3_is_a_dd
            || !raw.mamba3_is_id_rms
            || raw.mamba3_postgate_norm
            || raw.mamba3_remove_bc_bias
            || !raw.mamba3_remove_conv
            || !raw.mamba3_rope
        {
            return Err(ConfigError("unsupported Olala mini Mamba-3 profile".into()));
        }
        Ok(Self {
            layers: layout
                .layers()
                .iter()
                .enumerate()
                .filter_map(|(index, kind)| (*kind == LayerKind::Mamba3).then_some(index))
                .collect(),
        })
    }

    pub fn layers(&self) -> &[usize] {
        &self.layers
    }

    /// Largest unbatched token count admitted by the portable CPU reference.
    pub fn max_tokens(&self) -> usize {
        (MAX_ELEMENTS / PROJECTION).min(100_000)
    }

    /// Plan one contiguous sequence. The four input states must be zeros for
    /// a new document, or the prior committed states for its continuation.
    pub fn plan_sequence(
        &self,
        layer: usize,
        tokens: usize,
    ) -> Result<InvocationPlan, ConfigError> {
        if !self.layers.contains(&layer) || tokens == 0 || tokens > self.max_tokens() {
            return Err(ConfigError(
                "invalid Mamba-3 layer or sequence length".into(),
            ));
        }
        let prefix = format!("model.layers.{layer}.mixer");
        let weight = |name: &str, dtype, checkpoint_shape, operator_shape, transform| WeightSpec {
            checkpoint_name: format!("{prefix}.{name}"),
            checkpoint_dtype: dtype,
            checkpoint_shape,
            operator_shape,
            transform,
        };
        let matrix = |name, width| {
            weight(
                name,
                "BF16",
                vec![width, HIDDEN],
                vec![HIDDEN, width],
                WeightTransform::TransposeMatrix,
            )
        };
        let vector = |name, dtype, width| {
            weight(
                name,
                dtype,
                vec![width],
                vec![width],
                WeightTransform::Identity,
            )
        };
        let ranked = |name, width| {
            weight(
                name,
                "F32",
                vec![HEADS, RANK, width],
                vec![RANK, HEADS, width],
                WeightTransform::SwapHeadAndRank,
            )
        };
        let weights = vec![
            matrix("in_proj.weight", PROJECTION),
            matrix("in_proj_dyn.weight", DYNAMIC),
            vector("dt_bias", "F32", HEADS),
            vector("B_norm.norm.weight", "BF16", STATE_DIM),
            vector("C_norm.norm.weight", "BF16", STATE_DIM),
            ranked("C_bias", STATE_DIM),
            ranked("B_bias", STATE_DIM),
            ranked("in_proj_mimo_x", HEAD_DIM),
            ranked("in_proj_mimo_z", HEAD_DIM),
            ranked("out_proj_mimo", HEAD_DIM),
            vector("D", "F32", HEADS),
        ];
        let states = [
            vec![HEADS, ANGLES],
            vec![HEADS, HEAD_DIM, STATE_DIM],
            vec![RANK, HEADS, STATE_DIM],
            vec![HEADS, HEAD_DIM],
        ];
        let mut input_shapes = vec![vec![tokens, HIDDEN]];
        input_shapes.extend(states.iter().cloned());
        input_shapes.extend(weights.iter().map(|weight| weight.operator_shape.clone()));
        let mut output_shapes = vec![vec![tokens, HEADS * HEAD_DIM]];
        output_shapes.extend(states);
        Ok(InvocationPlan {
            operator: "mamba3-mixer",
            input_shapes,
            output_shapes,
            weights,
            heads: HEADS,
            groups: GROUPS,
            rank: RANK,
            head_dim: HEAD_DIM,
            state_dim: STATE_DIM,
            angles: ANGLES,
            epsilon: 1e-6,
            decay_floor: 1e-4,
            zero_centered_norm: true,
        })
    }
}
