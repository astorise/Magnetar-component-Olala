//! Checkpoint profile for the portable projected differential attention core.

use serde::Deserialize;

use crate::{ConfigError, LayerKind, LayerLayout};

const MAX_ELEMENTS: usize = 67_108_864;
const MAX_POSITIONS: usize = 16_777_216;

/// Dimensions and numerics of the supported no-RoPE attention form.
/// This describes the isolated attention class, not an executable model.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionProfile {
    /// Zero-based model layers that use this attention profile.
    pub attention_layers: Vec<usize>,
    pub hidden_size: usize,
    pub query_heads: usize,
    pub signal_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rank: usize,
    pub epsilon: f64,
    pub zero_centered_norm: bool,
    pub score_scale: f64,
    pub softcap: f64,
    /// Maximum key distance to the left, or None for an unlimited window.
    pub window_left: Option<usize>,
    /// The portable reference's largest unbatched query and history shapes.
    pub max_query_tokens: usize,
    pub max_history_tokens: usize,
}

/// One checkpoint tensor and the shape expected by the portable operator.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttentionWeightSpec {
    pub checkpoint_name: String,
    pub checkpoint_shape: Vec<usize>,
    pub operator_shape: Vec<usize>,
    /// Matrices are stored output-major and must be transposed on load.
    pub transpose: bool,
}

#[derive(Deserialize)]
struct RawAttention {
    hidden_size: usize,
    num_attention_heads: usize,
    num_signal_heads_diff: usize,
    head_dim: usize,
    tpa_rank: usize,
    max_position_embeddings: usize,
    qk_norm: bool,
    token_shift_attn: bool,
    scalable_softmax: bool,
    softcap_attn: f64,
    rope_theta: f64,
    rope_type: String,
    slw_wsize: i64,
    use_completed_p: bool,
    xsa: bool,
    cosnet: bool,
    normalization_type: String,
    norm_epsilon: f64,
    zero_centered_gamma: bool,
    use_value_embedding: bool,
    token_conv1d_attn: bool,
    intra_doc_masking: bool,
}

impl AttentionProfile {
    /// Parse the explicit checkpoint settings required by the portable core.
    /// Missing settings and unsupported variants are rejected; passing this
    /// check does not implement Component graph production or checkpoint loading.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, ConfigError> {
        let layout = LayerLayout::from_huggingface_config(bytes)?;
        if !layout
            .layers()
            .contains(&LayerKind::DifferentialTpaAttention)
        {
            return Err(ConfigError(
                "configuration has no differential attention layer".into(),
            ));
        }
        let raw: RawAttention = serde_json::from_slice(bytes)
            .map_err(|error| ConfigError(format!("invalid attention settings: {error}")))?;
        if !raw.qk_norm
            || !raw.token_shift_attn
            || !raw.scalable_softmax
            || raw.xsa
            || raw.cosnet
            || raw.use_value_embedding
            || raw.token_conv1d_attn
            || raw.intra_doc_masking
            || raw.normalization_type != "rmsnorm"
            || raw.rope_theta != 0.0
            || !raw.rope_type.is_empty()
        {
            return Err(ConfigError(
                "unsupported differential attention profile".into(),
            ));
        }
        let positive_f32 =
            |value: f64| value.is_finite() && (value as f32).is_finite() && value as f32 > 0.0;
        if !positive_f32(raw.norm_epsilon)
            || !raw.softcap_attn.is_finite()
            || raw.softcap_attn < 0.0
            || (raw.softcap_attn > 0.0 && !positive_f32(raw.softcap_attn))
        {
            return Err(ConfigError("invalid attention epsilon or softcap".into()));
        }
        let window_left = match raw.slw_wsize {
            -1 => None,
            value if value >= 0 => {
                Some(usize::try_from(value).map_err(|_| ConfigError("window too large".into()))?)
            }
            _ => return Err(ConfigError("invalid attention window".into())),
        };
        let signal = if raw.num_signal_heads_diff == 0 {
            raw.num_attention_heads / 2
        } else {
            raw.num_signal_heads_diff
        };
        let Some(kv_heads) = raw
            .num_attention_heads
            .checked_sub(signal)
            .filter(|&n| n > 0)
        else {
            return Err(ConfigError("invalid signal/noise head count".into()));
        };
        if signal == 0
            || !signal.is_multiple_of(kv_heads)
            || raw.hidden_size == 0
            || raw.head_dim == 0
            || raw.tpa_rank == 0
            || raw.max_position_embeddings == 0
        {
            return Err(ConfigError("invalid attention geometry".into()));
        }
        let product = |dims: &[usize]| {
            dims.iter()
                .try_fold(1usize, |n, &dim| {
                    n.checked_mul(dim).filter(|&value| value <= MAX_ELEMENTS)
                })
                .ok_or_else(|| {
                    ConfigError("attention dimensions exceed the portable reference limit".into())
                })
        };
        let query_width = product(&[raw.num_attention_heads, raw.head_dim])?;
        let kv_width = product(&[kv_heads, raw.head_dim])?;
        let a_width = product(&[kv_heads, raw.tpa_rank])?;
        let b_width = product(&[raw.tpa_rank, raw.head_dim])?;
        for width in [query_width, a_width, b_width, kv_heads] {
            product(&[raw.hidden_size, width])?;
        }
        let score_scale = if raw.use_completed_p {
            1.0 / raw.head_dim as f64
        } else {
            1.0 / (raw.head_dim as f64).sqrt()
        };
        let max_history_tokens = raw
            .max_position_embeddings
            .min(MAX_POSITIONS)
            .min(MAX_ELEMENTS / kv_width);
        let widest_query_row = [raw.hidden_size, query_width, a_width, b_width, kv_heads]
            .into_iter()
            .max()
            .expect("nonempty width list");
        Ok(Self {
            attention_layers: layout
                .layers()
                .iter()
                .enumerate()
                .filter_map(|(index, kind)| {
                    (*kind == LayerKind::DifferentialTpaAttention).then_some(index)
                })
                .collect(),
            hidden_size: raw.hidden_size,
            query_heads: raw.num_attention_heads,
            signal_heads: signal,
            kv_heads,
            head_dim: raw.head_dim,
            rank: raw.tpa_rank,
            epsilon: raw.norm_epsilon,
            zero_centered_norm: raw.zero_centered_gamma,
            score_scale,
            softcap: raw.softcap_attn,
            window_left,
            max_query_tokens: (MAX_ELEMENTS / widest_query_row).min(max_history_tokens),
            max_history_tokens,
        })
    }

    /// Required learned inputs, in projected-attention operator order.
    /// Checkpoint names and shapes alone do not load or transpose any weights.
    pub fn weight_specs(&self, layer: usize) -> Result<Vec<AttentionWeightSpec>, ConfigError> {
        if !self.attention_layers.contains(&layer) {
            return Err(ConfigError("layer is not differential attention".into()));
        }
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .filter(|&value| value > 0 && value <= MAX_ELEMENTS)
                .ok_or_else(|| ConfigError("attention weight dimensions overflow".into()))
        };
        let q = product(self.query_heads, self.head_dim)?;
        let a = product(self.kv_heads, self.rank)?;
        let b = product(self.rank, self.head_dim)?;
        for width in [q, a, b, self.kv_heads] {
            product(self.hidden_size, width)?;
        }
        let prefix = format!("model.layers.{layer}.mixer");
        let matrix = |name: &str, rows: usize| AttentionWeightSpec {
            checkpoint_name: format!("{prefix}.{name}.weight"),
            checkpoint_shape: vec![rows, self.hidden_size],
            operator_shape: vec![self.hidden_size, rows],
            transpose: true,
        };
        let vector = |name: &str, size: usize| AttentionWeightSpec {
            checkpoint_name: format!("{prefix}.{name}"),
            checkpoint_shape: vec![size],
            operator_shape: vec![size],
            transpose: false,
        };
        Ok(vec![
            matrix("c_q", q),
            matrix("W_A_k", a),
            matrix("W_B_k", b),
            matrix("W_A_v", a),
            matrix("W_B_v", b),
            matrix("shift_proj_k", self.kv_heads),
            matrix("shift_proj_v", self.kv_heads),
            matrix("lambda_proj", self.kv_heads),
            vector("q_norm.norm.weight", self.head_dim),
            vector("k_norm.norm.weight", self.head_dim),
            vector("softmax_scaler", self.query_heads),
        ])
    }
}
