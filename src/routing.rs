//! CPU reference for Olala's sigmoid router.
//!
//! This independently testable implementation describes the numerical routing
//! semantics needed by the future graph and Provider operators. It does not
//! load weights, execute experts, or connect a router to the Magnetar runtime.

use std::{error::Error, fmt};

/// One selected expert and its unbiased, scaled combination weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutedExpert {
    pub index: usize,
    pub weight: f32,
}

/// Route one token's precomputed router logits.
///
/// Selection uses `sigmoid(logit) + selection_bias`; weights use the original,
/// unbiased sigmoid values. With `top_k > 1` they are normalized by the sum
/// plus `1e-20`; top-1 is not normalized. Finally `routed_scaling` is applied.
///
/// An exact tie across the selection boundary is rejected: upstream PyTorch
/// does not specify which of the tied experts `topk` will select. Ties among
/// already selected experts use ascending index as a deterministic ordering,
/// and do not change the selected set or mathematical mixture.
pub fn route(
    logits: &[f32],
    selection_bias: &[f32],
    top_k: usize,
    routed_scaling: f32,
) -> Result<Vec<RoutedExpert>, RoutingError> {
    if logits.is_empty()
        || logits.len() != selection_bias.len()
        || !(1..=logits.len()).contains(&top_k)
    {
        return Err(RoutingError::InvalidDimensions);
    }
    if !routed_scaling.is_finite() || logits.iter().chain(selection_bias).any(|x| !x.is_finite()) {
        return Err(RoutingError::NonFinite);
    }

    let mut scores = Vec::with_capacity(logits.len());
    for (index, (&logit, &bias)) in logits.iter().zip(selection_bias).enumerate() {
        // Matches the upstream float32 sigmoid, including saturation to zero.
        let score = 1.0 / (1.0 + (-logit).exp());
        let selection_score = score + bias;
        if !selection_score.is_finite() {
            return Err(RoutingError::NonFinite);
        }
        scores.push((index, score, selection_score));
    }
    scores.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
    });
    if top_k < scores.len() && scores[top_k - 1].2 == scores[top_k].2 {
        return Err(RoutingError::AmbiguousBoundaryTie);
    }
    scores.truncate(top_k);
    let sum: f32 = scores.iter().map(|(_, score, _)| score).sum();
    scores
        .into_iter()
        .map(|(index, score, _)| {
            let weight = if top_k > 1 {
                score / (sum + 1e-20)
            } else {
                score
            } * routed_scaling;
            if weight.is_finite() {
                Ok(RoutedExpert { index, weight })
            } else {
                Err(RoutingError::NonFinite)
            }
        })
        .collect()
}

/// An input cannot be routed safely or unambiguously.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutingError {
    InvalidDimensions,
    NonFinite,
    AmbiguousBoundaryTie,
}

impl fmt::Display for RoutingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDimensions => "invalid expert count, bias length, or top-k",
            Self::NonFinite => "non-finite input or routing result",
            Self::AmbiguousBoundaryTie => "equal selection scores straddle the top-k boundary",
        })
    }
}

impl Error for RoutingError {}
