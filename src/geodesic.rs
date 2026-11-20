//! CPU reference for Olala's learned geodesic residual update.
//!
//! This is a standalone reference, not a Magnetar execution operator. The
//! runtime needs a portable representation and Provider implementations before
//! this update can be used in a production graph.

use std::{error::Error, fmt};

/// Compute one float32 Olala geodesic residual update.
///
/// `layer_index` is the original zero-based layer index, including when the
/// model is executed in segments. The caller supplies the learned scalar
/// parameters for the appropriate mixer or MLP update.
pub fn update(
    residual: &[f32],
    branch: &[f32],
    scale: f32,
    bias: f32,
    layer_index: usize,
) -> Result<Vec<f32>, GeodesicError> {
    if residual.is_empty() || residual.len() != branch.len() {
        return Err(GeodesicError::InvalidDimensions);
    }
    if !scale.is_finite()
        || !bias.is_finite()
        || residual
            .iter()
            .chain(branch)
            .any(|value| !value.is_finite())
    {
        return Err(GeodesicError::NonFinite);
    }
    let layer_number = layer_index
        .checked_add(1)
        .ok_or(GeodesicError::InvalidLayer)? as f32;

    let norm_squared: f32 = residual.iter().map(|x| x * x).sum();
    let dot: f32 = residual.iter().zip(branch).map(|(x, g)| x * g).sum();
    let coefficient = dot / norm_squared.max(1e-12);
    let tangent: Vec<f32> = residual
        .iter()
        .zip(branch)
        .map(|(x, g)| g - coefficient * x)
        .collect();
    let tangent_norm = tangent.iter().map(|g| g * g).sum::<f32>().sqrt();
    let safe_tangent_norm = tangent_norm.max(1e-8);
    let safe_radius = norm_squared.sqrt().max(1e-6);
    let raw_angle = (safe_tangent_norm / safe_radius).min(std::f32::consts::FRAC_PI_4);
    // The upstream clamp is an upper bound only: negative angles are valid.
    let angle = ((raw_angle * scale + bias) / layer_number).min(std::f32::consts::FRAC_PI_4);
    if !norm_squared.is_finite()
        || !dot.is_finite()
        || !coefficient.is_finite()
        || !tangent_norm.is_finite()
        || !angle.is_finite()
    {
        return Err(GeodesicError::NonFinite);
    }
    let (sin, cos) = angle.sin_cos();
    residual
        .iter()
        .zip(tangent)
        .map(|(x, g)| {
            let result = x * cos + (g / safe_tangent_norm) * safe_radius * sin;
            if result.is_finite() {
                Ok(result)
            } else {
                Err(GeodesicError::NonFinite)
            }
        })
        .collect()
}

/// Invalid inputs or non-finite intermediate results.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeodesicError {
    InvalidDimensions,
    InvalidLayer,
    NonFinite,
}

impl fmt::Display for GeodesicError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidDimensions => "residual and branch must be non-empty and equally sized",
            Self::InvalidLayer => "layer index is too large",
            Self::NonFinite => "non-finite input or geodesic intermediate",
        })
    }
}

impl Error for GeodesicError {}
