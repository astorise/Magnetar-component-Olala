//! Initial Olala model component support for Magnetar.
//!
//! Reads the checkpoint's ordered layer layout. Graph production, recurrent
//! execution and inference are not implemented yet.

use serde::Deserialize;
use std::error::Error;
use std::fmt;

pub mod geodesic;
pub mod mamba3;
pub mod routing;

const MAX_LAYERS: usize = 4096;

/// A layer type supported by the reference Olala mini modeling code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    Mamba3,
    DifferentialTpaAttention,
}

/// The ordered mixer layers described by an Olala checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerLayout {
    layers: Vec<LayerKind>,
}

impl LayerLayout {
    /// Parse the model type, layer count, and `layers_config` from a Hugging
    /// Face configuration. `M` denotes Mamba-3 and `V` differential TPA
    /// attention. Unknown forms are rejected rather than silently mapped.
    ///
    /// This validates only the layer layout, not the remaining configuration
    /// or execution support for a checkpoint.
    pub fn from_huggingface_config(bytes: &[u8]) -> Result<Self, ConfigError> {
        let config: RawConfig = serde_json::from_slice(bytes)
            .map_err(|error| ConfigError(format!("invalid Olala config: {error}")))?;
        if config.model_type != "olala" {
            return Err(ConfigError(format!(
                "expected model_type olala, got {:?}",
                config.model_type
            )));
        }
        if !(1..=MAX_LAYERS).contains(&config.num_hidden_layers) {
            return Err(ConfigError(format!(
                "num_hidden_layers must be between 1 and {MAX_LAYERS}"
            )));
        }
        let layers = config
            .layers_config
            .chars()
            .enumerate()
            .map(|(index, kind)| match kind {
                'M' => Ok(LayerKind::Mamba3),
                'V' => Ok(LayerKind::DifferentialTpaAttention),
                _ => Err(ConfigError(format!(
                    "unsupported layers_config[{index}] value {kind:?}"
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layers.len() != config.num_hidden_layers {
            return Err(ConfigError(format!(
                "layers_config has {} entries, expected {}",
                layers.len(),
                config.num_hidden_layers
            )));
        }
        Ok(Self { layers })
    }

    /// The sequence of mixers in model layer order.
    pub fn layers(&self) -> &[LayerKind] {
        &self.layers
    }
}

#[derive(Deserialize)]
struct RawConfig {
    model_type: String,
    num_hidden_layers: usize,
    layers_config: String,
}

/// A malformed or unsupported layer configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConfigError(String);

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ConfigError {}
