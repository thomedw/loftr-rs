//! Native Rust/tch `LoFTR` feature matching.
//!
//! This crate exposes a small, high-level API for constructing a `LoFTR` model,
//! loading compatible weights, and matching a pair of images.
//!
//! Typical usage:
//!
//! 1. Build a [`LoftrModel`] with a preset [`LoftrConfig`].
//! 2. Load exported weights with [`LoftrModel::load_weights`].
//! 3. Run [`LoftrModel::forward`] or [`LoftrModel::forward_debug`].
//!
//! The published docs are built with the `doc-only` feature so they can render
//! on docs.rs without linking libtorch. For local runs, either point `tch` at
//! an existing libtorch install or enable the `download-libtorch` feature.
//!
//! # Example
//!
//! ```no_run
//! use loftr::{LoftrConfig, LoftrModel};
//! use tch::{Device, Kind, Tensor};
//!
//! let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
//! model.load_weights("artifacts/weights/loftr_outdoor_state_dict.safetensors")?;
//!
//! let image0 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
//! let image1 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
//! let matches = model.forward(&image0, &image1)?;
//!
//! println!("match count = {}", matches.confidence.size()[0]);
//! # Ok::<(), loftr::LoftrError>(())
//! ```

mod backbone;
mod coarse_matching;
mod error;
mod fine_matching;
mod fine_preprocess;
mod linear_attention;
mod loftr;
mod loftr_config;
mod loftr_model;
mod numeric;
mod position_encoding;
mod transformer;

pub use crate::error::LoftrError;
pub use crate::loftr::{LoftrMatches, normalize_loftr_image};
pub use crate::loftr_config::{
    AttentionType, BackboneType, FineConfig, LoftrConfig, MatchCoarseConfig, MatchType,
    ResNetFpnConfig, TransformerConfig, TransformerLayerKind,
};
pub use crate::loftr_model::{
    CoarseDebugStats, LoFTRModel as LoftrModel, LoftrDebugStages, TensorDebugStats,
};
