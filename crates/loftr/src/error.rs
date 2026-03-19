use thiserror::Error;

/// Errors returned by the public `loftr` API.
///
/// These errors cover invalid model configuration, invalid runtime inputs, and
/// failures bubbled up from the underlying `tch` bindings.
#[derive(Debug, Error)]
pub enum LoftrError {
    /// The `LoFTR` configuration is internally inconsistent or unsupported.
    #[error("invalid LoFTR config: {0}")]
    InvalidConfig(String),
    /// The provided input tensors do not match the supported image layout.
    #[error("invalid LoFTR input: {0}")]
    InvalidInput(String),
    /// An error returned by the underlying `tch` or libtorch runtime.
    #[error("tch error: {0}")]
    Tch(#[from] tch::TchError),
}
