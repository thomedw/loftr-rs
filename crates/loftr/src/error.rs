use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoftrError {
    #[error("invalid LoFTR config: {0}")]
    InvalidConfig(String),
    #[error("invalid LoFTR input: {0}")]
    InvalidInput(String),
    #[error("tch error: {0}")]
    Tch(#[from] tch::TchError),
}
