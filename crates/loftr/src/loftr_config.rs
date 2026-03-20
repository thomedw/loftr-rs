/// Backbone variants supported by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackboneType {
    /// The `ResNetFPN` backbone used by the reference `LoFTR` models.
    ResNetFpn,
}

/// Backbone parameters for the `ResNetFPN` encoder used by `LoFTR`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResNetFpnConfig {
    /// Channel count produced by the initial stem convolution.
    pub initial_dim: i64,
    /// Output channel counts for the three residual stages.
    pub block_dims: [i64; 3],
}

/// Attention implementations supported by the `LoFTR` transformers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Linear attention used by the default `LoFTR` presets.
    Linear,
    /// Full attention supported by the internal transformer implementation.
    Full,
}

/// Transformer layer ordering used by `LoFTR` encoder stacks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransformerLayer {
    /// Attend within the same feature stream.
    SelfAttention,
    /// Attend across the left and right feature streams.
    CrossAttention,
}

/// Transformer parameters shared by `LoFTR` coarse and fine stages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerConfig {
    /// Token embedding width.
    pub d_model: i64,
    /// Feed-forward hidden width inside each encoder layer.
    pub d_ffn: i64,
    /// Number of attention heads.
    pub nhead: i64,
    /// Ordered encoder stage pattern, usually alternating self and cross attention.
    pub layers: Vec<TransformerLayer>,
    /// Attention implementation used by each encoder layer.
    pub attention: AttentionType,
    /// Whether to use Kornia's temperature bug-fix behavior in positional encoding.
    pub temp_bug_fix: bool,
}

/// Coarse matcher implementations supported by this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchType {
    /// Dual-softmax matching from the reference `LoFTR` models.
    DualSoftmax,
}

/// Matching parameters for the coarse `LoFTR` correspondence stage.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchCoarseConfig {
    /// Confidence threshold applied before mutual-nearest filtering.
    pub thr: f64,
    /// Number of border cells to suppress on each coarse feature map edge.
    pub border_rm: i64,
    /// Coarse matcher implementation to use.
    pub match_type: MatchType,
    /// Dual-softmax temperature scaling factor.
    pub dsmax_temperature: f64,
    /// Sinkhorn iteration count from the reference config.
    pub skh_iters: i64,
    /// Initial Sinkhorn bin score from the reference config.
    pub skh_init_bin_score: f64,
    /// Whether Sinkhorn prefiltering is enabled in the reference config.
    pub skh_prefilter: bool,
    /// Training-time coarse supervision percentage from the reference config.
    pub train_coarse_percent: f64,
    /// Minimum padding count for training-time coarse supervision.
    pub train_pad_num_gt_min: i64,
}

/// Transformer parameters for the fine matching refinement stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FineConfig {
    /// Token embedding width.
    pub d_model: i64,
    /// Feed-forward hidden width inside each encoder layer.
    pub d_ffn: i64,
    /// Number of attention heads.
    pub nhead: i64,
    /// Ordered encoder stage pattern for the fine transformer.
    pub layers: Vec<TransformerLayer>,
    /// Attention implementation used by each encoder layer.
    pub attention: AttentionType,
}

/// High-level `LoFTR` model configuration.
///
/// Most users should start from [`LoftrConfig::outdoor`] and only override
/// fields when matching a known reference setup.
#[derive(Debug, Clone, PartialEq)]
pub struct LoftrConfig {
    /// Backbone implementation to use.
    pub backbone_type: BackboneType,
    /// Output stride pair `(coarse_stride, fine_stride)`.
    pub resolution: (i64, i64),
    /// Fine-stage local window size in feature-grid cells.
    pub fine_window_size: i64,
    /// Whether fine matching concatenates coarse token features before projection.
    pub fine_concat_coarse_feat: bool,
    /// ResNet-FPN backbone parameters.
    pub resnetfpn: ResNetFpnConfig,
    /// Coarse transformer parameters.
    pub coarse: TransformerConfig,
    /// Coarse matching parameters.
    pub match_coarse: MatchCoarseConfig,
    /// Fine transformer parameters.
    pub fine: FineConfig,
}

impl LoftrConfig {
    /// Returns the default outdoor `LoFTR` preset used by this crate.
    ///
    /// This is also the [`Default`] configuration.
    #[must_use]
    pub fn outdoor() -> Self {
        Self {
            backbone_type: BackboneType::ResNetFpn,
            resolution: (8, 2),
            fine_window_size: 5,
            fine_concat_coarse_feat: true,
            resnetfpn: ResNetFpnConfig {
                initial_dim: 128,
                block_dims: [128, 196, 256],
            },
            coarse: TransformerConfig {
                d_model: 256,
                d_ffn: 256,
                nhead: 8,
                layers: vec![
                    TransformerLayer::SelfAttention,
                    TransformerLayer::CrossAttention,
                    TransformerLayer::SelfAttention,
                    TransformerLayer::CrossAttention,
                    TransformerLayer::SelfAttention,
                    TransformerLayer::CrossAttention,
                    TransformerLayer::SelfAttention,
                    TransformerLayer::CrossAttention,
                ],
                attention: AttentionType::Linear,
                temp_bug_fix: false,
            },
            match_coarse: MatchCoarseConfig {
                thr: 0.2,
                border_rm: 2,
                match_type: MatchType::DualSoftmax,
                dsmax_temperature: 0.1,
                skh_iters: 3,
                skh_init_bin_score: 1.0,
                skh_prefilter: true,
                train_coarse_percent: 0.4,
                train_pad_num_gt_min: 200,
            },
            fine: FineConfig {
                d_model: 128,
                d_ffn: 128,
                nhead: 8,
                layers: vec![
                    TransformerLayer::SelfAttention,
                    TransformerLayer::CrossAttention,
                ],
                attention: AttentionType::Linear,
            },
        }
    }

    /// Returns the legacy indoor `LoFTR` preset used by Kornia's shared fixtures.
    #[must_use]
    pub fn indoor() -> Self {
        Self::outdoor()
    }

    /// Returns the indoor `LoFTR` preset with Kornia's temperature fix enabled.
    #[must_use]
    pub fn indoor_new() -> Self {
        let mut config = Self::indoor();
        config.coarse.temp_bug_fix = true;
        config
    }
}

impl Default for LoftrConfig {
    fn default() -> Self {
        Self::outdoor()
    }
}

#[cfg(test)]
mod tests;
