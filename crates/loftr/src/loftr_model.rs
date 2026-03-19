use std::path::Path;

use serde::Serialize;

use tch::{
    Device, Kind, Tensor,
    nn::{self, VarStore},
};

use crate::{
    backbone::{Backbone, build_backbone},
    coarse_matching::{CoarseMatching, CoarseMatchingData, CoarseMatchingOutput},
    error::LoftrError,
    fine_matching::{FineMatching, FineMatchingData},
    fine_preprocess::{FinePreprocess, FinePreprocessData},
    loftr::{LoftrMatches, normalize_loftr_image},
    loftr_config::LoftrConfig,
    position_encoding::PositionEncodingSine,
    transformer::LocalFeatureTransformer,
};

#[derive(Debug)]
pub struct LoFTRModel {
    config: LoftrConfig,
    var_store: VarStore,
    backbone: Backbone,
    pos_encoding: PositionEncodingSine,
    loftr_coarse: LocalFeatureTransformer,
    coarse_matching: CoarseMatching,
    fine_preprocess: FinePreprocess,
    loftr_fine: LocalFeatureTransformer,
}

#[derive(Debug, Serialize)]
pub struct TensorDebugStats {
    shape: Vec<i64>,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    abs_mean: f64,
    l2_norm: f64,
    sample: Vec<f64>,
}

#[derive(Debug, Serialize)]
pub struct CoarseDebugStats {
    conf_matrix: TensorDebugStats,
    threshold_count: i64,
    mutual_count: i64,
    match_count: i64,
    confidence_mean: f64,
    confidence_max: f64,
}

#[derive(Debug, Serialize)]
pub struct LoftrDebugStages {
    image0: TensorDebugStats,
    image1: TensorDebugStats,
    feat_c0_backbone: TensorDebugStats,
    feat_c1_backbone: TensorDebugStats,
    feat_f0_backbone: TensorDebugStats,
    feat_f1_backbone: TensorDebugStats,
    feat_c0_pos: TensorDebugStats,
    feat_c1_pos: TensorDebugStats,
    feat_c0_coarse: TensorDebugStats,
    feat_c1_coarse: TensorDebugStats,
    coarse: CoarseDebugStats,
}

impl LoFTRModel {
    /// Builds a `LoFTR` model on the requested device.
    ///
    /// # Errors
    ///
    /// Returns an error when the model configuration is invalid or one of the
    /// internal modules cannot be constructed.
    pub fn new(device: Device, config: LoftrConfig) -> Result<Self, LoftrError> {
        let var_store = nn::VarStore::new(device);
        let root = var_store.root();
        let backbone = build_backbone(&root, &config)?;
        let pos_encoding = PositionEncodingSine::new(
            config.coarse.d_model,
            (256, 256),
            config.coarse.temp_bug_fix,
            device,
        )?;
        let loftr_coarse =
            LocalFeatureTransformer::new(&(root.clone() / "loftr_coarse"), &config.coarse)?;
        let coarse_matching = CoarseMatching::new(&config.match_coarse)?;
        let fine_preprocess = FinePreprocess::new(&(root.clone() / "fine_preprocess"), &config)?;
        let fine_transformer_config = crate::loftr_config::TransformerConfig {
            d_model: config.fine.d_model,
            d_ffn: config.fine.d_ffn,
            nhead: config.fine.nhead,
            layer_names: config.fine.layer_names.clone(),
            attention: config.fine.attention.clone(),
            temp_bug_fix: false,
        };
        let loftr_fine =
            LocalFeatureTransformer::new(&(root / "loftr_fine"), &fine_transformer_config)?;
        Ok(Self {
            config,
            var_store,
            backbone,
            pos_encoding,
            loftr_coarse,
            coarse_matching,
            fine_preprocess,
            loftr_fine,
        })
    }

    #[must_use]
    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.var_store
    }

    /// Loads model weights from a `safetensors` or Torch-compatible checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error when the file cannot be read or does not match the
    /// variables defined by this model.
    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<(), LoftrError> {
        self.var_store.load(path).map_err(LoftrError::from)
    }

    /// Runs `LoFTR` inference for one batch of image pairs.
    ///
    /// # Errors
    ///
    /// Returns an error when the input shapes are unsupported or if any stage of
    /// the network execution fails.
    pub fn forward(
        &mut self,
        image0: &Tensor,
        image1: &Tensor,
    ) -> Result<LoftrMatches, LoftrError> {
        let stages = self.forward_with_debug(image0, image1)?;
        Ok(stages.matches)
    }

    /// Runs `LoFTR` inference and returns intermediate debug statistics.
    ///
    /// # Errors
    ///
    /// Returns an error when the input shapes are unsupported or if any stage of
    /// the network execution fails.
    pub fn forward_debug(
        &mut self,
        image0: &Tensor,
        image1: &Tensor,
    ) -> Result<LoftrDebugStages, LoftrError> {
        let stages = self.forward_with_debug(image0, image1)?;
        Ok(stages.debug)
    }

    fn forward_with_debug(
        &mut self,
        image0: &Tensor,
        image1: &Tensor,
    ) -> Result<ForwardWithDebug, LoftrError> {
        let image0 = normalize_loftr_image(image0, self.var_store.device())?;
        let image1 = normalize_loftr_image(image1, self.var_store.device())?;
        let batch_size = image0.size()[0];
        let hw0_i = (image0.size()[2], image0.size()[3]);
        let hw1_i = (image1.size()[2], image1.size()[3]);

        let backbone_features =
            self.backbone_features(&image0, &image1, batch_size, hw0_i, hw1_i)?;
        let coarse_features = self.coarse_features(&backbone_features, batch_size)?;

        let coarse = self.coarse_matching.forward(
            &coarse_features.transformed0,
            &coarse_features.transformed1,
            &CoarseMatchingData {
                hw0_i,
                hw1_i,
                hw0_c: coarse_features.hw0_c,
                hw1_c: coarse_features.hw1_c,
                scale0: None,
                scale1: None,
            },
            None,
            None,
        )?;

        let (mut feat_f0_unfold, mut feat_f1_unfold) = self.fine_preprocess.forward(
            &backbone_features.fine0,
            &backbone_features.fine1,
            &coarse_features.transformed0,
            &coarse_features.transformed1,
            &FinePreprocessData {
                hw0_f: coarse_features.hw0_f,
                hw0_c: coarse_features.hw0_c,
                b_ids: coarse.b_ids.shallow_clone(),
                i_ids: coarse.i_ids.shallow_clone(),
                j_ids: coarse.j_ids.shallow_clone(),
            },
        )?;
        if feat_f0_unfold.size()[0] != 0 {
            let (next0, next1) =
                self.loftr_fine
                    .forward(&feat_f0_unfold, &feat_f1_unfold, None, None)?;
            feat_f0_unfold = next0;
            feat_f1_unfold = next1;
        }

        let fine = FineMatching::forward(
            &feat_f0_unfold,
            &feat_f1_unfold,
            &FineMatchingData {
                hw0_i,
                hw0_f: coarse_features.hw0_f,
                mkpts0_c: coarse.mkpts0_c.shallow_clone(),
                mkpts1_c: coarse.mkpts1_c.shallow_clone(),
                mconf: coarse.mconf.shallow_clone(),
                b_ids: coarse.m_bids.shallow_clone(),
                scale1: None,
            },
        )?;

        Ok(ForwardWithDebug {
            matches: LoftrMatches {
                keypoints0: fine.mkpts0_f,
                keypoints1: fine.mkpts1_f,
                confidence: coarse.mconf.shallow_clone(),
                batch_indexes: coarse.m_bids.shallow_clone(),
            },
            debug: self.build_debug_stages(
                &image0,
                &image1,
                &backbone_features,
                &coarse_features,
                &coarse,
            ),
        })
    }

    fn backbone_features(
        &self,
        image0: &Tensor,
        image1: &Tensor,
        batch_size: i64,
        hw0_i: (i64, i64),
        hw1_i: (i64, i64),
    ) -> Result<BackboneFeatures, LoftrError> {
        if hw0_i == hw1_i {
            let stacked_images = Tensor::cat(&[image0.shallow_clone(), image1.shallow_clone()], 0);
            let (coarse_backbone, fine_backbone) =
                self.backbone.forward_t(&stacked_images, false)?;
            let coarse_backbone = coarse_backbone.split(batch_size, 0);
            let fine_backbone = fine_backbone.split(batch_size, 0);
            return Ok(BackboneFeatures {
                coarse0: coarse_backbone[0].shallow_clone(),
                coarse1: coarse_backbone[1].shallow_clone(),
                fine0: fine_backbone[0].shallow_clone(),
                fine1: fine_backbone[1].shallow_clone(),
            });
        }

        let (coarse0, fine0) = self.backbone.forward_t(image0, false)?;
        let (coarse1, fine1) = self.backbone.forward_t(image1, false)?;
        Ok(BackboneFeatures {
            coarse0,
            coarse1,
            fine0,
            fine1,
        })
    }

    fn coarse_features(
        &mut self,
        backbone_features: &BackboneFeatures,
        batch_size: i64,
    ) -> Result<CoarseFeatures, LoftrError> {
        let hw0_c = (
            backbone_features.coarse0.size()[2],
            backbone_features.coarse0.size()[3],
        );
        let hw1_c = (
            backbone_features.coarse1.size()[2],
            backbone_features.coarse1.size()[3],
        );
        let hw0_f = (
            backbone_features.fine0.size()[2],
            backbone_features.fine0.size()[3],
        );

        let positional0 = self
            .pos_encoding
            .forward(&backbone_features.coarse0)?
            .permute([0, 2, 3, 1])
            .reshape([batch_size, -1, self.config.coarse.d_model]);
        let positional1 = self
            .pos_encoding
            .forward(&backbone_features.coarse1)?
            .permute([0, 2, 3, 1])
            .reshape([batch_size, -1, self.config.coarse.d_model]);
        let (transformed0, transformed1) =
            self.loftr_coarse
                .forward(&positional0, &positional1, None, None)?;

        Ok(CoarseFeatures {
            hw0_c,
            hw1_c,
            hw0_f,
            positional0,
            positional1,
            transformed0,
            transformed1,
        })
    }

    fn build_debug_stages(
        &self,
        image0: &Tensor,
        image1: &Tensor,
        backbone_features: &BackboneFeatures,
        coarse_features: &CoarseFeatures,
        coarse: &CoarseMatchingOutput,
    ) -> LoftrDebugStages {
        LoftrDebugStages {
            image0: tensor_debug_stats(image0),
            image1: tensor_debug_stats(image1),
            feat_c0_backbone: tensor_debug_stats(&backbone_features.coarse0),
            feat_c1_backbone: tensor_debug_stats(&backbone_features.coarse1),
            feat_f0_backbone: tensor_debug_stats(&backbone_features.fine0),
            feat_f1_backbone: tensor_debug_stats(&backbone_features.fine1),
            feat_c0_pos: tensor_debug_stats(&coarse_features.positional0),
            feat_c1_pos: tensor_debug_stats(&coarse_features.positional1),
            feat_c0_coarse: tensor_debug_stats(&coarse_features.transformed0),
            feat_c1_coarse: tensor_debug_stats(&coarse_features.transformed1),
            coarse: CoarseDebugStats {
                conf_matrix: tensor_debug_stats(&coarse.conf_matrix),
                threshold_count: coarse
                    .conf_matrix
                    .gt(self.config.match_coarse.thr)
                    .sum(Kind::Int64)
                    .int64_value(&[]),
                mutual_count: confidence_mutual_count(&coarse.conf_matrix),
                match_count: coarse.mconf.size()[0],
                confidence_mean: mean_or_zero(&coarse.mconf),
                confidence_max: max_or_zero(&coarse.mconf),
            },
        }
    }
}

#[derive(Debug)]
struct ForwardWithDebug {
    matches: LoftrMatches,
    debug: LoftrDebugStages,
}

#[derive(Debug)]
struct BackboneFeatures {
    coarse0: Tensor,
    coarse1: Tensor,
    fine0: Tensor,
    fine1: Tensor,
}

#[derive(Debug)]
struct CoarseFeatures {
    hw0_c: (i64, i64),
    hw1_c: (i64, i64),
    hw0_f: (i64, i64),
    positional0: Tensor,
    positional1: Tensor,
    transformed0: Tensor,
    transformed1: Tensor,
}

fn tensor_debug_stats(tensor: &Tensor) -> TensorDebugStats {
    // Debug stats should never crash calibration; if CPU transfer fails,
    // continue with the original tensor as a best-effort fallback.
    let tensor = match tensor.f_to_device(Device::Cpu) {
        Ok(cpu_tensor) => cpu_tensor,
        Err(_) => tensor.shallow_clone(),
    };
    let flat = tensor.reshape([-1]).to_kind(Kind::Float);
    let sample_len = flat.size()[0].min(8);
    let mut sample = Vec::new();
    for index in 0..sample_len {
        sample.push(flat.double_value(&[index]));
    }
    TensorDebugStats {
        shape: tensor.size(),
        mean: flat.mean(Kind::Float).double_value(&[]),
        std: flat.std(true).double_value(&[]),
        min: flat.min().double_value(&[]),
        max: flat.max().double_value(&[]),
        abs_mean: flat.abs().mean(Kind::Float).double_value(&[]),
        l2_norm: flat
            .pow_tensor_scalar(2.0)
            .sum(Kind::Float)
            .sqrt()
            .double_value(&[]),
        sample,
    }
}

fn confidence_mutual_count(conf_matrix: &Tensor) -> i64 {
    conf_matrix
        .eq_tensor(&conf_matrix.max_dim(2, true).0)
        .logical_and(&conf_matrix.eq_tensor(&conf_matrix.max_dim(1, true).0))
        .sum(Kind::Int64)
        .int64_value(&[])
}

fn mean_or_zero(tensor: &Tensor) -> f64 {
    if tensor.numel() == 0 {
        0.0
    } else {
        tensor.mean(Kind::Float).double_value(&[])
    }
}

fn max_or_zero(tensor: &Tensor) -> f64 {
    if tensor.numel() == 0 {
        0.0
    } else {
        tensor.max().double_value(&[])
    }
}

#[cfg(test)]
mod tests;
