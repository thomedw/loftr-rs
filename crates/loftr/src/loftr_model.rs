use std::path::Path;

use serde::Serialize;

use tch::{
    Device, Kind, Tensor,
    nn::{self, VarStore},
};

use crate::{
    backbone::{Backbone, build_backbone},
    coarse_matching::{CoarseMatching, CoarseMatchingData},
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
    fine_matching: FineMatching,
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
            fine_matching: FineMatching,
        })
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        &mut self.var_store
    }

    pub fn load_weights<P: AsRef<Path>>(&mut self, path: P) -> Result<(), LoftrError> {
        self.var_store.load(path).map_err(LoftrError::from)
    }

    pub fn forward(
        &mut self,
        image0: &Tensor,
        image1: &Tensor,
    ) -> Result<LoftrMatches, LoftrError> {
        let stages = self.forward_with_debug(image0, image1)?;
        Ok(stages.matches)
    }

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

        let ((feat_c0_backbone, feat_f0_backbone), (feat_c1_backbone, feat_f1_backbone)) =
            if hw0_i == hw1_i {
                let images = Tensor::cat(&[image0.shallow_clone(), image1.shallow_clone()], 0);
                let (feat_c, feat_f) = self.backbone.forward_t(&images, false)?;
                let feat_c = feat_c.split(batch_size, 0);
                let feat_f = feat_f.split(batch_size, 0);
                (
                    (feat_c[0].shallow_clone(), feat_f[0].shallow_clone()),
                    (feat_c[1].shallow_clone(), feat_f[1].shallow_clone()),
                )
            } else {
                (
                    self.backbone.forward_t(&image0, false)?,
                    self.backbone.forward_t(&image1, false)?,
                )
            };

        let hw0_c = (feat_c0_backbone.size()[2], feat_c0_backbone.size()[3]);
        let hw1_c = (feat_c1_backbone.size()[2], feat_c1_backbone.size()[3]);
        let hw0_f = (feat_f0_backbone.size()[2], feat_f0_backbone.size()[3]);

        let feat_c0_pos = self
            .pos_encoding
            .forward(&feat_c0_backbone)?
            .permute([0, 2, 3, 1])
            .reshape([batch_size, -1, self.config.coarse.d_model]);
        let feat_c1_pos = self
            .pos_encoding
            .forward(&feat_c1_backbone)?
            .permute([0, 2, 3, 1])
            .reshape([batch_size, -1, self.config.coarse.d_model]);
        let (feat_c0_coarse, feat_c1_coarse) =
            self.loftr_coarse
                .forward(&feat_c0_pos, &feat_c1_pos, None, None)?;

        let coarse = self.coarse_matching.forward(
            &feat_c0_coarse,
            &feat_c1_coarse,
            &CoarseMatchingData {
                hw0_i,
                hw1_i,
                hw0_c,
                hw1_c,
                scale0: None,
                scale1: None,
            },
            None,
            None,
        )?;

        let (mut feat_f0_unfold, mut feat_f1_unfold) = self.fine_preprocess.forward(
            &feat_f0_backbone,
            &feat_f1_backbone,
            &feat_c0_coarse,
            &feat_c1_coarse,
            &FinePreprocessData {
                hw0_f,
                hw0_c,
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

        let fine = self.fine_matching.forward(
            &feat_f0_unfold,
            &feat_f1_unfold,
            &FineMatchingData {
                hw0_i,
                hw0_f,
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
            debug: LoftrDebugStages {
                image0: tensor_debug_stats(&image0),
                image1: tensor_debug_stats(&image1),
                feat_c0_backbone: tensor_debug_stats(&feat_c0_backbone),
                feat_c1_backbone: tensor_debug_stats(&feat_c1_backbone),
                feat_f0_backbone: tensor_debug_stats(&feat_f0_backbone),
                feat_f1_backbone: tensor_debug_stats(&feat_f1_backbone),
                feat_c0_pos: tensor_debug_stats(&feat_c0_pos),
                feat_c1_pos: tensor_debug_stats(&feat_c1_pos),
                feat_c0_coarse: tensor_debug_stats(&feat_c0_coarse),
                feat_c1_coarse: tensor_debug_stats(&feat_c1_coarse),
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
            },
        })
    }
}

#[derive(Debug)]
struct ForwardWithDebug {
    matches: LoftrMatches,
    debug: LoftrDebugStages,
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
    let mut sample = Vec::with_capacity(sample_len as usize);
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
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::{env, fs, path::Path};

    use image::{DynamicImage, imageops::FilterType};
    use tch::Tensor;

    #[test]
    fn model_variable_names_match_kornia_prefixes() {
        let model = LoFTRModel::new(Device::Cpu, LoftrConfig::outdoor()).expect("model");
        let vars = model.var_store().variables();
        for name in [
            "backbone.conv1.weight",
            "backbone.layer1.0.conv1.weight",
            "fine_preprocess.down_proj.weight",
            "loftr_coarse.layers.0.q_proj.weight",
            "loftr_fine.layers.0.q_proj.weight",
        ] {
            assert!(vars.contains_key(name), "missing variable `{name}`");
        }
    }

    #[test]
    fn model_forward_smoke_returns_consistent_output_shapes() {
        let mut model = LoFTRModel::new(Device::Cpu, LoftrConfig::outdoor()).expect("model");
        let image0 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
        let image1 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
        let out = model.forward(&image0, &image1).expect("forward");
        assert_eq!(out.keypoints0.size().len(), 2);
        assert_eq!(out.keypoints1.size().len(), 2);
        assert_eq!(out.confidence.size().len(), 1);
        assert_eq!(out.batch_indexes.size().len(), 1);
        assert_eq!(out.keypoints0.size()[0], out.keypoints1.size()[0]);
        assert_eq!(out.keypoints0.size()[0], out.confidence.size()[0]);
        assert_eq!(out.keypoints0.size()[0], out.batch_indexes.size()[0]);
        assert_eq!(out.keypoints0.size()[1], 2);
        assert_eq!(out.keypoints1.size()[1], 2);
    }

    #[test]
    #[ignore = "requires local weights and image paths via environment variables"]
    fn exported_weights_match_local_pair() {
        let weights = env::var("LOFTR_TEST_WEIGHTS").expect("LOFTR_TEST_WEIGHTS");
        let left = env::var("LOFTR_TEST_LEFT").expect("LOFTR_TEST_LEFT");
        let right = env::var("LOFTR_TEST_RIGHT").expect("LOFTR_TEST_RIGHT");

        let mut model = LoFTRModel::new(Device::Cpu, LoftrConfig::outdoor()).expect("model");
        model
            .load_weights(Path::new(&weights))
            .expect("load weights");

        let image0 = load_local_grayscale(Path::new(&left));
        let image1 = load_local_grayscale(Path::new(&right));
        let out = model.forward(&image0, &image1).expect("forward");
        assert_eq!(out.keypoints0.size()[1], 2);
        assert_eq!(out.keypoints1.size()[1], 2);
        assert_eq!(out.confidence.size()[0], out.keypoints0.size()[0]);
    }

    #[test]
    #[ignore = "writes Rust LoFTR stage stats for Python comparison"]
    fn dump_local_stage_stats() {
        let weights = env::var("LOFTR_TEST_WEIGHTS").expect("LOFTR_TEST_WEIGHTS");
        let left = env::var("LOFTR_TEST_LEFT").expect("LOFTR_TEST_LEFT");
        let right = env::var("LOFTR_TEST_RIGHT").expect("LOFTR_TEST_RIGHT");
        let output = env::var("LOFTR_STAGE_DUMP")
            .unwrap_or_else(|_| String::from("target/loftr_stage_stats_rust.json"));

        let mut model = LoFTRModel::new(Device::Cpu, LoftrConfig::outdoor()).expect("model");
        model.load_weights(&weights).expect("load weights");

        let image0 = load_local_grayscale(Path::new(&left));
        let image1 = load_local_grayscale(Path::new(&right));
        let debug = model
            .forward_debug(&image0, &image1)
            .expect("debug forward");
        fs::write(&output, serde_json::to_vec_pretty(&debug).expect("json")).expect("write json");
        eprintln!("wrote {:?}", output);
    }

    fn load_local_grayscale(path: &Path) -> Tensor {
        let image = image::open(path).expect("load image");
        let image = resize_for_loftr(image).to_luma32f();
        let height = i64::from(image.height());
        let width = i64::from(image.width());
        let data = image.into_raw();
        Tensor::from_slice(&data)
            .view([1, height, width])
            .unsqueeze(0)
            .to_kind(Kind::Float)
    }

    fn resize_for_loftr(image: DynamicImage) -> DynamicImage {
        image.resize_exact(960, 540, FilterType::Triangle)
    }
}
