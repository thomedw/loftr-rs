use super::*;
use crate::loftr_config::LoftrConfig;
use tch::{Device, Kind, nn};

#[test]
fn encoder_layer_preserves_shape() -> Result<(), LoftrError> {
    let vs = nn::VarStore::new(Device::Cpu);
    let layer = LoFTREncoderLayer::new(&vs.root(), 256, 8, "linear")?;
    let x = Tensor::randn([2, 12, 256], (Kind::Float, Device::Cpu));
    let source = Tensor::randn([2, 15, 256], (Kind::Float, Device::Cpu));
    let out = layer.forward(&x, &source, None, None)?;
    assert_eq!(out.size(), vec![2, 12, 256]);
    Ok(())
}

#[test]
fn transformer_preserves_feature_shapes() -> Result<(), LoftrError> {
    let vs = nn::VarStore::new(Device::Cpu);
    let config = LoftrConfig::outdoor().coarse;
    let transformer = LocalFeatureTransformer::new(&vs.root(), &config)?;
    let feat0 = Tensor::randn([1, 24, 256], (Kind::Float, Device::Cpu));
    let feat1 = Tensor::randn([1, 30, 256], (Kind::Float, Device::Cpu));
    let (out0, out1) = transformer.forward(&feat0, &feat1, None, None)?;
    assert_eq!(out0.size(), vec![1, 24, 256]);
    assert_eq!(out1.size(), vec![1, 30, 256]);
    Ok(())
}

#[test]
fn transformer_rejects_wrong_feature_dim() {
    let vs = nn::VarStore::new(Device::Cpu);
    let config = LoftrConfig::outdoor().fine;
    let transformer_config = TransformerConfig {
        d_model: config.d_model,
        d_ffn: config.d_ffn,
        nhead: config.nhead,
        layer_names: config.layer_names,
        attention: config.attention,
        temp_bug_fix: false,
    };
    let transformer = match LocalFeatureTransformer::new(&vs.root(), &transformer_config) {
        Ok(transformer) => transformer,
        Err(err) => panic!("transformer construction failed unexpectedly: {err}"),
    };
    let feat0 = Tensor::randn([1, 12, 64], (Kind::Float, Device::Cpu));
    let feat1 = Tensor::randn([1, 12, 128], (Kind::Float, Device::Cpu));
    match transformer.forward(&feat0, &feat1, None, None) {
        Ok(_) => panic!("wrong feature dim should fail"),
        Err(err) => assert!(format!("{err}").contains("expected feature dim")),
    }
}
