use tch::{
    Tensor,
    nn::{self},
};

use crate::{
    error::LoftrError,
    linear_attention::{FullAttention, LinearAttention},
    loftr_config::TransformerConfig,
};

#[derive(Debug)]
enum AttentionKind {
    Linear(LinearAttention),
    Full(FullAttention),
}

impl AttentionKind {
    fn forward(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        q_mask: Option<&Tensor>,
        kv_mask: Option<&Tensor>,
    ) -> Result<Tensor, LoftrError> {
        match self {
            Self::Linear(attention) => attention.forward(queries, keys, values, q_mask, kv_mask),
            Self::Full(attention) => attention.forward(queries, keys, values, q_mask, kv_mask),
        }
    }
}

#[derive(Debug)]
pub struct LoFTREncoderLayer {
    dim: i64,
    nhead: i64,
    q_proj: nn::Linear,
    k_proj: nn::Linear,
    v_proj: nn::Linear,
    attention: AttentionKind,
    merge: nn::Linear,
    mlp: nn::Sequential,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
}

impl LoFTREncoderLayer {
    pub fn new(
        vs: &nn::Path<'_>,
        d_model: i64,
        nhead: i64,
        attention: &str,
    ) -> Result<Self, LoftrError> {
        if d_model <= 0 || nhead <= 0 || d_model % nhead != 0 {
            return Err(LoftrError::InvalidConfig(format!(
                "LoFTREncoderLayer requires positive d_model and nhead, with d_model divisible by nhead; got d_model={d_model}, nhead={nhead}"
            )));
        }

        let attention = if attention == "linear" {
            AttentionKind::Linear(LinearAttention::default())
        } else {
            AttentionKind::Full(FullAttention::default())
        };
        let linear_config = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        let mlp = nn::seq()
            .add(nn::linear(
                vs / "mlp" / "0",
                d_model * 2,
                d_model * 2,
                linear_config,
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                vs / "mlp" / "2",
                d_model * 2,
                d_model,
                linear_config,
            ));
        let layer_norm_config = nn::LayerNormConfig::default();

        Ok(Self {
            dim: d_model / nhead,
            nhead,
            q_proj: nn::linear(vs / "q_proj", d_model, d_model, linear_config),
            k_proj: nn::linear(vs / "k_proj", d_model, d_model, linear_config),
            v_proj: nn::linear(vs / "v_proj", d_model, d_model, linear_config),
            attention,
            merge: nn::linear(vs / "merge", d_model, d_model, linear_config),
            mlp,
            norm1: nn::layer_norm(vs / "norm1", vec![d_model], layer_norm_config),
            norm2: nn::layer_norm(vs / "norm2", vec![d_model], layer_norm_config),
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        source: &Tensor,
        x_mask: Option<&Tensor>,
        source_mask: Option<&Tensor>,
    ) -> Result<Tensor, LoftrError> {
        validate_sequence_tensor(x, "x")?;
        validate_sequence_tensor(source, "source")?;
        if x.size()[0] != source.size()[0] || x.size()[2] != source.size()[2] {
            return Err(LoftrError::InvalidConfig(format!(
                "LoFTREncoderLayer source mismatch: x={:?}, source={:?}",
                x.size(),
                source.size()
            )));
        }

        let batch_size = x.size()[0];
        let query = x
            .apply(&self.q_proj)
            .view([batch_size, -1, self.nhead, self.dim]);
        let key = source
            .apply(&self.k_proj)
            .view([batch_size, -1, self.nhead, self.dim]);
        let value = source
            .apply(&self.v_proj)
            .view([batch_size, -1, self.nhead, self.dim]);
        let message = self
            .attention
            .forward(&query, &key, &value, x_mask, source_mask)?
            .view([batch_size, -1, self.nhead * self.dim])
            .apply(&self.merge)
            .apply(&self.norm1);
        let message = Tensor::cat(&[x, &message], 2)
            .apply(&self.mlp)
            .apply(&self.norm2);
        Ok(x + message)
    }
}

#[derive(Debug)]
pub struct LocalFeatureTransformer {
    d_model: i64,
    layer_names: Vec<String>,
    layers: Vec<LoFTREncoderLayer>,
}

impl LocalFeatureTransformer {
    pub fn new(vs: &nn::Path<'_>, config: &TransformerConfig) -> Result<Self, LoftrError> {
        let mut layers = Vec::with_capacity(config.layer_names.len());
        for (index, _) in config.layer_names.iter().enumerate() {
            layers.push(LoFTREncoderLayer::new(
                &(vs / "layers" / index.to_string()),
                config.d_model,
                config.nhead,
                &config.attention,
            )?);
        }

        Ok(Self {
            d_model: config.d_model,
            layer_names: config.layer_names.clone(),
            layers,
        })
    }

    pub fn forward(
        &self,
        feat0: &Tensor,
        feat1: &Tensor,
        mask0: Option<&Tensor>,
        mask1: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor), LoftrError> {
        validate_sequence_tensor(feat0, "feat0")?;
        validate_sequence_tensor(feat1, "feat1")?;
        if feat0.size()[2] != self.d_model || feat1.size()[2] != self.d_model {
            return Err(LoftrError::InvalidConfig(format!(
                "LocalFeatureTransformer expected feature dim {}, got feat0={:?}, feat1={:?}",
                self.d_model,
                feat0.size(),
                feat1.size()
            )));
        }

        let mut feat0 = feat0.shallow_clone();
        let mut feat1 = feat1.shallow_clone();
        for (layer, name) in self.layers.iter().zip(self.layer_names.iter()) {
            match name.as_str() {
                "self" => {
                    feat0 = layer.forward(&feat0, &feat0, mask0, mask0)?;
                    feat1 = layer.forward(&feat1, &feat1, mask1, mask1)?;
                }
                "cross" => {
                    let next0 = layer.forward(&feat0, &feat1, mask0, mask1)?;
                    let next1 = layer.forward(&feat1, &next0, mask1, mask0)?;
                    feat0 = next0;
                    feat1 = next1;
                }
                other => {
                    return Err(LoftrError::InvalidConfig(format!(
                        "LocalFeatureTransformer unknown layer name `{other}`"
                    )));
                }
            }
        }

        Ok((feat0, feat1))
    }
}

fn validate_sequence_tensor(tensor: &Tensor, label: &str) -> Result<(), LoftrError> {
    let dims = tensor.size();
    if dims.len() != 3 {
        return Err(LoftrError::InvalidInput(format!(
            "LocalFeatureTransformer `{label}` expects [N,L,C]; got {:?}",
            dims
        )));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::loftr_config::LoftrConfig;
    use tch::{Device, Kind};

    #[test]
    fn encoder_layer_preserves_shape() {
        let vs = nn::VarStore::new(Device::Cpu);
        let layer = LoFTREncoderLayer::new(&vs.root(), 256, 8, "linear").expect("encoder layer");
        let x = Tensor::randn([2, 12, 256], (Kind::Float, Device::Cpu));
        let source = Tensor::randn([2, 15, 256], (Kind::Float, Device::Cpu));
        let out = layer.forward(&x, &source, None, None).expect("forward");
        assert_eq!(out.size(), vec![2, 12, 256]);
    }

    #[test]
    fn transformer_preserves_feature_shapes() {
        let vs = nn::VarStore::new(Device::Cpu);
        let config = LoftrConfig::outdoor().coarse;
        let transformer = LocalFeatureTransformer::new(&vs.root(), &config).expect("transformer");
        let feat0 = Tensor::randn([1, 24, 256], (Kind::Float, Device::Cpu));
        let feat1 = Tensor::randn([1, 30, 256], (Kind::Float, Device::Cpu));
        let (out0, out1) = transformer
            .forward(&feat0, &feat1, None, None)
            .expect("forward");
        assert_eq!(out0.size(), vec![1, 24, 256]);
        assert_eq!(out1.size(), vec![1, 30, 256]);
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
        let transformer =
            LocalFeatureTransformer::new(&vs.root(), &transformer_config).expect("transformer");
        let feat0 = Tensor::randn([1, 12, 64], (Kind::Float, Device::Cpu));
        let feat1 = Tensor::randn([1, 12, 128], (Kind::Float, Device::Cpu));
        let err = transformer
            .forward(&feat0, &feat1, None, None)
            .expect_err("wrong dim");
        assert!(format!("{err}").contains("expected feature dim"));
    }
}
