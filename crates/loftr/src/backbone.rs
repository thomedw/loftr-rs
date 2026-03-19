use tch::{
    Tensor,
    nn::{self},
};

use crate::{
    error::LoftrError,
    loftr_config::{LoftrConfig, ResNetFpnConfig},
};

fn conv1x1(vs: &nn::Path<'_>, in_planes: i64, out_planes: i64, stride: i64) -> nn::Conv2D {
    nn::conv2d(
        vs,
        in_planes,
        out_planes,
        1,
        nn::ConvConfig {
            stride,
            padding: 0,
            bias: false,
            ..Default::default()
        },
    )
}

fn conv3x3(vs: &nn::Path<'_>, in_planes: i64, out_planes: i64, stride: i64) -> nn::Conv2D {
    nn::conv2d(
        vs,
        in_planes,
        out_planes,
        3,
        nn::ConvConfig {
            stride,
            padding: 1,
            bias: false,
            ..Default::default()
        },
    )
}

#[derive(Debug)]
struct BasicBlock {
    conv1: nn::Conv2D,
    conv2: nn::Conv2D,
    bn1: nn::BatchNorm,
    bn2: nn::BatchNorm,
    downsample: Option<(nn::Conv2D, nn::BatchNorm)>,
}

impl BasicBlock {
    fn new(vs: &nn::Path<'_>, in_planes: i64, planes: i64, stride: i64) -> Self {
        let downsample = if stride == 1 {
            None
        } else {
            Some((
                conv1x1(&(vs / "downsample" / "0"), in_planes, planes, stride),
                nn::batch_norm2d(
                    &(vs / "downsample" / "1"),
                    planes,
                    nn::BatchNormConfig::default(),
                ),
            ))
        };
        Self {
            conv1: conv3x3(&(vs / "conv1"), in_planes, planes, stride),
            conv2: conv3x3(&(vs / "conv2"), planes, planes, 1),
            bn1: nn::batch_norm2d(&(vs / "bn1"), planes, nn::BatchNormConfig::default()),
            bn2: nn::batch_norm2d(&(vs / "bn2"), planes, nn::BatchNormConfig::default()),
            downsample,
        }
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let y = x
            .apply(&self.conv1)
            .apply_t(&self.bn1, train)
            .relu()
            .apply(&self.conv2)
            .apply_t(&self.bn2, train);
        let residual = match &self.downsample {
            Some((conv, bn)) => x.apply(conv).apply_t(bn, train),
            None => x.shallow_clone(),
        };
        (residual + y).relu()
    }
}

#[derive(Debug)]
struct ResNetLayer {
    block1: BasicBlock,
    block2: BasicBlock,
}

impl ResNetLayer {
    fn new(vs: &nn::Path<'_>, in_planes: i64, dim: i64, stride: i64) -> (Self, i64) {
        let block1 = BasicBlock::new(&(vs / "0"), in_planes, dim, stride);
        let block2 = BasicBlock::new(&(vs / "1"), dim, dim, 1);
        (Self { block1, block2 }, dim)
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        let x = self.block1.forward_t(x, train);
        self.block2.forward_t(&x, train)
    }
}

#[derive(Debug)]
struct FpnHead {
    conv1: nn::Conv2D,
    bn: nn::BatchNorm,
    conv2: nn::Conv2D,
}

impl FpnHead {
    fn new(vs: &nn::Path<'_>, in_planes: i64, hidden_planes: i64, out_planes: i64) -> Self {
        Self {
            conv1: conv3x3(&(vs / "0"), in_planes, hidden_planes, 1),
            bn: nn::batch_norm2d(&(vs / "1"), hidden_planes, nn::BatchNormConfig::default()),
            conv2: conv3x3(&(vs / "3"), hidden_planes, out_planes, 1),
        }
    }

    fn forward_t(&self, x: &Tensor, train: bool) -> Tensor {
        x.apply(&self.conv1)
            .apply_t(&self.bn, train)
            .leaky_relu()
            .apply(&self.conv2)
    }
}

#[derive(Debug)]
pub struct ResNetFpn8_2 {
    conv1: nn::Conv2D,
    bn1: nn::BatchNorm,
    layer1: ResNetLayer,
    layer2: ResNetLayer,
    layer3: ResNetLayer,
    layer3_outconv: nn::Conv2D,
    layer2_outconv: nn::Conv2D,
    layer2_outconv2: FpnHead,
    layer1_outconv: nn::Conv2D,
    layer1_outconv2: FpnHead,
}

impl ResNetFpn8_2 {
    pub fn new(vs: &nn::Path<'_>, config: &ResNetFpnConfig) -> Result<Self, LoftrError> {
        let initial_dim = config.initial_dim;
        let block_dims = config.block_dims;
        if initial_dim <= 0 || block_dims.iter().any(|dim| *dim <= 0) {
            return Err(LoftrError::InvalidConfig(format!(
                "ResNetFpn8_2 requires positive dimensions; got initial_dim={initial_dim}, block_dims={block_dims:?}"
            )));
        }

        let conv1 = nn::conv2d(
            &(vs / "conv1"),
            1,
            initial_dim,
            7,
            nn::ConvConfig {
                stride: 2,
                padding: 3,
                bias: false,
                ..Default::default()
            },
        );
        let bn1 = nn::batch_norm2d(&(vs / "bn1"), initial_dim, nn::BatchNormConfig::default());
        let (layer1, next_planes) =
            ResNetLayer::new(&(vs / "layer1"), initial_dim, block_dims[0], 1);
        let (layer2, next_planes) =
            ResNetLayer::new(&(vs / "layer2"), next_planes, block_dims[1], 2);
        let (layer3, _) = ResNetLayer::new(&(vs / "layer3"), next_planes, block_dims[2], 2);

        Ok(Self {
            conv1,
            bn1,
            layer1,
            layer2,
            layer3,
            layer3_outconv: conv1x1(&(vs / "layer3_outconv"), block_dims[2], block_dims[2], 1),
            layer2_outconv: conv1x1(&(vs / "layer2_outconv"), block_dims[1], block_dims[2], 1),
            layer2_outconv2: FpnHead::new(
                &(vs / "layer2_outconv2"),
                block_dims[2],
                block_dims[2],
                block_dims[1],
            ),
            layer1_outconv: conv1x1(&(vs / "layer1_outconv"), block_dims[0], block_dims[1], 1),
            layer1_outconv2: FpnHead::new(
                &(vs / "layer1_outconv2"),
                block_dims[1],
                block_dims[1],
                block_dims[0],
            ),
        })
    }

    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<(Tensor, Tensor), LoftrError> {
        let dims = x.size();
        if dims.len() != 4 || dims[1] != 1 {
            return Err(LoftrError::InvalidInput(format!(
                "ResNetFpn8_2 expects grayscale [N,1,H,W] input; got {dims:?}"
            )));
        }

        let x0 = x.apply(&self.conv1).apply_t(&self.bn1, train).relu();
        let x1 = self.layer1.forward_t(&x0, train);
        let x2 = self.layer2.forward_t(&x1, train);
        let x3 = self.layer3.forward_t(&x2, train);

        let x3_out = x3.apply(&self.layer3_outconv);
        let x2_out = x2.apply(&self.layer2_outconv);
        let x3_out_2x =
            x3_out.upsample_bilinear2d([x2_out.size()[2], x2_out.size()[3]], true, None, None);
        let x2_out = self.layer2_outconv2.forward_t(&(x2_out + x3_out_2x), train);

        let x1_out = x1.apply(&self.layer1_outconv);
        let x2_out_2x =
            x2_out.upsample_bilinear2d([x1_out.size()[2], x1_out.size()[3]], true, None, None);
        let x1_out = self.layer1_outconv2.forward_t(&(x1_out + x2_out_2x), train);

        Ok((x3_out, x1_out))
    }
}

#[derive(Debug)]
pub enum Backbone {
    ResNetFpn8_2(ResNetFpn8_2),
}

impl Backbone {
    pub fn forward_t(&self, x: &Tensor, train: bool) -> Result<(Tensor, Tensor), LoftrError> {
        match self {
            Self::ResNetFpn8_2(backbone) => backbone.forward_t(x, train),
        }
    }
}

pub fn build_backbone(vs: &nn::Path<'_>, config: &LoftrConfig) -> Result<Backbone, LoftrError> {
    if config.backbone_type != "ResNetFPN" {
        return Err(LoftrError::InvalidConfig(format!(
            "Unsupported LoFTR backbone type `{}`",
            config.backbone_type
        )));
    }
    match config.resolution {
        (8, 2) => Ok(Backbone::ResNetFpn8_2(ResNetFpn8_2::new(
            &(vs / "backbone"),
            &config.resnetfpn,
        )?)),
        other => Err(LoftrError::InvalidConfig(format!(
            "Unsupported LoFTR backbone resolution {other:?}; only (8, 2) is ported so far"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Device, Kind};

    #[test]
    fn resnet_fpn_8_2_matches_expected_output_shapes() -> Result<(), LoftrError> {
        let vs = nn::VarStore::new(Device::Cpu);
        let backbone = ResNetFpn8_2::new(&vs.root(), &LoftrConfig::outdoor().resnetfpn)?;
        let input = Tensor::randn([1, 1, 240, 320], (Kind::Float, Device::Cpu));
        let (coarse, fine) = backbone.forward_t(&input, false)?;
        assert_eq!(coarse.size(), vec![1, 256, 30, 40]);
        assert_eq!(fine.size(), vec![1, 128, 120, 160]);
        Ok(())
    }

    #[test]
    fn build_backbone_supports_outdoor_config() -> Result<(), LoftrError> {
        let vs = nn::VarStore::new(Device::Cpu);
        let backbone = build_backbone(&vs.root(), &LoftrConfig::outdoor())?;
        let input = Tensor::randn([1, 1, 120, 160], (Kind::Float, Device::Cpu));
        let (coarse, fine) = backbone.forward_t(&input, false)?;
        assert_eq!(coarse.size(), vec![1, 256, 15, 20]);
        assert_eq!(fine.size(), vec![1, 128, 60, 80]);
        Ok(())
    }

    #[test]
    fn build_backbone_rejects_unsupported_resolution() {
        let vs = nn::VarStore::new(Device::Cpu);
        let mut config = LoftrConfig::outdoor();
        config.resolution = (16, 4);
        match build_backbone(&vs.root(), &config) {
            Ok(_) => panic!("unsupported resolution should fail"),
            Err(err) => assert!(format!("{err}").contains("only (8, 2) is ported")),
        }
    }
}
