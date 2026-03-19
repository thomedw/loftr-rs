use super::*;
use tch::Device;

#[test]
fn linear_attention_preserves_shape() -> Result<(), LoftrError> {
    let queries = Tensor::randn([2, 5, 4, 8], (Kind::Float, Device::Cpu));
    let keys = Tensor::randn([2, 7, 4, 8], (Kind::Float, Device::Cpu));
    let values = Tensor::randn([2, 7, 4, 8], (Kind::Float, Device::Cpu));
    let out = LinearAttention::default().forward(&queries, &keys, &values, None, None)?;
    assert_eq!(out.size(), vec![2, 5, 4, 8]);
    Ok(())
}

#[test]
fn linear_attention_zeroes_masked_queries() -> Result<(), LoftrError> {
    let queries = Tensor::ones([1, 2, 1, 2], (Kind::Float, Device::Cpu));
    let keys = Tensor::ones([1, 2, 1, 2], (Kind::Float, Device::Cpu));
    let values = Tensor::ones([1, 2, 1, 2], (Kind::Float, Device::Cpu));
    let q_mask = Tensor::from_slice(&[1_i64, 0]).view([1, 2]);
    let out = LinearAttention::default().forward(&queries, &keys, &values, Some(&q_mask), None)?;
    assert!(out.get(0).get(1).abs().sum(Kind::Float).double_value(&[]) < 1e-9);
    Ok(())
}

#[test]
fn full_attention_preserves_shape() -> Result<(), LoftrError> {
    let queries = Tensor::randn([1, 3, 2, 4], (Kind::Float, Device::Cpu));
    let keys = Tensor::randn([1, 6, 2, 4], (Kind::Float, Device::Cpu));
    let values = Tensor::randn([1, 6, 2, 4], (Kind::Float, Device::Cpu));
    let out = FullAttention::default().forward(&queries, &keys, &values, None, None)?;
    assert_eq!(out.size(), vec![1, 3, 2, 4]);
    Ok(())
}
