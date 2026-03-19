use tch::{Kind, Tensor};

use crate::{error::LoftrError, numeric::i64_to_f64};

#[derive(Debug, Clone, Copy)]
pub struct LinearAttention {
    eps: f64,
}

impl Default for LinearAttention {
    fn default() -> Self {
        Self { eps: 1e-6 }
    }
}

impl LinearAttention {
    pub fn forward(
        self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        q_mask: Option<&Tensor>,
        kv_mask: Option<&Tensor>,
    ) -> Result<Tensor, LoftrError> {
        validate_attention_tensors(queries, keys, values)?;

        let mut q = queries.elu() + 1.0;
        let mut k = keys.elu() + 1.0;
        let mut v = values.shallow_clone();

        if let Some(mask) = q_mask {
            let mask = expand_attention_mask(mask, queries)?;
            q *= mask;
        }
        if let Some(mask) = kv_mask {
            let mask = expand_attention_mask(mask, keys)?;
            k *= &mask;
            v *= mask;
        }

        let value_length = i64_to_f64(values.size()[1], "attention value length")?;
        let v = &v / value_length;
        let kv = Tensor::einsum("nshd,nshv->nhdv", &[&k, &v], None::<&[i64]>);
        let z = (Tensor::einsum(
            "nlhd,nhd->nlh",
            &[&q, &k.sum_dim_intlist([1].as_slice(), false, Kind::Float)],
            None::<&[i64]>,
        ) + self.eps)
            .reciprocal();
        Ok(
            (Tensor::einsum("nlhd,nhdv,nlh->nlhv", &[&q, &kv, &z], None::<&[i64]>) * value_length)
                .contiguous(),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FullAttention {
    use_dropout: bool,
    attention_dropout: f64,
}

impl Default for FullAttention {
    fn default() -> Self {
        Self {
            use_dropout: false,
            attention_dropout: 0.1,
        }
    }
}

impl FullAttention {
    pub fn forward(
        self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        q_mask: Option<&Tensor>,
        kv_mask: Option<&Tensor>,
    ) -> Result<Tensor, LoftrError> {
        validate_attention_tensors(queries, keys, values)?;

        let mut qk = Tensor::einsum("nlhd,nshd->nlsh", &[queries, keys], None::<&[i64]>);
        if let (Some(q_mask), Some(kv_mask)) = (q_mask, kv_mask) {
            let q_mask = expand_full_mask(q_mask, queries, true)?;
            let kv_mask = expand_full_mask(kv_mask, keys, false)?;
            let valid = q_mask.logical_and(&kv_mask);
            qk = qk.f_masked_fill(&valid.logical_not(), f64::NEG_INFINITY)?;
        }

        let softmax_temp = 1.0 / i64_to_f64(queries.size()[3], "attention head dim")?.sqrt();
        let mut attention = (qk * softmax_temp).softmax(2, Kind::Float);
        if self.use_dropout && self.attention_dropout > 0.0 {
            attention = attention.dropout(self.attention_dropout, false);
        }
        Ok(Tensor::einsum("nlsh,nshd->nlhd", &[&attention, values], None::<&[i64]>).contiguous())
    }
}

fn validate_attention_tensors(
    queries: &Tensor,
    keys: &Tensor,
    values: &Tensor,
) -> Result<(), LoftrError> {
    let q_dims = queries.size();
    let k_dims = keys.size();
    let v_dims = values.size();
    if q_dims.len() != 4 || k_dims.len() != 4 || v_dims.len() != 4 {
        return Err(LoftrError::InvalidConfig(format!(
            "Attention expects [N,L,H,D], [N,S,H,D], [N,S,H,D]; got queries={q_dims:?}, keys={k_dims:?}, values={v_dims:?}"
        )));
    }
    if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
        return Err(LoftrError::InvalidConfig(format!(
            "Attention batch mismatch: queries={}, keys={}, values={}",
            q_dims[0], k_dims[0], v_dims[0]
        )));
    }
    if k_dims[1] != v_dims[1] || k_dims[2] != v_dims[2] || k_dims[3] != v_dims[3] {
        return Err(LoftrError::InvalidConfig(format!(
            "Attention key/value mismatch: keys={k_dims:?}, values={v_dims:?}"
        )));
    }
    if q_dims[2] != k_dims[2] || q_dims[3] != k_dims[3] {
        return Err(LoftrError::InvalidConfig(format!(
            "Attention query/key head mismatch: queries={q_dims:?}, keys={k_dims:?}"
        )));
    }
    Ok(())
}

fn expand_attention_mask(mask: &Tensor, like: &Tensor) -> Result<Tensor, LoftrError> {
    let dims = mask.size();
    let expected = [like.size()[0], like.size()[1]];
    if dims != expected {
        return Err(LoftrError::InvalidConfig(format!(
            "Attention mask expects {expected:?}; got {dims:?}"
        )));
    }
    Ok(mask
        .f_to_device(like.device())?
        .f_to_kind(like.kind())?
        .unsqueeze(-1)
        .unsqueeze(-1))
}

fn expand_full_mask(mask: &Tensor, like: &Tensor, is_query: bool) -> Result<Tensor, LoftrError> {
    let dims = mask.size();
    let expected = [like.size()[0], like.size()[1]];
    if dims != expected {
        return Err(LoftrError::InvalidConfig(format!(
            "Attention mask expects {expected:?}; got {dims:?}"
        )));
    }
    let mask = mask.f_to_device(like.device())?.f_to_kind(Kind::Bool)?;
    Ok(if is_query {
        mask.unsqueeze(-1).unsqueeze(-1)
    } else {
        mask.unsqueeze(1).unsqueeze(-1)
    })
}

#[cfg(test)]
mod tests {
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
        let out =
            LinearAttention::default().forward(&queries, &keys, &values, Some(&q_mask), None)?;
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
}
