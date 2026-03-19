use tch::{
    Device, Kind, Tensor,
    nn::{self},
};

use crate::{error::LoftrError, loftr_config::LoftrConfig};

#[derive(Debug)]
pub struct FinePreprocessData {
    pub hw0_f: (i64, i64),
    pub hw0_c: (i64, i64),
    pub b_ids: Tensor,
    pub i_ids: Tensor,
    pub j_ids: Tensor,
}

impl FinePreprocessData {
    pub fn stride(&self) -> Result<i64, LoftrError> {
        if self.hw0_f.0 <= 0 || self.hw0_f.1 <= 0 || self.hw0_c.0 <= 0 || self.hw0_c.1 <= 0 {
            return Err(LoftrError::InvalidConfig(format!(
                "FinePreprocessData requires positive fine/coarse shapes; got hw0_f={:?}, hw0_c={:?}",
                self.hw0_f, self.hw0_c
            )));
        }
        let stride_h = self.hw0_f.0 / self.hw0_c.0;
        if stride_h <= 0 {
            return Err(LoftrError::InvalidConfig(format!(
                "FinePreprocessData requires positive coarse stride; got hw0_f={:?}, hw0_c={:?}",
                self.hw0_f, self.hw0_c
            )));
        }
        Ok(stride_h)
    }

    pub fn match_count(&self) -> Result<i64, LoftrError> {
        let batch_count = first_dim(&self.b_ids, "b_ids")?;
        let i_count = first_dim(&self.i_ids, "i_ids")?;
        let j_count = first_dim(&self.j_ids, "j_ids")?;
        if batch_count != i_count || batch_count != j_count {
            return Err(LoftrError::InvalidConfig(format!(
                "FinePreprocessData index lengths must match; got b_ids={batch_count}, i_ids={i_count}, j_ids={j_count}"
            )));
        }
        Ok(batch_count)
    }
}

#[derive(Debug)]
pub struct FinePreprocess {
    cat_coarse_feat: bool,
    window_size: i64,
    d_model_f: i64,
    down_proj: Option<nn::Linear>,
    merge_feat: Option<nn::Linear>,
}

impl FinePreprocess {
    pub fn new(vs: &nn::Path<'_>, config: &LoftrConfig) -> Result<Self, LoftrError> {
        if config.fine_window_size <= 0 || config.fine_window_size % 2 == 0 {
            return Err(LoftrError::InvalidConfig(format!(
                "FinePreprocess requires a positive odd fine_window_size; got {}",
                config.fine_window_size
            )));
        }

        let d_model_c = config.coarse.d_model;
        let d_model_f = config.fine.d_model;
        let linear_config = nn::LinearConfig {
            ws_init: nn::init::DEFAULT_KAIMING_NORMAL,
            ..Default::default()
        };

        let (down_proj, merge_feat) = if config.fine_concat_coarse_feat {
            let down_proj = nn::linear(vs / "down_proj", d_model_c, d_model_f, linear_config);
            let merge_feat = nn::linear(vs / "merge_feat", 2 * d_model_f, d_model_f, linear_config);
            (Some(down_proj), Some(merge_feat))
        } else {
            (None, None)
        };

        Ok(Self {
            cat_coarse_feat: config.fine_concat_coarse_feat,
            window_size: config.fine_window_size,
            d_model_f,
            down_proj,
            merge_feat,
        })
    }

    pub fn forward(
        &self,
        fine_map0: &Tensor,
        fine_map1: &Tensor,
        coarse_tokens0: &Tensor,
        coarse_tokens1: &Tensor,
        data: &FinePreprocessData,
    ) -> Result<(Tensor, Tensor), LoftrError> {
        validate_fine_map(fine_map0, "fine_map0", self.d_model_f)?;
        validate_fine_map(fine_map1, "fine_map1", self.d_model_f)?;
        validate_coarse_sequence(coarse_tokens0, "coarse_tokens0")?;
        validate_coarse_sequence(coarse_tokens1, "coarse_tokens1")?;

        let match_count = data.match_count()?;
        let stride = data.stride()?;
        if match_count == 0 {
            let empty = Tensor::empty(
                [0, self.window_size * self.window_size, self.d_model_f],
                (Kind::Float, fine_map0.device()),
            );
            return Ok((empty.shallow_clone(), empty));
        }

        let fine_windows0 = unfold_local_windows(fine_map0, self.window_size, stride)?;
        let fine_windows1 = unfold_local_windows(fine_map1, self.window_size, stride)?;

        let fine_windows0 = select_unfold_windows(&fine_windows0, &data.b_ids, &data.i_ids)?;
        let fine_windows1 = select_unfold_windows(&fine_windows1, &data.b_ids, &data.j_ids)?;

        if !self.cat_coarse_feat {
            return Ok((fine_windows0, fine_windows1));
        }

        let down_proj = self.down_proj.as_ref().ok_or_else(|| {
            LoftrError::InvalidConfig(String::from(
                "FinePreprocess missing down_proj while fine_concat_coarse_feat is enabled",
            ))
        })?;
        let merge_feat = self.merge_feat.as_ref().ok_or_else(|| {
            LoftrError::InvalidConfig(String::from(
                "FinePreprocess missing merge_feat while fine_concat_coarse_feat is enabled",
            ))
        })?;

        let coarse_context0 = select_sequence_tokens(coarse_tokens0, &data.b_ids, &data.i_ids)?;
        let coarse_context1 = select_sequence_tokens(coarse_tokens1, &data.b_ids, &data.j_ids)?;
        let coarse_context = Tensor::cat(&[coarse_context0, coarse_context1], 0).apply(down_proj);
        let fine_windows = Tensor::cat(
            &[fine_windows0.shallow_clone(), fine_windows1.shallow_clone()],
            0,
        );
        let coarse_context =
            coarse_context
                .unsqueeze(1)
                .repeat([1, self.window_size * self.window_size, 1]);
        let merged = Tensor::cat(&[fine_windows, coarse_context], -1).apply(merge_feat);
        let chunks = merged.chunk(2, 0);
        Ok((chunks[0].shallow_clone(), chunks[1].shallow_clone()))
    }
}

fn first_dim(tensor: &Tensor, label: &str) -> Result<i64, LoftrError> {
    let dims = tensor.size();
    if dims.len() != 1 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocessData `{label}` must be rank-1; got {dims:?}"
        )));
    }
    Ok(dims[0])
}

fn validate_fine_map(
    tensor: &Tensor,
    label: &str,
    expected_channels: i64,
) -> Result<(), LoftrError> {
    let dims = tensor.size();
    if dims.len() != 4 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess `{label}` expects [N,C,H,W]; got {dims:?}"
        )));
    }
    if dims[1] != expected_channels {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess `{label}` expected {} channels; got {}",
            expected_channels, dims[1]
        )));
    }
    Ok(())
}

fn validate_coarse_sequence(tensor: &Tensor, label: &str) -> Result<(), LoftrError> {
    let dims = tensor.size();
    if dims.len() != 3 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess `{label}` expects [N,L,C]; got {dims:?}"
        )));
    }
    Ok(())
}

fn unfold_local_windows(
    feat: &Tensor,
    window_size: i64,
    stride: i64,
) -> Result<Tensor, LoftrError> {
    let dims = feat.size();
    let unfolded = feat.im2col(
        [window_size, window_size],
        [1, 1],
        [window_size / 2, window_size / 2],
        [stride, stride],
    );
    let unfolded_dims = unfolded.size();
    if unfolded_dims.len() != 3 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess im2col expected [N,C*W*W,L]; got {unfolded_dims:?}"
        )));
    }
    let window_area = window_size * window_size;
    if unfolded_dims[1] % window_area != 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess im2col channel area mismatch: {unfolded_dims:?} with window_area={window_area}"
        )));
    }
    let channels = unfolded_dims[1] / window_area;
    if channels != dims[1] {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess im2col changed channel count unexpectedly: expected {}, got {}",
            dims[1], channels
        )));
    }
    Ok(unfolded
        .reshape([dims[0], channels, window_area, unfolded_dims[2]])
        .permute([0, 3, 2, 1]))
}

fn select_unfold_windows(
    windows: &Tensor,
    b_ids: &Tensor,
    token_ids: &Tensor,
) -> Result<Tensor, LoftrError> {
    let dims = windows.size();
    if dims.len() != 4 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess windows expect [N,L,WW,C]; got {dims:?}"
        )));
    }
    let batch_offsets = normalize_index_tensor(b_ids, windows.device())? * dims[1];
    let token_ids = normalize_index_tensor(token_ids, windows.device())?;
    let linear_ids = batch_offsets + token_ids;
    Ok(windows
        .reshape([dims[0] * dims[1], dims[2], dims[3]])
        .index_select(0, &linear_ids))
}

fn select_sequence_tokens(
    sequence: &Tensor,
    b_ids: &Tensor,
    token_ids: &Tensor,
) -> Result<Tensor, LoftrError> {
    let dims = sequence.size();
    if dims.len() != 3 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess coarse sequence expects [N,L,C]; got {dims:?}"
        )));
    }
    let batch_offsets = normalize_index_tensor(b_ids, sequence.device())? * dims[1];
    let token_ids = normalize_index_tensor(token_ids, sequence.device())?;
    let linear_ids = batch_offsets + token_ids;
    Ok(sequence
        .reshape([dims[0] * dims[1], dims[2]])
        .index_select(0, &linear_ids))
}

fn normalize_index_tensor(indexes: &Tensor, device: Device) -> Result<Tensor, LoftrError> {
    let dims = indexes.size();
    if dims.len() != 1 {
        return Err(LoftrError::InvalidConfig(format!(
            "FinePreprocess indexes must be rank-1; got {dims:?}"
        )));
    }
    Ok(indexes.f_to_device(device)?.f_to_kind(Kind::Int64)?)
}

#[cfg(test)]
mod tests;
