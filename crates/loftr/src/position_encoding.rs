use tch::{Device, Kind, Tensor};

use crate::{error::LoftrError, numeric::i64_to_f64};

#[derive(Debug)]
pub struct PositionEncodingSine {
    d_model: i64,
    temp_bug_fix: bool,
    pe: Tensor,
}

impl PositionEncodingSine {
    pub fn new(
        d_model: i64,
        max_shape: (i64, i64),
        temp_bug_fix: bool,
        device: Device,
    ) -> Result<Self, LoftrError> {
        let pe = create_position_encoding(d_model, max_shape, temp_bug_fix, device)?;
        Ok(Self {
            d_model,
            temp_bug_fix,
            pe,
        })
    }

    pub fn update_position_encoding_size(
        &mut self,
        max_shape: (i64, i64),
    ) -> Result<(), LoftrError> {
        self.pe =
            create_position_encoding(self.d_model, max_shape, self.temp_bug_fix, self.pe.device())?;
        Ok(())
    }

    pub fn forward(&mut self, x: &Tensor) -> Result<Tensor, LoftrError> {
        let dims = x.size();
        if dims.len() != 4 {
            return Err(LoftrError::InvalidInput(format!(
                "PositionEncodingSine expects [N,C,H,W]; got {dims:?}"
            )));
        }
        if dims[1] != self.d_model {
            return Err(LoftrError::InvalidInput(format!(
                "PositionEncodingSine d_model mismatch: expected {}, got {}",
                self.d_model, dims[1]
            )));
        }
        if dims[2] > self.pe.size()[2] || dims[3] > self.pe.size()[3] {
            let max_shape = (
                dims[2].max(self.pe.size()[2]),
                dims[3].max(self.pe.size()[3]),
            );
            self.update_position_encoding_size(max_shape)?;
        }
        Ok(x + self.pe.slice(2, 0, dims[2], 1).slice(3, 0, dims[3], 1))
    }
}

fn create_position_encoding(
    d_model: i64,
    max_shape: (i64, i64),
    temp_bug_fix: bool,
    device: Device,
) -> Result<Tensor, LoftrError> {
    if d_model <= 0 || d_model % 4 != 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "PositionEncodingSine requires d_model > 0 and divisible by 4; got {d_model}"
        )));
    }
    if max_shape.0 <= 0 || max_shape.1 <= 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "PositionEncodingSine requires positive max_shape; got {max_shape:?}"
        )));
    }

    let options = (Kind::Float, device);
    let pe = Tensor::zeros([d_model, max_shape.0, max_shape.1], options);
    let y_position = Tensor::ones([max_shape.0, max_shape.1], options)
        .cumsum(0, Kind::Float)
        .unsqueeze(0);
    let x_position = Tensor::ones([max_shape.0, max_shape.1], options)
        .cumsum(1, Kind::Float)
        .unsqueeze(0);
    let div_base = if temp_bug_fix {
        -(10000.0_f64.ln()) / i64_to_f64(d_model / 2, "position encoding half d_model")?
    } else {
        (-(10000.0_f64.ln()) / i64_to_f64(d_model, "position encoding d_model")? / 2.0).floor()
    };
    let div_term = (Tensor::arange_start_step(0, d_model / 2, 2, options) * div_base)
        .exp()
        .unsqueeze(1)
        .unsqueeze(2);

    pe.slice(0, 0, d_model, 4)
        .copy_(&(x_position.shallow_clone() * &div_term).sin());
    pe.slice(0, 1, d_model, 4)
        .copy_(&(x_position.shallow_clone() * &div_term).cos());
    pe.slice(0, 2, d_model, 4)
        .copy_(&(y_position.shallow_clone() * &div_term).sin());
    pe.slice(0, 3, d_model, 4)
        .copy_(&(y_position.shallow_clone() * &div_term).cos());
    Ok(pe.unsqueeze(0))
}

#[cfg(test)]
mod tests;
