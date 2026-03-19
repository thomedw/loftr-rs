use num_traits::ToPrimitive;

use crate::error::LoftrError;

pub fn i64_to_f64(value: i64, label: &str) -> Result<f64, LoftrError> {
    value.to_f64().ok_or_else(|| {
        LoftrError::InvalidConfig(format!("{label}={value} cannot be represented as f64"))
    })
}

pub fn i64_pair_ratio_to_f64(
    numerator: i64,
    denominator: i64,
    label: &str,
) -> Result<f64, LoftrError> {
    if denominator == 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "{label} denominator must be non-zero"
        )));
    }
    Ok(i64_to_f64(numerator, label)? / i64_to_f64(denominator, label)?)
}

pub fn perfect_square_root(value: i64, label: &str) -> Result<i64, LoftrError> {
    if value <= 0 {
        return Err(LoftrError::InvalidConfig(format!(
            "{label} must be positive; got {value}"
        )));
    }

    let mut candidate = 1_i64;
    while candidate
        .checked_mul(candidate)
        .is_some_and(|square| square < value)
    {
        candidate += 1;
    }

    if candidate
        .checked_mul(candidate)
        .is_some_and(|square| square == value)
    {
        Ok(candidate)
    } else {
        Err(LoftrError::InvalidConfig(format!(
            "{label} must be a perfect square; got {value}"
        )))
    }
}
