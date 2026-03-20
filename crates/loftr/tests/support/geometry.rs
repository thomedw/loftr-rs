use std::error::Error;

use tch::{Kind, Tensor};

type Mat3 = [[f64; 3]; 3];
type ModelEstimator = fn(&[[f64; 2]], &[[f64; 2]]) -> Result<Mat3, Box<dyn Error>>;

#[derive(Clone, Copy)]
pub(crate) struct GeometryRansacConfig {
    pub(crate) max_iterations: usize,
    pub(crate) threshold: f64,
    pub(crate) min_inliers: usize,
    pub(crate) seed: u64,
}

pub(crate) struct RansacFit {
    pub(crate) model: Mat3,
    pub(crate) inlier_count: usize,
}

pub(crate) fn points_from_tensor(tensor: &Tensor) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    let values = Vec::<f64>::try_from(tensor.f_to_kind(Kind::Double)?.reshape([-1]))?;
    let mut points = Vec::with_capacity(values.len() / 2);
    for chunk in values.chunks_exact(2) {
        points.push([chunk[0], chunk[1]]);
    }
    Ok(points)
}

pub(crate) fn fit_homography_ransac(
    points0: &[[f64; 2]],
    points1: &[[f64; 2]],
    config: GeometryRansacConfig,
) -> Result<RansacFit, Box<dyn Error>> {
    fit_ransac(
        points0,
        points1,
        4,
        config,
        estimate_homography,
        homography_error,
    )
}

pub(crate) fn project_homography(model: &Mat3, point: [f64; 2]) -> [f64; 2] {
    let x = point[0];
    let y = point[1];
    let denominator = model[2][0] * x + model[2][1] * y + model[2][2];
    let projected_x = (model[0][0] * x + model[0][1] * y + model[0][2]) / denominator;
    let projected_y = (model[1][0] * x + model[1][1] * y + model[1][2]) / denominator;
    [projected_x, projected_y]
}

pub(crate) fn mat3_from_tensor(tensor: &Tensor) -> Mat3 {
    tensor_to_mat3(tensor)
}

pub(crate) fn sampson_distance(model: &Mat3, point0: [f64; 2], point1: [f64; 2]) -> f64 {
    let x1 = [point0[0], point0[1], 1.0];
    let x2 = [point1[0], point1[1], 1.0];
    let fx1 = mat3_mul_vec(model, x1);
    let f_t = mat3_transpose(model);
    let ftx2 = mat3_mul_vec(&f_t, x2);
    let error = dot3(x2, fx1);
    let denominator = fx1[0] * fx1[0] + fx1[1] * fx1[1] + ftx2[0] * ftx2[0] + ftx2[1] * ftx2[1];
    if denominator <= 1e-12 {
        error * error
    } else {
        (error * error) / denominator
    }
}

fn fit_ransac(
    points0: &[[f64; 2]],
    points1: &[[f64; 2]],
    sample_size: usize,
    config: GeometryRansacConfig,
    estimate_model: ModelEstimator,
    point_error: fn(&Mat3, [f64; 2], [f64; 2]) -> f64,
) -> Result<RansacFit, Box<dyn Error>> {
    if points0.len() != points1.len() || points0.len() < sample_size {
        return Err(format!(
            "RANSAC needs equal point counts and at least {sample_size} correspondences"
        )
        .into());
    }

    let mut rng_state = if config.seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        config.seed
    };
    let mut best_model: Option<Mat3> = None;
    let mut best_inlier_indexes = Vec::<usize>::new();
    let mut best_score = f64::INFINITY;

    for _ in 0..config.max_iterations {
        let sample_indexes = sample_unique_indexes(points0.len(), sample_size, &mut rng_state)?;
        let sample0 = sample_indexes
            .iter()
            .map(|&index| points0[index])
            .collect::<Vec<_>>();
        let sample1 = sample_indexes
            .iter()
            .map(|&index| points1[index])
            .collect::<Vec<_>>();

        let Ok(model) = estimate_model(&sample0, &sample1) else {
            continue;
        };

        let mut inlier_indexes = Vec::new();
        let mut score = 0.0;
        for (index, (&point0, &point1)) in points0.iter().zip(points1.iter()).enumerate() {
            let error = point_error(&model, point0, point1);
            if error <= config.threshold {
                inlier_indexes.push(index);
                score += error;
            }
        }

        if inlier_indexes.len() > best_inlier_indexes.len()
            || (inlier_indexes.len() == best_inlier_indexes.len() && score < best_score)
        {
            best_model = Some(model);
            best_inlier_indexes = inlier_indexes;
            best_score = score;
        }
    }

    if best_inlier_indexes.len() < config.min_inliers {
        return Err(format!(
            "RANSAC found only {} inliers, need at least {}",
            best_inlier_indexes.len(),
            config.min_inliers
        )
        .into());
    }

    let refined0 = best_inlier_indexes
        .iter()
        .map(|&index| points0[index])
        .collect::<Vec<_>>();
    let refined1 = best_inlier_indexes
        .iter()
        .map(|&index| points1[index])
        .collect::<Vec<_>>();
    let model = estimate_model(&refined0, &refined1).or_else(|_| {
        best_model.ok_or_else(|| -> Box<dyn Error> { "RANSAC did not produce a model".into() })
    })?;

    Ok(RansacFit {
        model,
        inlier_count: best_inlier_indexes.len(),
    })
}

fn estimate_homography(points0: &[[f64; 2]], points1: &[[f64; 2]]) -> Result<Mat3, Box<dyn Error>> {
    if points0.len() != points1.len() || points0.len() < 4 {
        return Err("homography estimation needs at least four correspondences".into());
    }

    let mut rows = Vec::with_capacity(points0.len() * 18);
    for (&[x, y], &[u, v]) in points0.iter().zip(points1.iter()) {
        rows.extend_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u]);
        rows.extend_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v]);
    }

    let a = Tensor::from_slice(&rows)
        .view([usize_to_i64(points0.len() * 2)?, 9])
        .to_kind(Kind::Double);
    let (_, _, v) = a.svd(true, true);
    let last_column = v.select(1, v.size()[1] - 1).view([3, 3]);
    let denominator = last_column.double_value(&[2, 2]);
    if denominator.abs() <= 1e-12 {
        return Err("homography normalization failed".into());
    }

    Ok(tensor_to_mat3(&(last_column / denominator)))
}

fn homography_error(model: &Mat3, point0: [f64; 2], point1: [f64; 2]) -> f64 {
    let projected = project_homography(model, point0);
    let dx = projected[0] - point1[0];
    let dy = projected[1] - point1[1];
    (dx * dx + dy * dy).sqrt()
}

fn sample_unique_indexes(
    total: usize,
    sample_size: usize,
    rng_state: &mut u64,
) -> Result<Vec<usize>, Box<dyn Error>> {
    if total < sample_size {
        return Err("sample size cannot exceed total correspondences".into());
    }

    let total_u64 = u64::try_from(total)?;
    let mut indexes = Vec::with_capacity(sample_size);
    while indexes.len() < sample_size {
        let candidate = usize::try_from(next_random(rng_state) % total_u64)?;
        if !indexes.contains(&candidate) {
            indexes.push(candidate);
        }
    }
    Ok(indexes)
}

fn next_random(state: &mut u64) -> u64 {
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    *state = value;
    value
}

fn tensor_to_mat3(tensor: &Tensor) -> Mat3 {
    [
        [
            tensor.double_value(&[0, 0]),
            tensor.double_value(&[0, 1]),
            tensor.double_value(&[0, 2]),
        ],
        [
            tensor.double_value(&[1, 0]),
            tensor.double_value(&[1, 1]),
            tensor.double_value(&[1, 2]),
        ],
        [
            tensor.double_value(&[2, 0]),
            tensor.double_value(&[2, 1]),
            tensor.double_value(&[2, 2]),
        ],
    ]
}

fn mat3_transpose(matrix: &Mat3) -> Mat3 {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn mat3_mul_vec(matrix: &Mat3, vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn dot3(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn usize_to_i64(value: usize) -> Result<i64, Box<dyn Error>> {
    Ok(i64::try_from(value)?)
}
