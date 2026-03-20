#[path = "support/geometry.rs"]
mod geometry;
mod support;

use std::error::Error;

use geometry::{
    GeometryRansacConfig, fit_homography_ransac, mat3_from_tensor, points_from_tensor,
    project_homography, sampson_distance,
};
use loftr::LoftrConfig;
use support::{ReferenceFixture, load_model};

#[test]
fn outdoor_matches_support_homography_estimation() -> Result<(), Box<dyn Error>> {
    let fixture = ReferenceFixture::load("loftr_outdoor_reference")?;
    let mut model = load_model(
        "loftr_outdoor_state_dict.safetensors",
        LoftrConfig::outdoor(),
    )?;

    let image0 = fixture.tensor("image0")?;
    let image1 = fixture.tensor("image1")?;
    let matches = model.forward(&image0, &image1)?;
    let points0 = points_from_tensor(&matches.keypoints0)?;
    let points1 = points_from_tensor(&matches.keypoints1)?;
    let fit = fit_homography_ransac(
        &points0,
        &points1,
        GeometryRansacConfig {
            max_iterations: 400,
            threshold: 3.0,
            min_inliers: 30,
            seed: 1,
        },
    )?;
    assert!(fit.inlier_count >= 30);

    let clean0 = points_from_tensor(&fixture.tensor("pts0")?)?;
    let clean1 = points_from_tensor(&fixture.tensor("pts1")?)?;
    for (&point0, &point1) in clean0.iter().zip(clean1.iter()) {
        let projected = project_homography(&fit.model, point0);
        let tolerance_x = 5.0 + 0.15 * point1[0].abs();
        let tolerance_y = 5.0 + 0.15 * point1[1].abs();
        assert!((projected[0] - point1[0]).abs() <= tolerance_x);
        assert!((projected[1] - point1[1]).abs() <= tolerance_y);
    }
    Ok(())
}

#[test]
fn indoor_matches_support_reference_epipolar_geometry() -> Result<(), Box<dyn Error>> {
    let fixture = ReferenceFixture::load("loftr_indoor_reference")?;
    let mut model = load_model("loftr_indoor_state_dict.safetensors", LoftrConfig::indoor())?;

    let image0 = fixture.tensor("image0")?;
    let image1 = fixture.tensor("image1")?;
    let matches = model.forward(&image0, &image1)?;
    let points0 = points_from_tensor(&matches.keypoints0)?;
    let points1 = points_from_tensor(&matches.keypoints1)?;
    let fundamental = mat3_from_tensor(&fixture.tensor("F_gt")?);

    let inliers = points0
        .iter()
        .zip(points1.iter())
        .filter(|(point0, point1)| {
            sampson_distance(&fundamental, **point0, **point1).sqrt() <= 10.0
        })
        .count();
    assert!(inliers >= 480);
    Ok(())
}

#[test]
fn indoor_clean_correspondences_match_reference_fundamental() -> Result<(), Box<dyn Error>> {
    let fixture = ReferenceFixture::load("loftr_indoor_reference")?;
    let fundamental = mat3_from_tensor(&fixture.tensor("F_gt")?);
    let clean0 = points_from_tensor(&fixture.tensor("pts0")?)?;
    let clean1 = points_from_tensor(&fixture.tensor("pts1")?)?;

    let gross_errors = clean0
        .iter()
        .zip(clean1.iter())
        .filter(|(point0, point1)| sampson_distance(&fundamental, **point0, **point1).sqrt() > 1.0)
        .count();
    assert_eq!(gross_errors, 0);
    Ok(())
}
