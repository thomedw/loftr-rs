use super::*;

#[test]
fn model_variable_names_match_kornia_prefixes() -> Result<(), LoftrError> {
    let model = LoFTRModel::new(Device::Cpu, LoftrConfig::outdoor())?;
    let vars = model.var_store().variables();
    for name in [
        "backbone.conv1.weight",
        "backbone.layer1.0.conv1.weight",
        "fine_preprocess.down_proj.weight",
        "loftr_coarse.layers.0.q_proj.weight",
        "loftr_fine.layers.0.q_proj.weight",
    ] {
        assert!(vars.contains_key(name), "missing variable `{name}`");
    }
    Ok(())
}

#[test]
fn model_forward_smoke_returns_consistent_output_shapes() -> Result<(), LoftrError> {
    let mut model = LoFTRModel::new(Device::Cpu, LoftrConfig::outdoor())?;
    let image0 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
    let image1 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
    let out = model.forward(&image0, &image1)?;
    assert_eq!(out.keypoints0.size().len(), 2);
    assert_eq!(out.keypoints1.size().len(), 2);
    assert_eq!(out.confidence.size().len(), 1);
    assert_eq!(out.batch_indexes.size().len(), 1);
    assert_eq!(out.keypoints0.size()[0], out.keypoints1.size()[0]);
    assert_eq!(out.keypoints0.size()[0], out.confidence.size()[0]);
    assert_eq!(out.keypoints0.size()[0], out.batch_indexes.size()[0]);
    assert_eq!(out.keypoints0.size()[1], 2);
    assert_eq!(out.keypoints1.size()[1], 2);
    Ok(())
}
