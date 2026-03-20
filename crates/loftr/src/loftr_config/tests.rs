use super::*;

#[test]
fn outdoor_config_matches_kornia_defaults() {
    let config = LoftrConfig::outdoor();
    assert_eq!(config.backbone_type, BackboneType::ResNetFpn);
    assert_eq!(config.resolution, (8, 2));
    assert_eq!(config.fine_window_size, 5);
    assert!(config.fine_concat_coarse_feat);
    assert_eq!(config.resnetfpn.block_dims, [128, 196, 256]);
    assert_eq!(config.coarse.layers.len(), 8);
    assert_eq!(config.coarse.attention, AttentionType::Linear);
    assert_eq!(config.match_coarse.match_type, MatchType::DualSoftmax);
    assert_eq!(
        config.fine.layers,
        vec![
            TransformerLayer::SelfAttention,
            TransformerLayer::CrossAttention
        ]
    );
    assert!(!config.coarse.temp_bug_fix);
}

#[test]
fn indoor_matches_outdoor_defaults() {
    let outdoor = LoftrConfig::outdoor();
    let indoor = LoftrConfig::indoor();
    assert_eq!(indoor, outdoor);
}

#[test]
fn indoor_new_only_flips_temp_bug_fix() {
    let indoor = LoftrConfig::indoor();
    let indoor_new = LoftrConfig::indoor_new();
    assert!(indoor_new.coarse.temp_bug_fix);
    assert_eq!(indoor_new.backbone_type, indoor.backbone_type);
    assert!((indoor_new.match_coarse.thr - indoor.match_coarse.thr).abs() < f64::EPSILON);
    assert_eq!(indoor_new.fine, indoor.fine);
}
