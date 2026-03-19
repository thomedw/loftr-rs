use super::*;

#[test]
fn outdoor_config_matches_kornia_defaults() {
    let config = LoftrConfig::outdoor();
    assert_eq!(config.backbone_type, BackboneType::ResNetFpn);
    assert_eq!(config.resolution, (8, 2));
    assert_eq!(config.fine_window_size, 5);
    assert!(config.fine_concat_coarse_feat);
    assert_eq!(config.resnetfpn.block_dims, [128, 196, 256]);
    assert_eq!(config.coarse.layer_kinds.len(), 8);
    assert_eq!(config.coarse.attention, AttentionType::Linear);
    assert_eq!(config.match_coarse.match_type, MatchType::DualSoftmax);
    assert_eq!(
        config.fine.layer_kinds,
        vec![
            TransformerLayerKind::SelfAttention,
            TransformerLayerKind::CrossAttention,
        ]
    );
    assert!(!config.coarse.temp_bug_fix);
}

#[test]
fn indoor_new_only_flips_temp_bug_fix() {
    let outdoor = LoftrConfig::outdoor();
    let indoor = LoftrConfig::indoor_new();
    assert!(indoor.coarse.temp_bug_fix);
    assert_eq!(indoor.backbone_type, outdoor.backbone_type);
    assert!((indoor.match_coarse.thr - outdoor.match_coarse.thr).abs() < f64::EPSILON);
    assert_eq!(indoor.fine, outdoor.fine);
}
