#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResNetFpnConfig {
    pub initial_dim: i64,
    pub block_dims: [i64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformerConfig {
    pub d_model: i64,
    pub d_ffn: i64,
    pub nhead: i64,
    pub layer_names: Vec<String>,
    pub attention: String,
    pub temp_bug_fix: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MatchCoarseConfig {
    pub thr: f64,
    pub border_rm: i64,
    pub match_type: String,
    pub dsmax_temperature: f64,
    pub skh_iters: i64,
    pub skh_init_bin_score: f64,
    pub skh_prefilter: bool,
    pub train_coarse_percent: f64,
    pub train_pad_num_gt_min: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FineConfig {
    pub d_model: i64,
    pub d_ffn: i64,
    pub nhead: i64,
    pub layer_names: Vec<String>,
    pub attention: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoftrConfig {
    pub backbone_type: String,
    pub resolution: (i64, i64),
    pub fine_window_size: i64,
    pub fine_concat_coarse_feat: bool,
    pub resnetfpn: ResNetFpnConfig,
    pub coarse: TransformerConfig,
    pub match_coarse: MatchCoarseConfig,
    pub fine: FineConfig,
}

impl LoftrConfig {
    pub fn outdoor() -> Self {
        Self {
            backbone_type: String::from("ResNetFPN"),
            resolution: (8, 2),
            fine_window_size: 5,
            fine_concat_coarse_feat: true,
            resnetfpn: ResNetFpnConfig {
                initial_dim: 128,
                block_dims: [128, 196, 256],
            },
            coarse: TransformerConfig {
                d_model: 256,
                d_ffn: 256,
                nhead: 8,
                layer_names: vec![
                    String::from("self"),
                    String::from("cross"),
                    String::from("self"),
                    String::from("cross"),
                    String::from("self"),
                    String::from("cross"),
                    String::from("self"),
                    String::from("cross"),
                ],
                attention: String::from("linear"),
                temp_bug_fix: false,
            },
            match_coarse: MatchCoarseConfig {
                thr: 0.2,
                border_rm: 2,
                match_type: String::from("dual_softmax"),
                dsmax_temperature: 0.1,
                skh_iters: 3,
                skh_init_bin_score: 1.0,
                skh_prefilter: true,
                train_coarse_percent: 0.4,
                train_pad_num_gt_min: 200,
            },
            fine: FineConfig {
                d_model: 128,
                d_ffn: 128,
                nhead: 8,
                layer_names: vec![String::from("self"), String::from("cross")],
                attention: String::from("linear"),
            },
        }
    }

    pub fn indoor_new() -> Self {
        let mut config = Self::outdoor();
        config.coarse.temp_bug_fix = true;
        config
    }
}

impl Default for LoftrConfig {
    fn default() -> Self {
        Self::outdoor()
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn outdoor_config_matches_kornia_defaults() {
        let config = LoftrConfig::outdoor();
        assert_eq!(config.backbone_type, "ResNetFPN");
        assert_eq!(config.resolution, (8, 2));
        assert_eq!(config.fine_window_size, 5);
        assert!(config.fine_concat_coarse_feat);
        assert_eq!(config.resnetfpn.block_dims, [128, 196, 256]);
        assert_eq!(config.coarse.layer_names.len(), 8);
        assert_eq!(config.match_coarse.match_type, "dual_softmax");
        assert_eq!(config.fine.layer_names, vec!["self", "cross"]);
        assert!(!config.coarse.temp_bug_fix);
    }

    #[test]
    fn indoor_new_only_flips_temp_bug_fix() {
        let outdoor = LoftrConfig::outdoor();
        let indoor = LoftrConfig::indoor_new();
        assert!(indoor.coarse.temp_bug_fix);
        assert_eq!(indoor.backbone_type, outdoor.backbone_type);
        assert_eq!(indoor.match_coarse.thr, outdoor.match_coarse.thr);
        assert_eq!(indoor.fine, outdoor.fine);
    }
}
