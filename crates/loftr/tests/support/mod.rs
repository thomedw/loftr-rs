use std::{
    collections::HashMap,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use loftr::{LoftrConfig, LoftrModel};
use tch::{Device, Tensor};

const FIXTURE_SETUP_HINT: &str =
    "run `./scripts/prepare_test_fixtures.sh` from the workspace root and retry";

pub(crate) struct ReferenceFixture {
    tensors: HashMap<String, Tensor>,
}

impl ReferenceFixture {
    pub(crate) fn load(stem: &str) -> Result<Self, Box<dyn Error>> {
        let metadata_path = fixture_path(&format!("{stem}.json"))?;
        let metadata = fs::read_to_string(&metadata_path)?;
        let _: serde_json::Value = serde_json::from_str(&metadata)?;

        let tensor_path = fixture_path(&format!("{stem}.safetensors"))?;
        let tensors = Tensor::read_safetensors(&tensor_path)?
            .into_iter()
            .collect::<HashMap<_, _>>();
        Ok(Self { tensors })
    }

    pub(crate) fn tensor(&self, name: &str) -> Result<Tensor, Box<dyn Error>> {
        self.tensors
            .get(name)
            .map(Tensor::shallow_clone)
            .ok_or_else(|| format!("missing reference tensor `{name}`").into())
    }
}

pub(crate) fn fixture_path(name: &str) -> Result<PathBuf, Box<dyn Error>> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    if path.is_file() {
        Ok(path)
    } else {
        Err(format!(
            "missing test fixture `{name}` at `{}`; {FIXTURE_SETUP_HINT}",
            path.display()
        )
        .into())
    }
}

pub(crate) fn load_model(
    weights_name: &str,
    config: LoftrConfig,
) -> Result<LoftrModel, Box<dyn Error>> {
    let weights = fixture_path(weights_name)?;
    let mut model = LoftrModel::new(Device::Cpu, config)?;
    model.load_weights(&weights)?;
    Ok(model)
}
