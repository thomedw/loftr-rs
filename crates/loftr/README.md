# `loftr`

Native Rust/tch LoFTR feature matching.

This crate is alpha-quality. Expect APIs, docs, and runtime behavior to change before `1.0`, and
validate results carefully before relying on it in production workflows.

## What This Crate Includes

- LoFTR config presets
- native model construction in Rust with `tch`
- loading Kornia-compatible state dict weights
- pairwise feature matching
- optional debug-stage dumps for validation

## Upstream References

- [LoFTR upstream](https://github.com/zju3dv/LoFTR)
- [Kornia upstream](https://github.com/kornia/kornia)
- [kornia-rs upstream](https://github.com/kornia/kornia-rs)

## Weights

This crate does not ship pretrained weights.

Generate them locally from the workspace root:

```bash
./scripts/generate_loftr_state_dict.sh
```

Then load them from:

```text
artifacts/weights/loftr_outdoor_state_dict.safetensors
```

## Example

```rust
use loftr::{LoftrConfig, LoftrModel};
use tch::{Device, Kind, Tensor};

let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
model.load_weights("artifacts/weights/loftr_outdoor_state_dict.safetensors")?;

let image0 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
let image1 = Tensor::rand([1, 1, 128, 128], (Kind::Float, Device::Cpu));
let matches = model.forward(&image0, &image1)?;

println!("match count = {}", matches.confidence.size()[0]);
# Ok::<(), loftr::LoftrError>(())
```
