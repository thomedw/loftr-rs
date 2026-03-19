use std::{cmp::Ordering, env, error::Error, fs, path::Path};

use image::{DynamicImage, GrayImage, Rgb, RgbImage, imageops::FilterType};
use loftr::{LoftrConfig, LoftrMatches, LoftrModel};
use tch::{Device, Kind, Tensor};

const DEMO_WIDTH: u32 = 376;
const DEMO_HEIGHT: u32 = 600;

#[derive(Clone, Debug)]
struct MatchViz {
    start: (f32, f32),
    end: (f32, f32),
    confidence: f32,
}

#[derive(Debug)]
struct MatchSelection {
    visible: Vec<MatchViz>,
    total: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if !(args.len() >= 5 && args.len() <= 7) {
        eprintln!(
            "usage: cargo run -p loftr --example render_demo -- <weights> <left> <right> <output> [top_k] [bottom_k]"
        );
        std::process::exit(2);
    }

    let top_k = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(2000);
    let bottom_k = args
        .get(6)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(0);

    if bottom_k > top_k {
        return Err(
            format!("bottom_k must be <= top_k; got bottom_k={bottom_k}, top_k={top_k}").into(),
        );
    }

    let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
    model.load_weights(&args[1])?;

    let (left_tensor, left_preview) = load_grayscale(Path::new(&args[2]))?;
    let (right_tensor, right_preview) = load_grayscale(Path::new(&args[3]))?;
    let matches = model.forward(&left_tensor, &right_tensor)?;
    let selection = select_matches(&matches, top_k, bottom_k)?;
    render_demo(
        &left_preview,
        &right_preview,
        &selection.visible,
        Path::new(&args[4]),
    )?;

    println!(
        "rendered {} of {} matches to {} (showing range {}:{})",
        selection.visible.len(),
        selection.total,
        args[4],
        bottom_k,
        top_k
    );
    Ok(())
}

fn load_grayscale(path: &Path) -> Result<(Tensor, GrayImage), Box<dyn Error>> {
    let image = image::open(path)?;
    let image = resize_for_loftr(image);

    let preview = image.to_luma8();
    let image = image.to_luma32f();
    let height = i64::from(image.height());
    let width = i64::from(image.width());
    let data = image.into_raw();
    let tensor = Tensor::from_slice(&data)
        .view([1, height, width])
        .unsqueeze(0)
        .to_kind(Kind::Float);
    Ok((tensor, preview))
}

fn resize_for_loftr(image: DynamicImage) -> DynamicImage {
    // Kornia's tutorial uses a 600x375 (H x W) resize for this pair.
    // The current native port needs even spatial dimensions for fine matching,
    // so the demo rounds the width up by one pixel.
    image.resize_exact(DEMO_WIDTH, DEMO_HEIGHT, FilterType::Triangle)
}

fn select_matches(
    matches: &LoftrMatches,
    top_k: usize,
    bottom_k: usize,
) -> Result<MatchSelection, Box<dyn Error>> {
    let keypoints0 = tensor_to_points(&matches.keypoints0)?;
    let keypoints1 = tensor_to_points(&matches.keypoints1)?;
    let confidence = Vec::<f32>::try_from(matches.confidence.reshape([-1]))?;

    let mut candidates = keypoints0
        .into_iter()
        .zip(keypoints1)
        .zip(confidence)
        .map(|((start, end), confidence)| MatchViz {
            start,
            end,
            confidence,
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(Ordering::Equal)
    });

    let total = candidates.len();
    let end = top_k.min(total);
    let start = bottom_k.min(end);
    let visible = candidates[start..end].to_vec();
    Ok(MatchSelection { visible, total })
}

fn tensor_to_points(tensor: &Tensor) -> Result<Vec<(f32, f32)>, Box<dyn Error>> {
    let flat = tensor.reshape([-1]);
    let values = Vec::<f32>::try_from(flat)?;
    let mut points = Vec::with_capacity(values.len() / 2);
    for chunk in values.chunks_exact(2) {
        points.push((chunk[0], chunk[1]));
    }
    Ok(points)
}

fn render_demo(
    left: &GrayImage,
    right: &GrayImage,
    matches: &[MatchViz],
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let pad = 24u32;
    let gap = 36u32;
    let border = 6u32;
    let width = pad * 2 + left.width() + right.width() + gap + border * 4;
    let height = pad * 2 + left.height() + border * 2;
    let mut canvas = RgbImage::from_pixel(width, height, Rgb([12, 18, 24]));

    let left_box = (
        pad,
        pad,
        left.width() + border * 2,
        left.height() + border * 2,
    );
    let right_box = (
        pad + left_box.2 + gap,
        pad,
        right.width() + border * 2,
        right.height() + border * 2,
    );
    fill_rect(&mut canvas, left_box, Rgb([232, 236, 241]));
    fill_rect(&mut canvas, right_box, Rgb([232, 236, 241]));

    let left_origin = (left_box.0 + border, left_box.1 + border);
    let right_origin = (right_box.0 + border, right_box.1 + border);
    blit_grayscale(&mut canvas, left, left_origin);
    blit_grayscale(&mut canvas, right, right_origin);

    let max_confidence = matches
        .iter()
        .fold(0.0_f32, |acc, matched| acc.max(matched.confidence));

    for matched in matches {
        let normalized = if max_confidence > 1e-5 {
            (matched.confidence / (max_confidence + 1e-5)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let color = jet_color(normalized);
        let start = (
            left_origin.0 as f32 + matched.start.0,
            left_origin.1 as f32 + matched.start.1,
        );
        let end = (
            right_origin.0 as f32 + matched.end.0,
            right_origin.1 as f32 + matched.end.1,
        );
        draw_line(&mut canvas, start, end, color, 0.8);
        draw_disc(
            &mut canvas,
            start.0.round() as i32,
            start.1.round() as i32,
            2,
            color,
            1.0,
        );
        draw_disc(
            &mut canvas,
            end.0.round() as i32,
            end.1.round() as i32,
            2,
            color,
            1.0,
        );
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    canvas.save(output)?;
    Ok(())
}

fn fill_rect(image: &mut RgbImage, rect: (u32, u32, u32, u32), color: Rgb<u8>) {
    let (left, top, width, height) = rect;
    for y in top..top + height {
        for x in left..left + width {
            image.put_pixel(x, y, color);
        }
    }
}

fn blit_grayscale(canvas: &mut RgbImage, image: &GrayImage, origin: (u32, u32)) {
    for y in 0..image.height() {
        for x in 0..image.width() {
            let value = image.get_pixel(x, y)[0];
            canvas.put_pixel(origin.0 + x, origin.1 + y, Rgb([value, value, value]));
        }
    }
}

fn jet_color(value: f32) -> Rgb<u8> {
    let x = value.clamp(0.0, 1.0);
    let red = (1.5 - (4.0 * x - 3.0).abs()).clamp(0.0, 1.0);
    let green = (1.5 - (4.0 * x - 2.0).abs()).clamp(0.0, 1.0);
    let blue = (1.5 - (4.0 * x - 1.0).abs()).clamp(0.0, 1.0);
    Rgb([
        (red * 255.0).round() as u8,
        (green * 255.0).round() as u8,
        (blue * 255.0).round() as u8,
    ])
}

fn draw_line(
    canvas: &mut RgbImage,
    start: (f32, f32),
    end: (f32, f32),
    color: Rgb<u8>,
    alpha: f32,
) {
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let steps = dx.abs().max(dy.abs()).max(1.0).ceil() as i32;
    for step in 0..=steps {
        let t = step as f32 / steps as f32;
        let x = start.0 + dx * t;
        let y = start.1 + dy * t;
        draw_disc(canvas, x.round() as i32, y.round() as i32, 1, color, alpha);
    }
}

fn draw_disc(
    canvas: &mut RgbImage,
    center_x: i32,
    center_y: i32,
    radius: i32,
    color: Rgb<u8>,
    alpha: f32,
) {
    for offset_y in -radius..=radius {
        for offset_x in -radius..=radius {
            if offset_x * offset_x + offset_y * offset_y > radius * radius {
                continue;
            }
            blend_pixel(
                canvas,
                center_x + offset_x,
                center_y + offset_y,
                color,
                alpha,
            );
        }
    }
}

fn blend_pixel(canvas: &mut RgbImage, x: i32, y: i32, color: Rgb<u8>, alpha: f32) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as u32;
    let y = y as u32;
    if x >= canvas.width() || y >= canvas.height() {
        return;
    }

    let destination = canvas.get_pixel_mut(x, y);
    let src_alpha = alpha.clamp(0.0, 1.0);
    let dst_alpha = 1.0 - src_alpha;
    for channel in 0..3 {
        destination[channel] = (destination[channel] as f32 * dst_alpha
            + color[channel] as f32 * src_alpha)
            .round() as u8;
    }
}
