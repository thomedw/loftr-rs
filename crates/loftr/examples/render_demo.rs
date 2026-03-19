use std::{cmp::Ordering, env, error::Error, fs, path::Path};

use image::{DynamicImage, GrayImage, Rgb, RgbImage, imageops::FilterType};
use loftr::{LoftrConfig, LoftrMatches, LoftrModel};
use tch::{Device, Kind, Tensor};

#[derive(Clone, Debug)]
struct MatchViz {
    start: (f32, f32),
    end: (f32, f32),
    confidence: f32,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if !(args.len() == 5 || args.len() == 6) {
        eprintln!(
            "usage: cargo run -p loftr --example render_demo -- <weights> <left> <right> <output> [max_matches]"
        );
        std::process::exit(2);
    }

    let max_matches = args
        .get(5)
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(96);

    let mut model = LoftrModel::new(Device::Cpu, LoftrConfig::outdoor())?;
    model.load_weights(&args[1])?;

    let (left_tensor, left_preview) = load_grayscale(Path::new(&args[2]))?;
    let (right_tensor, right_preview) = load_grayscale(Path::new(&args[3]))?;
    let matches = model.forward(&left_tensor, &right_tensor)?;
    let selected = select_matches(
        &matches,
        left_preview.width(),
        left_preview.height(),
        max_matches,
    )?;
    render_demo(
        &left_preview,
        &right_preview,
        &selected,
        Path::new(&args[4]),
    )?;

    println!("rendered {} matches to {}", selected.len(), args[4]);
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
    image.resize_exact(960, 540, FilterType::Triangle)
}

fn select_matches(
    matches: &LoftrMatches,
    width: u32,
    height: u32,
    max_matches: usize,
) -> Result<Vec<MatchViz>, Box<dyn Error>> {
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

    let cols = 12usize;
    let rows = 8usize;
    let mut occupied = vec![false; cols * rows];
    let mut used = vec![false; candidates.len()];
    let mut selected = Vec::with_capacity(max_matches.min(candidates.len()));

    for (index, candidate) in candidates.iter().enumerate() {
        let cell = grid_index(candidate.start, width, height, cols, rows);
        if occupied[cell] {
            continue;
        }
        occupied[cell] = true;
        used[index] = true;
        selected.push(candidate.clone());
        if selected.len() == max_matches {
            return Ok(selected);
        }
    }

    for (index, candidate) in candidates.iter().enumerate() {
        if used[index] {
            continue;
        }
        selected.push(candidate.clone());
        if selected.len() == max_matches {
            break;
        }
    }

    Ok(selected)
}

fn grid_index(point: (f32, f32), width: u32, height: u32, cols: usize, rows: usize) -> usize {
    let x = (point.0 / width as f32).clamp(0.0, 0.999_999);
    let y = (point.1 / height as f32).clamp(0.0, 0.999_999);
    let col = (x * cols as f32) as usize;
    let row = (y * rows as f32) as usize;
    row * cols + col
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

    for (index, matched) in matches.iter().enumerate() {
        let color = palette(index, matches.len());
        let start = (
            left_origin.0 as f32 + matched.start.0,
            left_origin.1 as f32 + matched.start.1,
        );
        let end = (
            right_origin.0 as f32 + matched.end.0,
            right_origin.1 as f32 + matched.end.1,
        );
        draw_line(&mut canvas, start, end, color, 0.68);
        draw_disc(
            &mut canvas,
            start.0.round() as i32,
            start.1.round() as i32,
            3,
            color,
            0.95,
        );
        draw_disc(
            &mut canvas,
            end.0.round() as i32,
            end.1.round() as i32,
            3,
            color,
            0.95,
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

fn palette(index: usize, count: usize) -> Rgb<u8> {
    let hue = 0.05 + 0.85 * (index as f32 / count.max(1) as f32);
    let (red, green, blue) = hsv_to_rgb(hue.fract(), 0.72, 0.96);
    Rgb([
        (red * 255.0).round() as u8,
        (green * 255.0).round() as u8,
        (blue * 255.0).round() as u8,
    ])
}

fn hsv_to_rgb(hue: f32, saturation: f32, value: f32) -> (f32, f32, f32) {
    let chroma = value * saturation;
    let scaled = hue * 6.0;
    let x = chroma * (1.0 - ((scaled % 2.0) - 1.0).abs());
    let (red, green, blue) = match scaled as i32 {
        0 => (chroma, x, 0.0),
        1 => (x, chroma, 0.0),
        2 => (0.0, chroma, x),
        3 => (0.0, x, chroma),
        4 => (x, 0.0, chroma),
        _ => (chroma, 0.0, x),
    };
    let match_value = value - chroma;
    (red + match_value, green + match_value, blue + match_value)
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
