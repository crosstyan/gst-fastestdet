use anyhow::anyhow;
use clap::{Parser};
use gstfastestdet::fastestdet::common::{nms_handle, paint_targets, ImageModel};
use gstfastestdet::fastestdet::fastest_det::FastestDet;
use gstfastestdet::fastestdet::yolo_fastest::YoloFastest;
use image::buffer::ConvertBuffer;
use image::{RgbImage, Rgb32FImage};
use serde_derive::{Deserialize};
use protobuf::Message;
use crate::matrix::matrix::Mat;
mod matrix;

#[derive(Parser, Debug)]
#[command(author, about, long_about = None)]
struct Args {
    /// The directory to search for pictures
    #[arg(short, long)]
    input: String,
    /// param
    #[arg(long)]
    param_path: String,
    /// bin
    #[arg(long)]
    model_path: String,
    /// toml
    #[arg(long)]
    classes_path: String,
    /// nms
    #[arg(short, long, default_value_t = 0.25)]
    nms_threshold: f32,
    /// output
    #[arg(short, long)]
    output: String,
}

#[derive(Deserialize, Debug)]
struct Classes {
    pub classes: Vec<String>,
}

fn mat_to_rgbimg(mat:&ncnn_rs::Mat) -> anyhow::Result<Rgb32FImage>{
  let (w, h) = (mat.w(), mat.h());
  let data = unsafe {
    let p = mat.data() as *mut f32;
    std::slice::from_raw_parts_mut(p, (w * h * 3) as usize)
  }.to_vec();
  let mut img = image::Rgb32FImage::from_raw(w as u32, h as u32, data).ok_or(anyhow!("not rgb"))?;
  Ok(img)
}

fn pb_mat_to_ncnn(mat:&Mat)-> anyhow::Result<ncnn_rs::Mat>{
  let (w, h) = (mat.width, mat.height);
  println!("w: {}, h: {}", w, h);
  let v = mat.data.iter().map(|x| x.to_ne_bytes()).flatten().collect::<Vec<u8>>();
  const NUM_CHN:i32 = 3;
  const UNIT_SZ:usize = std::mem::size_of::<f32>();
  let dummy = vec![0; (w * h * NUM_CHN) as usize];
  println!("v: {}, dummy: {}", v.len(), dummy.len());
  let m = ncnn_rs::Mat::from_pixels(&dummy, ncnn_rs::MatPixelType::BGR, w, h, None)?;
  unsafe {
    let p = m.data() as *mut u8;
    let s = std::slice::from_raw_parts_mut(p, (w * h * NUM_CHN) as usize * UNIT_SZ);
    s.copy_from_slice(&v);
  }
  Ok(m)
}

fn img_resize(img: &RgbImage, size: (i32, i32)) -> anyhow::Result<Rgb32FImage> {
    use ncnn_rs::Mat;
    let img_data = img.as_flat_samples().samples;
    let img_size = (img.width() as i32, img.height() as i32);
    let (_, _, height_stride) = img.as_flat_samples().strides_cwh();
    dbg!(height_stride);
    use ncnn_rs::MatPixelType;
    let stride = height_stride;
    let mut input = Mat::from_pixels_resize(
        img_data,
        MatPixelType::RGB.convert(&MatPixelType::BGR),
        img_size,
        stride as i32,
        size,
        None,
    )?;

  let mean_vals: Vec<f32> = vec![0.0, 0.0, 0.0];
  let norm_vals: Vec<f32> = vec![1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0];
  input.substract_mean_normalize(&mean_vals, &norm_vals);
  let resized = mat_to_rgbimg(&input)?;
  Ok(resized)
}

pub fn main() -> Result<(), anyhow::Error>{
  let args = Args::parse();
  let mut img = image::open(args.input)?;
  let (w, h) = (img.width() as i32, img.height() as i32);
  let content = std::fs::read_to_string(args.classes_path)?;
  let classes = toml::from_str::<Classes>(&content)?.classes;
  // let mut det = FastestDet::new(
  //   args.param_path,
  //   args.model_path,
  //   (352, 352),
  //   classes
  // )?;
  let mut det = YoloFastest::new(
    args.param_path,
    args.model_path,
    classes
  )?;
  let rgb_img = img.as_mut_rgb8().ok_or(anyhow::anyhow!("not rgb8"))?;
  let img_mat = det.preprocess(rgb_img)?;
  let targets = det.detect(&img_mat, (w, h), 0.3)?;
  let nms_targets = nms_handle(&targets, args.nms_threshold);
  println!("nms_targets: {}", nms_targets.len());
  dbg!(&nms_targets);
  paint_targets(rgb_img, &nms_targets, det.labels())?;
  rgb_img.save(args.output)?;
  Ok(())
}
