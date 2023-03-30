use clap::{ArgAction, Parser};
use gstfastestdet::fastestdet::common::{nms_handle, paint_targets, ImageModel};
use gstfastestdet::fastestdet::fastest_det::FastestDet;
use serde_derive::{Deserialize, Serialize};

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
    #[arg(short, long, default_value_t = 0.45)]
    nms_threshold: f32,
    /// output
    #[arg(short, long)]
    output: String,
}

#[derive(Deserialize, Debug)]
struct Classes {
    pub classes: Vec<String>,
}


pub fn main() -> Result<(), anyhow::Error>{
  let args = Args::parse();
  let mut img = image::open(args.input)?;
  let (w, h) = (img.width() as i32, img.height() as i32);
  let content = std::fs::read_to_string(args.classes_path)?;
  let classes = toml::from_str::<Classes>(&content)?.classes;
  let mut det = FastestDet::new(
    args.param_path,
    args.model_path,
    (352, 352),
    classes
  )?;
  let rgb_img = img.as_mut_rgb8().ok_or(anyhow::anyhow!("not rgb8"))?;
  let mat = det.preprocess(rgb_img)?;
  let targets = det.detect(&mat, (w, h), 0.65).unwrap();
  let nms_targets = nms_handle(&targets, args.nms_threshold);
  dbg!(&nms_targets);
  paint_targets(rgb_img, &nms_targets, det.labels())?;
  rgb_img.save(args.output)?;
  Ok(())
}
