use image::{ImageBuffer, Rgb};
use ncnn_rs::{Mat};
use anyhow::{Result};
use once_cell::sync::Lazy;
use rusttype::{Font, Scale};
use serde_derive::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut, Index};

static FONT: Lazy<&[u8]> = Lazy::new(|| include_bytes!("DejaVuSans.ttf"));

/// trait aliases are experimental
///
/// ```rust
/// trait D = Deref<Target = [u8]> + DerefMut<Target=[u8]> + AsRef<[u8]>;
/// ```
pub type RgbBuffer<T> = ImageBuffer<Rgb<u8>, T>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    pub score: f32,
    pub class: i32,
}

impl TargetBox {
    pub fn width(&self) -> i32 {
        self.x2 - self.x1
    }
    pub fn height(&self) -> i32 {
        self.y2 - self.y1
    }
    pub fn area(&self) -> i32 {
        self.width() * self.height()
    }
    pub fn intersection_area(&self, other: &TargetBox) -> i32 {
        intersection_area(self, other)
    }
}

pub fn paint_targets<T: Deref<Target = [u8]> + DerefMut<Target = [u8]>>(
    paint_img: &mut RgbBuffer<T>,
    targets: &Vec<TargetBox>,
    classes: &Vec<String>,
) -> Result<(), anyhow::Error> {
    for target in targets.iter() {
        let (x1, y1, x2, y2) = (target.x1, target.y1, target.x2, target.y2);
        let rect = imageproc::rect::Rect::at(x1 as i32, y1 as i32)
            .of_size((x2 - x1).try_into().unwrap(), (y2 - y1).try_into().unwrap())
            .try_into()?;
        let color_text = image::Rgb([242, 255, 128]);
        let color = image::Rgb([0, 255, 2]);
        imageproc::drawing::draw_hollow_rect_mut(paint_img, rect, color);
        let class_name = classes.index(target.class as usize);
        let font = Font::try_from_bytes(&FONT).ok_or(anyhow::anyhow!("font error"))?;
        let height = 24.8;
        let scale = Scale {
            x: height,
            y: height,
        };
        imageproc::drawing::draw_text_mut(
            paint_img, color_text, x1 as i32, y1 as i32, scale, &font, class_name,
        );
    }
    Ok(())
}

pub fn intersection_area(a: &TargetBox, b: &TargetBox) -> i32 {
    if a.x1 > b.x2 || a.x2 < b.x1 || a.y1 > b.y2 || a.y2 < b.y1 {
        // no intersection
        return 0;
    }
    let intersect_width = (a.x2.min(b.x2) - a.x1.max(b.x1)).max(0);
    let intersect_height = (a.y2.min(b.y2) - a.y1.max(b.y1)).max(0);
    intersect_width * intersect_height
}

// will allocate a new vector
pub fn nms_handle(boxes: &[TargetBox], nms_threshold: f32) -> Vec<TargetBox> {
    let mut picked: Vec<TargetBox> = Vec::new();
    let mut sorted_boxes: Vec<TargetBox> = boxes.to_owned();
    // unwrap here is intentional
    sorted_boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    for target_box in sorted_boxes.iter() {
        let mut keep = true;
        for picked_box in picked.iter() {
            let intersection = intersection_area(target_box, picked_box);
            let union = target_box.area() + picked_box.area() - intersection;
            // Intersection over Union
            // avoid divide by zero (in real world, this should not happen)
            let iou = if union != 0 {
                intersection as f32 / union as f32
            } else {
                0.0
            };
            if iou > nms_threshold && target_box.class == picked_box.class {
                keep = false;
                break;
            }
        }
        if keep {
            picked.push(target_box.clone());
        }
    }
    picked
}

pub trait ImageModel {
    fn preprocess<T: Deref<Target = [u8]> + AsRef<[u8]>>(
        &self,
        img: &RgbBuffer<T>,
    ) -> Result<Mat>;

    fn inference(&mut self, input: &Mat, img_size: (i32, i32), thresh: f32) -> Result<Vec<TargetBox>>;
    fn detect(&mut self , img: &RgbBuffer<Vec<u8>>, thresh: f32) -> Result<Vec<TargetBox>> {
        let input = self.preprocess(img)?;
        let img_size = (img.width() as i32, img.height() as i32);
        self.inference(&input, img_size, thresh)
    }
    fn labels(&self) -> &Vec<String>;
}
