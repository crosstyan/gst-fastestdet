// adapted from
// https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.cpp
use super::common::{ImageModel, RgbBuffer, TargetBox};

use anyhow::{bail, Result};
use ncnn_rs::{Allocator as NcnnAllocator, Mat, Net};
use std::ops::{Deref};

const NUM_ANCHOR: usize = 3;
const ANCHOR: [f32; 12] = [
    12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87,
];
pub struct YoloFastest {
    alloc: NcnnAllocator,
    net: Net,
    classes: Vec<String>,
    model_size: (i32, i32),
}

// Maybe I should use mutex instead
unsafe impl Send for YoloFastest {}
unsafe impl Sync for YoloFastest {}

/// See also `getCategory`
/// return (Category, Index, Score)
fn category_score<T>(values: &[f32], index: usize, category: &[T]) -> (String, usize, f32)
where
    T: AsRef<str>,
{
    let num_anchor = NUM_ANCHOR;
    let num_category = category.len();
    let obj_score = values[4 * num_anchor + index as usize];
    let (idx, score) = (0..num_category)
        .map(|i| {
            let score = values[4 * num_anchor + num_anchor + i];
            let class_score = obj_score * score;
            (i, class_score)
        })
        .max_by(|(_, score1), (_, score2)| score1.partial_cmp(score2).unwrap())
        .unwrap();
    let idx = idx as usize;
    (category[idx].as_ref().to_string(), idx, score)
}

impl ImageModel for YoloFastest {
    // https://github.com/Tencent/ncnn/blob/bae2ee375fe025776d18a489a92a7f2357af7312/src/c_api.h#L103
    /// I assume you will read it from image crate with
    fn preprocess<T: Deref<Target = [u8]> + AsRef<[u8]>>(&self, img: &RgbBuffer<T>) -> Result<Mat> {
        let mean_vals: Vec<f32> = vec![0.0, 0.0, 0.0];
        let norm_vals: Vec<f32> = vec![1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0];
        let img_size = (img.width() as i32, img.height() as i32);
        dbg!(img.as_flat_samples().layout);
        let img_data = img.as_flat_samples().samples;
        // NOTE: not sure whether it is correct
        // https://blog.csdn.net/qianqing13579/article/details/45318279
        let (_, _, height_stride) = img.as_flat_samples().strides_cwh();
        dbg!(height_stride);
        use ncnn_rs::MatPixelType;
        let stride = height_stride;
        let mut input = Mat::from_pixels_resize(
            img_data,
            MatPixelType::RGB.convert(&MatPixelType::BGR),
            img_size,
            stride as i32,
            self.model_size,
            Some(&self.alloc),
        )?;
        input.substract_mean_normalize(&mean_vals, &norm_vals);
        Ok(input)
    }

    fn detect(&mut self, input: &Mat, img_size: (i32, i32), thresh: f32) -> Result<Vec<TargetBox>> {
        let mut ex = self.net.create_extractor();
        // magic string
        // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L142
        ex.input("input.1", input)?;
        let s = (self.model_size.0 * self.model_size.1 * 3) as usize;
        let mut v = Vec::<f32>::with_capacity(s);
        for it in 0..s {
            v.push(input[it]);
        }
        let acc = v.iter()
        .fold((0, 0f32),|(cnt, cum), val| (cnt+1, cum+val));
        let average = acc.1 / acc.0 as f32;
        println!("first {} el: {}", s, average);
        // dbg!(&input[..10]);
        let mut outputs: [Mat; 2] = [Mat::new(), Mat::new()];
        ex.extract("794", &mut outputs[0])?;
        ex.extract("796", &mut outputs[1])?;
        let mut target_boxes: Vec<TargetBox> = Vec::new();
        let (input_height, input_width) = self.model_size;
        let (img_w, img_h) = img_size;
        let scale_w = img_w as f32 / input_width as f32;
        let scale_h = img_h as f32 / input_height as f32;
        for (i, output) in outputs.iter().enumerate() {
            let out_h = output.c();
            let out_w = output.h();
            let out_c = output.w();
            assert!(input_height / out_h == input_width / out_w);
            let stride = input_height / out_h;
            // dbg!(out_h, out_w, out_c, stride, scale_w, scale_h);
            for h in 0..out_h {
                let mut values = unsafe { output.channel_data(h) };
                for w in 0..out_w {
                    for b in 0..NUM_ANCHOR {
                        let b = b;
                        let (_, idx, score) = category_score(&values, b, &self.classes);
                        if score > thresh {
                            let bcx = (values[b * 4 + 0] * 2.0 - 0.5 + w as f32) * stride as f32;
                            let bcy = (values[b * 4 + 1] * 2.0 - 0.5 + h as f32) * stride as f32;
                            let bw = values[b * 4 + 2].powi(2)
                                * ANCHOR[(i * NUM_ANCHOR * 2) + b as usize * 2 + 0];
                            let bh = values[b * 4 + 3].powi(2)
                                * ANCHOR[(i * NUM_ANCHOR * 2) + b as usize * 2 + 1];
                            let x1 = ((bcx - bw / 2.0) * scale_w) as i32;
                            let x2 = ((bcx + bw / 2.0) * scale_w) as i32;
                            let y1 = ((bcy - bh / 2.0) * scale_h) as i32;
                            let y2 = ((bcy + bh / 2.0) * scale_h) as i32;
                            let target = TargetBox {
                                x1,
                                y1,
                                x2,
                                y2,
                                class: idx as i32,
                                score,
                            };
                            target_boxes.push(target);
                        }
                    }
                    unsafe {
                        let p = (values.as_ptr().offset(out_c as isize)) as *mut f32;
                        values = std::slice::from_raw_parts_mut(p, std::usize::MAX);
                    };
                }
            }
        }
        Ok(target_boxes)
    }

    fn labels(&self) -> &Vec<String> {
        &self.classes
    }
}

impl YoloFastest {
    /// model_size: (width, height)
    pub fn new<P>(
        param_path: P,
        model_path: P,
        // model_size should be 352*352 for pretrained FastestDet
        classes: Vec<String>,
    ) -> Result<Self>
    where
        P: AsRef<str>,
    {
        let mut fastest_det = YoloFastest {
            alloc: unsafe { NcnnAllocator::new() },
            net: Net::new(),
            classes,
            model_size: (352, 352),
        };
        if let Err(e) = fastest_det.net.load_param(param_path.as_ref()) {
            anyhow::bail!("load param failed: {}", e);
        }
        if let Err(e) = fastest_det.net.load_model(model_path.as_ref()) {
            anyhow::bail!("load model failed: {}", e);
        };
        Ok(fastest_det)
    }
}