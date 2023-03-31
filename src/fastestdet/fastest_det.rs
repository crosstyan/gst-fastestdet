// adapted from
// https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.cpp
use super::common::{ImageModel, RgbBuffer, TargetBox};
use super::utils::*;
use anyhow::{bail, Result};
use ncnn_rs::{Allocator as NcnnAllocator, Mat, Net};
use std::ops::{Deref, Index};

pub struct FastestDet {
    alloc: NcnnAllocator,
    net: Net,
    classes: Vec<String>,
    /// 模型输入宽高
    model_size: (i32, i32),
}

// Maybe I should use mutex instead
unsafe impl Send for FastestDet {}
unsafe impl Sync for FastestDet {}

impl ImageModel for FastestDet {
    // https://github.com/Tencent/ncnn/blob/bae2ee375fe025776d18a489a92a7f2357af7312/src/c_api.h#L103
    /// I assume you will read it from image crate with
    fn preprocess<T: Deref<Target = [u8]> + AsRef<[u8]>>(&self, img: &RgbBuffer<T>) -> Result<Mat> {
        let mean_vals: Vec<f32> = vec![0.0, 0.0, 0.0];
        let norm_vals: Vec<f32> = vec![1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0];
        let img_size = (img.width() as i32, img.height() as i32);
        let img_data = img.as_flat_samples().samples;
        let (_, _, height_stride) = img.as_flat_samples().strides_cwh();
        let stride = height_stride;
        use ncnn_rs::MatPixelType;
        let mut input = Mat::from_pixels_resize(
            img_data,
            MatPixelType::RGB.to_int(),
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
        if let Err(e) = ex.input("input.1", input) {
            bail!("ex.input error: {}", e);
        };
        let mut output = Mat::new();
        // magic name
        if let Err(e) = ex.extract("758", &mut output) {
            bail!("ex.extract error: {}", e);
        };
        let mut target_boxes: Vec<TargetBox> = Vec::new();
        let (img_width, img_height) = img_size;
        let out_h = output.h();
        let out_w = output.w();
        let class_num = self.classes.len();
        // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L152
        for h in 0..out_h {
            for w in 0..out_w {
                let output = output.as_slice::<f32>();
                let obj_score_idx = (0 * out_h * out_w) + (h * out_w) + w;
                let obj_score = output[obj_score_idx as usize];
                let mut max_score: f32 = 0.0;
                let mut class_index = 0;
                for idx in 0..class_num {
                    let idx = idx as i32;
                    // why 5? magic number?
                    // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L165
                    let score_idx = ((idx + 5) * out_h * out_w) + (h * out_w) + w;
                    let score = output[score_idx as usize].clone();
                    if score > max_score {
                        max_score = score;
                        class_index = idx;
                    }
                }
                let score = max_score.powf(0.4) * obj_score.powf(0.6);
                if score > thresh {
                    let x_offset_index = (1 * out_h * out_w) + (h * out_w) + w;
                    let y_offset_index = (2 * out_h * out_w) + (h * out_w) + w;
                    let box_width_index = (3 * out_h * out_w) + (h * out_w) + w;
                    let box_height_index = (4 * out_h * out_w) + (h * out_w) + w;

                    let x_offset = output.index(x_offset_index as usize).tanh();
                    let y_offset = output.index(y_offset_index as usize).tanh();
                    let box_width = output.index(box_width_index as usize).sigmoid();
                    let box_height = output.index(box_height_index as usize).sigmoid();

                    let cx = (w as f32 + x_offset) / out_w as f32;
                    let cy = (h as f32 + y_offset) / out_h as f32;

                    let x1 = ((cx - box_width * 0.5) * img_width as f32) as i32;
                    let y1 = ((cy - box_height * 0.5) * img_height as f32) as i32;
                    let x2 = ((cx + box_width * 0.5) * img_width as f32) as i32;
                    let y2 = ((cy + box_height * 0.5) * img_height as f32) as i32;
                    let target_box = TargetBox {
                        x1,
                        y1,
                        x2,
                        y2,
                        score,
                        class: class_index,
                    };
                    target_boxes.push(target_box);
                }
            }
        }
        Ok(target_boxes)
    }

    fn labels(&self) -> &Vec<String> {
        &self.classes
    }
}

impl FastestDet {
    /// model_size: (width, height)
    pub fn new<P>(
        param_path: P,
        model_path: P,
        // model_size should be 352*352 for pretrained FastestDet
        model_size: (i32, i32),
        classes: Vec<String>,
    ) -> Result<Self>
    where
        P: AsRef<str>,
    {
        let mut det = FastestDet {
            alloc: unsafe { NcnnAllocator::new() },
            net: Net::new(),
            classes,
            model_size,
        };
        det.net.load_param(param_path.as_ref())?;
        det.net.load_model(model_path.as_ref())?;
        Ok(det)
    }
}
