use super::utils::*;
use anyhow::{bail, Result};
use ncnn_rs::{Allocator as ncnn_Allocator, Mat, Net};
use opencv::core::Mat as CvMat;
use opencv::core::*;
use std::ops::Index;

// adapted from
// https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.cpp

#[derive(Debug, Clone)]
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

pub struct FastestDet {
    alloc: ncnn_Allocator,
    net: Net,
    classes: Vec<String>,
    /// 模型输入宽高
    model_size: (i32, i32),
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
        let fastest_det = FastestDet {
            alloc: ncnn_Allocator::new(),
            net: Net::new(),
            classes,
            model_size,
        };
        if let Err(e) = fastest_det.net.load_param(param_path.as_ref()) {
            anyhow::bail!("load param failed: {}", e);
        }
        if let Err(e) = fastest_det.net.load_model(model_path.as_ref()) {
            anyhow::bail!("load model failed: {}", e);
        };
        Ok(fastest_det)
    }

    // https://github.com/Tencent/ncnn/blob/bae2ee375fe025776d18a489a92a7f2357af7312/src/c_api.h#L103
    /// I assume you will read it from OpenCV so the colorspace is BGR!
    pub fn preprocess(&self, img: &CvMat) -> Result<Mat> {
        let mean_vals: Vec<f32> = vec![0.0, 0.0, 0.0];
        let norm_vals: Vec<f32> = vec![1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0];
        let img_size = (img.cols(), img.rows());
        let img_data = unsafe { std::slice::from_raw_parts(img.data(), img.total() as usize) };
        // stride = cols * channels = 500 * 3 = 1500
        let stride = match img.step1(0) {
            Ok(s) => s,
            Err(e) => bail!("img.step1(0) failed: {:?}", e),
        };
        // https://github.com/Tencent/ncnn/blob/5eb56b2ea5a99fb5a3d6f3669ef1743b73a9a53e/src/mat.h#L261
        // https://learn.microsoft.com/en-us/windows/win32/medfound/image-stride
        // https://stackoverflow.com/questions/11572156/stride-on-image-using-opencv-c
        // In fact, if isContinuous() returns true it means that stride == cols, so
        // the "stride" becomes the same of the image width. What is zero in that
        // case is, maybe, the padding.
        // convenient construct from pixel data with stride(bytes-per-row) parameter
        // https://docs.opencv.org/2.4/modules/core/doc/basic_structures.html#mat-mat
        // step – Number of bytes each matrix row occupies. The value should include
        // the padding bytes at the end of each row, if any. If the parameter is
        // missing (set to AUTO_STEP ), no padding is assumed and the actual step is
        // calculated as cols*elemSize() .
        let mut input = Mat::from_pixels_resize(
            img_data,
            ncnn_bind::NCNN_MAT_PIXEL_BGR as i32,
            img_size,
            stride as i32,
            self.model_size,
            &self.alloc,
        );
        input.substract_mean_normalize(&mean_vals, &norm_vals);
        Ok(input)
    }

    pub fn detect(&self, input: &Mat, img_size: (i32, i32), thresh: f32) -> Result<Vec<TargetBox>> {
        let ex = self.net.create_extractor();
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
        let out_h = output.get_h();
        let out_w = output.get_w();
        let class_num = self.classes.len();
        // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L152
        for h in 0..out_h {
            for w in 0..out_w {
                let obj_score_idx = (0 * out_h * out_w) + (h * out_w) + w;
                let obj_score = output.index(obj_score_idx as isize);
                let mut max_score: f32 = 0.0;
                let mut class_index = 0;
                for idx in 0..class_num {
                    let idx = idx as i32;
                    // why 5? magic number?
                    // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L165
                    let score_idx = ((idx + 5) * out_h * out_w) + (h * out_w) + w;
                    let score = output.index(score_idx as isize).clone();
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

                    let x_offset = output.index(x_offset_index as isize).tanh();
                    let y_offset = output.index(y_offset_index as isize).tanh();
                    let box_width = output.index(box_width_index as isize).sigmoid();
                    let box_height = output.index(box_height_index as isize).sigmoid();

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
    pub fn classes(&self) -> &Vec<String> {
        &self.classes
    }
}
