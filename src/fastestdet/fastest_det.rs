use super::utils::*;
use anyhow::{bail, Result};
use ncnn_rs::{Allocator as ncnn_Allocator, Mat, Net};
use opencv::core::Mat as CvMat;
use opencv::core::*;
use std::ops::Index;
use serde_derive::{Deserialize, Serialize};

// adapted from
// https://github.com/dog-qiuqiu/FastestDet/blob/main/example/ncnn/FastestDet.cpp

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

// Maybe I should use mutex instead
unsafe impl Send for FastestDet {}
unsafe impl Sync for FastestDet {}

// const bias
// https://github.com/crosstyan/YoloFastestExample/blob/fe1d25c4e6709511f5e32090516dd8b808b12dd0/lib/yolo/yolo-fastestv2.cpp#L34
const ANCHOR: [f32; 12] = [12.64f32, 19.39, 37.88, 51.48, 55.71, 138.31,126.91, 78.23, 131.57, 214.55, 279.92, 258.87];

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

    pub fn detect(&self, input: &Mat, img_size: (i32, i32), thresh: f32, output_name: &str, output_index: i32) -> Result<Vec<TargetBox>> {
        let ex = self.net.create_extractor();
        // magic string
        // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L142
        // 
        // https://github.com/crosstyan/YoloFastestExample/blob/eccd6d557750833041b6e51d5ab8a7c26adc0d16/lib/yolo/yolo-fastestv2.cpp#L204
        // 模型输入输出节点名称
        // inputName = "input.1";
        // outputName1 = "794"; //22x22
        // outputName2 = "796"; //11x11
        if let Err(e) = ex.input("input.1", input) {
            bail!("ex.input error: {}", e);
        };
        let mut output = Mat::new();
        // magic name
        if let Err(e) = ex.extract(output_name, &mut output) {
            bail!("ex.extract error: {}", e);
        };
        let mut target_boxes: Vec<TargetBox> = Vec::new();
        let out_h = output.get_h();
        let out_w = output.get_w();
        //模型输入尺寸大小
        // 352 is a magic number
        let (input_w, input_h) = (352, 352);
        let (img_w, img_h) = img_size;
        // https://github.com/crosstyan/YoloFastestExample/blob/fe1d25c4e6709511f5e32090516dd8b808b12dd0/lib/yolo/yolo-fastestv2.cpp#L184
        let scale_w = img_w as f32 / input_w as f32;
        let scale_h = img_h as f32 / input_h as f32;
        dbg!(input_h, input_w, out_h, out_w, scale_h, scale_w);
        // input_h = 352
        // input_w = 352
        // out_h = 11
        // out_w = 95
        // why out_w is 95?
        // it should be 11x11 or 22x22
        assert!(input_h/out_h == input_w/out_w);
        // https://github.com/crosstyan/YoloFastestExample/blob/fe1d25c4e6709511f5e32090516dd8b808b12dd0/lib/yolo/yolo-fastestv2.cpp#L142
        let stride = input_w as f32 / out_w as f32;
        let class_num = self.classes.len();
        // https://github.com/dog-qiuqiu/FastestDet/blob/50473cd155cb088aa4a99e64ff6a4b3c24fa07e1/example/ncnn/FastestDet.cpp#L152
        // https://github.com/crosstyan/YoloFastestExample/blob/eccd6d557750833041b6e51d5ab8a7c26adc0d16/lib/yolo/yolo-fastestv2.cpp#L111
        // it use anchor...? what is anchor?
        // numAnchor = 3;
        let anchor_num = 3;
        for h in 0..out_h {
            for w in 0..out_w {
                for b in 0..anchor_num{
                    let obj_score_idx = 4 * anchor_num + b;
                    let obj_score = output.index(obj_score_idx as isize);
                    let mut max_score: f32 = 0.0;
                    let mut class_index = 0;
                    // https://github.com/crosstyan/YoloFastestExample/blob/eccd6d557750833041b6e51d5ab8a7c26adc0d16/lib/yolo/yolo-fastestv2.cpp#L111
                    // 4 is a magic number as well as anchor_num
                    // I didn't train the model, so I don't know why...
                    for idx in 0..class_num {
                        let idx = idx as i32;
                        let score_idx = 4 * anchor_num + anchor_num + b;
                        let score = output.index(score_idx as isize).clone();
                        let cls_score = score * obj_score;
                        if score > max_score {
                            max_score = cls_score;
                            class_index = idx;
                        }
                    }
                    let score = max_score;
                    if score > thresh {
                        let val_x = *output.index(b * 4 + 0);
                        let val_y = *output.index(b * 4 + 1);
                        let val_w = *output.index(b * 4 + 2);
                        let val_h = *output.index(b * 4 + 3);
                        let bcx = ((val_x * 2.0 - 0.5) + w as f32) * stride;
                        let bcy = ((val_y * 2.0 - 0.5) + h as f32) * stride;
                        let bw = ((val_w * 2.0).powf(2.0)) * ANCHOR[(output_index * anchor_num as i32 * 2 + b as i32 * 2 + 0) as usize];
                        let bh = ((val_h * 2.0).powf(2.0)) * ANCHOR[(output_index * anchor_num as i32 * 2 + b as i32 * 2 + 1) as usize];

                        let x1 = ((bcx - 0.5 * bw) * scale_w as f32) as i32;
                        let y1 = ((bcy - 0.5 * bh) * scale_h as f32) as i32;
                        let x2 = ((bcx + 0.5 * bw) * scale_w as f32) as i32;
                        let y2 = ((bcy + 0.5 * bh) * scale_h as f32) as i32;

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
        }
        Ok(target_boxes)
    }
    pub fn classes(&self) -> &Vec<String> {
        &self.classes
    }
}
