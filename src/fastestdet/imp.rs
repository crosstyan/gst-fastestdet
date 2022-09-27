use gst::glib;
// use gst::glib::subclass::prelude::*;
use super::fastest_det::{nms_handle, FastestDet, TargetBox};
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{gst_debug, gst_error, gst_info, gst_log, gst_trace, gst_warning};
use gst_base::subclass::prelude::*;
use gst_video::subclass::prelude::*;
use opencv::core::Mat as CvMat;
use opencv::core::*;
use opencv::prelude::*;
use serde_derive::{Deserialize, Serialize};

use std::ffi::c_void;

use std::i32;
use std::ops::Index;
use std::sync::Mutex;

use once_cell::sync::Lazy;

// VideoInfo is a struct that contains various fields like width/height,
// framerate and the video format and allows to conveniently with the
// properties of (raw) video formats. We have to store it inside a Mutex in our
// struct as this can (in theory) be accessed from multiple threads at
// the same time.

// https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-good/sys/v4l2/gstv4l2src.c

/// This module contains the private implementation details of our element
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "fastestdetrs",
        gst::DebugColorFlags::empty(),
        Some("Rust FastestDet Element"),
    )
});

// https://gstreamer.freedesktop.org/documentation/application-development/advanced/buffering.html?gi-language=c#timeshift-buffering
pub fn paint_targets(
    paint_img: &mut CvMat,
    targets: &Vec<TargetBox>,
    classes: &Vec<String>,
) -> Result<(), anyhow::Error> {
    for target in targets.iter() {
        let blue = Scalar::new(255.0, 255.0, 0.0, 0.0);
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let thickness = 2;
        let line_type = opencv::imgproc::LINE_8;
        let shift = 0;
        opencv::imgproc::rectangle(
            paint_img,
            Rect::new(
                target.x1,
                target.y1,
                target.x2 - target.x1,
                target.y2 - target.y1,
            ),
            blue,
            thickness,
            line_type,
            shift,
        )?;
        let class_name = classes.index(target.class as usize);
        let text = format!("{}: {:.2}", class_name, target.score);
        opencv::imgproc::put_text(
            paint_img,
            &text,
            Point::new(target.x1, target.y1),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.75,
            green,
            thickness,
            line_type,
            false,
        )?;
    }
    Ok(())
}

#[derive(Deserialize, Debug)]
struct Classes {
    pub classes: Vec<String>,
}

const DEFAULT_MODEL_PATH: &'static str = "models.bin";
const DEFAULT_PARAM_PATH: &'static str = "models.param";
const DEFAULT_CLASSES_PATH: &'static str = "classes.toml";

pub struct Settings {
    model_path: String,
    param_path: String,
    classes_path: String,
    det: Option<FastestDet>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model_path: DEFAULT_MODEL_PATH.to_string(),
            param_path: DEFAULT_PARAM_PATH.to_string(),
            classes_path: DEFAULT_CLASSES_PATH.to_string(),
            det: None,
        }
    }
}

#[derive(Default)]
pub struct GstFastestDet {
    settings: Mutex<Settings>,
}

impl GstFastestDet {
    pub fn try_get_det(settings: &Settings) -> Result<FastestDet, anyhow::Error> {
        let classes_text = std::fs::read_to_string(settings.classes_path.clone())?;
        let classes = toml::from_str::<Classes>(&classes_text)?;
        let c = classes.classes;
        // TODO: using config
        let model_size = (352, 352);
        let det = FastestDet::new(&settings.param_path, &settings.model_path, model_size, c)?;
        Ok(det)
    }
}

// This trait registers our type with the GObject object system and
// provides the entry points for creating a new instance and setting
// up the class data
#[glib::object_subclass]
impl ObjectSubclass for GstFastestDet {
    const NAME: &'static str = "FastestDetRs";
    type Type = super::GstFastestDet;
    type ParentType = gst_video::VideoFilter;
    // See ElementImpl
}

// VideoFilter
// @extends gst_base::BaseTransform, gst::Element, gst::Object
// so you should implement the following traits:
/*
GObject                                   ObjectImpl
    ╰──GInitiallyUnowned
        ╰──GstObject                      GstObjectImpl
            ╰──GstElement                 ElementImpl
                ╰──GstBaseTransform       BaseTransformImpl
                    ╰──GstVideoFilter     VideoFilterImpl
 */

// Implementation of glib::Object virtual methods
impl ObjectImpl for GstFastestDet {
    // https://www.freedesktop.org/software/gstreamer-sdk/data/docs/latest/gobject/gobject-GParamSpec.html#G-PARAM-CONSTRUCT-ONLY:CAPS
    // https://www.freedesktop.org/software/gstreamer-sdk/data/docs/2012.5/gobject/howto-gobject-construction.html
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("model-path")
                    .nick("Model")
                    .blurb("Model path which should be ended with `.bin`")
                    .default_value(Some(DEFAULT_MODEL_PATH))
                    .flags(glib::ParamFlags::READWRITE)
                    .build(),
                glib::ParamSpecString::builder("config-path")
                    .nick("Config")
                    .blurb("Config path which should be ended with `.toml`")
                    .default_value(Some(DEFAULT_CLASSES_PATH))
                    .flags(glib::ParamFlags::READWRITE)
                    .build(),
                glib::ParamSpecString::builder("param-path")
                    .nick("Param")
                    .blurb("Param path which should be ended with `.param`")
                    .default_value(Some(DEFAULT_CLASSES_PATH))
                    .flags(glib::ParamFlags::READWRITE)
                    .build(),
                // TODO: use signal to reload model
                glib::ParamSpecBoolean::builder("run")
                    .nick("Run")
                    .blurb("if true, try to load and run model")
                    .default_value(false)
                    .flags(glib::ParamFlags::READWRITE)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }
    fn set_property(
        &self,
        obj: &Self::Type,
        _id: usize,
        value: &glib::Value,
        pspec: &glib::ParamSpec,
    ) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                // TODO: proper error detection
                settings.model_path = value.get().unwrap();
                settings.model_path = settings.model_path.trim().to_string();
                if settings.model_path.find(",").is_some() {
                    gst_warning!(CAT, obj: obj, "path should not contain `,`");
                }
                gst_info!(CAT, obj: obj, "Set model path to {}", settings.model_path);
            }
            "config-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.classes_path = value.get().unwrap();
                settings.classes_path = settings.classes_path.trim().to_string();
                gst_info!(
                    CAT,
                    obj: obj,
                    "Set config path to {}",
                    settings.classes_path
                );
            }
            "param-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.param_path = value.get().unwrap();
                settings.param_path = settings.param_path.trim().to_string();
                gst_info!(CAT, obj: obj, "Set param path to {}", settings.param_path);
            }
            "run" => {
                // https://coaxion.net/blog/2016/09/writing-gstreamer-elements-in-rust-part-2-dont-panic-we-have-better-assertions-now-and-other-updates/
                let run = value.get().unwrap();
                let mut settings = self.settings.lock().unwrap();
                if run {
                    let maybe_det = Self::try_get_det(&settings);
                    match maybe_det {
                        Ok(det) => {
                            settings.det = Some(det);
                            gst_info!(CAT, obj: obj, "model loaded");
                        }
                        Err(e) => {
                            gst_error!(CAT, obj: obj, "Failed to create det: {}", e);
                            panic!("Failed to create det: {}", e);
                        }
                    }
                } else {
                    settings.det = None;
                }
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _obj: &Self::Type, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        match pspec.name() {
            "model-path" => {
                let settings = self.settings.lock().unwrap();
                settings.model_path.to_value()
            }
            "config-path" => {
                let settings = self.settings.lock().unwrap();
                settings.classes_path.to_value()
            }
            "param-path" => {
                let settings = self.settings.lock().unwrap();
                settings.param_path.to_value()
            }
            "run" => {
                let settings = self.settings.lock().unwrap();
                settings.det.is_some().to_value()
            }
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for GstFastestDet {}

impl ElementImpl for GstFastestDet {
    // there's no set_metadata() in Class now
    // That part is moved to ElementImpl::metadata()
    // The tutorial is outdated
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "FastestDet Rust",
                "Filter/Effect/Converter/Video",
                "Run FastestDet object detection model with opencv and ncnn",
                "Crosstyan <crosstyan@outlook.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv/gstopencvutils.cpp#L116
    // https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html?gi-language=c
    // copy and paste from tutorial
    // src and sink should only support BGR
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field(
                    "format",
                    gst::List::new([gst_video::VideoFormat::Bgr.to_str()]),
                )
                .field("width", gst::IntRange::new(0, i32::MAX))
                .field("height", gst::IntRange::new(0, i32::MAX))
                .field(
                    "framerate",
                    gst::FractionRange::new(
                        gst::Fraction::new(0, 1),
                        gst::Fraction::new(i32::MAX, 1),
                    ),
                )
                .build();
            // The src pad template must be named "src" for basetransform
            // and specific a pad that is always there
            let src_pad_template = gst::PadTemplate::new(
                "src",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            // On the sink pad, we can accept BGR of any
            // width/height and with any framerate
            let caps = gst::Caps::builder("video/x-raw")
                .field("format", gst_video::VideoFormat::Bgr.to_str())
                .field("width", gst::IntRange::new(0, i32::MAX))
                .field("height", gst::IntRange::new(0, i32::MAX))
                .field(
                    "framerate",
                    gst::FractionRange::new(
                        gst::Fraction::new(0, 1),
                        gst::Fraction::new(i32::MAX, 1),
                    ),
                )
                .build();
            // The sink pad template must be named "sink" for basetransform
            // and specific a pad that is always there
            let sink_pad_template = gst::PadTemplate::new(
                "sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &caps,
            )
            .unwrap();

            let text_caps = gst::Caps::builder("application/x-json").build();
            let sink_text_pad_template = gst::PadTemplate::new(
                "text_sink",
                gst::PadDirection::Sink,
                gst::PadPresence::Always,
                &text_caps,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template, sink_text_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}

// Implementation of gst_base::BaseTransform virtual methods
// https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html?gi-language=c#processing
impl BaseTransformImpl for GstFastestDet {
    // If the always_in_place flag is set, non-writable buffers will be copied and
    // passed to the transform_ip function, otherwise a new buffer will be created
    // and the transform function called.
    // https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=c#modifications-inplace-input-buffer-and-output-buffer-are-the-same-thing
    const MODE: gst_base::subclass::BaseTransformMode = gst_base::subclass::BaseTransformMode::Both;
    // https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=c#passthrough-mode
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;
}

impl VideoFilterImpl for GstFastestDet {
    // Does the actual transformation of the input buffer to the output buffer
    // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv/gstopencvvideofilter.cpp#L152
    // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv/gstopencvvideofilter.cpp#L173
    // https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html?gi-language=c#GstVideoFilterClass::transform_frame
    // https://gstreamer.freedesktop.org/documentation/application-development/advanced/pipeline-manipulation.html?gi-language=c
    fn transform_frame(
        &self,
        element: &Self::Type,
        in_frame: &gst_video::VideoFrameRef<&gst::BufferRef>,
        out_frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Keep the various metadata we need for working with the video frames in
        // local variables. This saves some typing below.
        let cols = in_frame.width() as i32;
        let rows = in_frame.height() as i32;
        let in_data = in_frame.plane_data(0).unwrap();
        let in_format = in_frame.format();
        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_data = out_frame.plane_data_mut(0).unwrap();
        out_data.copy_from_slice(in_data);
        let out_ptr = out_data.as_mut_ptr() as *mut c_void;

        // ~I guess out_frame has the same data as in_frame.~
        // NO. out_frame is empty. have to copy the content manually
        // let size = rows * in_stride as i32;
        // unsafe {
        //     std::ptr::copy_nonoverlapping(in_ptr, out_ptr, size as usize);
        // }

        assert_eq!(in_format, gst_video::VideoFormat::Bgr);
        assert_eq!(out_format, gst_video::VideoFormat::Bgr);

        let settings = self.settings.lock().unwrap();
        let det = settings.det.as_ref();

        // TODO: find pad when starting
        let sinks = element.sink_pads();
        let text_sink = sinks.iter().find(|pad| pad.name() == "text_sink");
        // if name == "application/x-json" {
        //     let buffer = sink.get_sticky_event("text", 0).unwrap().get_structure().unwrap().get_value("text").unwrap().get::<String>().unwrap();
        //     let json: serde_json::Value = serde_json::from_str(&buffer).unwrap();
        //     let mut det = settings.det.lock().unwrap();
        //     det.update(json);
        // }
        match det {
            Some(det) => {
                let mut out_mat = match unsafe {
                    CvMat::new_rows_cols_with_data(
                        rows,
                        cols,
                        opencv::core::CV_8UC3,
                        out_ptr,
                        out_stride,
                    )
                } {
                    Ok(mat) => mat,
                    Err(_) => return Err(gst::FlowError::Error),
                };
                let input = det.preprocess(&out_mat).unwrap();
                let (w, h) = (out_mat.cols(), out_mat.rows());
                let targets = det.detect(&input, (w, h), 0.65).unwrap();
                let nms_targets = nms_handle(&targets, 0.45);
                if let Some(sink) = text_sink {
                    gst_debug!(CAT, obj: element, "text sink found");
                    let serialized = serde_json::to_string(&nms_targets).unwrap();
                    let buffer = gst::Buffer::from_mut_slice(serialized.into_bytes());
                    sink.push(buffer).unwrap();
                }
                let res = paint_targets(&mut out_mat, &nms_targets, det.classes());
                match res {
                    Ok(_) => return Ok(gst::FlowSuccess::Ok),
                    Err(_) => return Err(gst::FlowError::Error),
                }
            }
            None => {
                return Ok(gst::FlowSuccess::Ok);
            }
        }
        // Ok(gst::FlowSuccess::Ok)
    }

    fn transform_frame_ip(
        &self,
        _element: &Self::Type,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let cols = frame.width() as i32;
        let rows = frame.height() as i32;
        let stride = frame.plane_stride()[0] as usize;
        let data = frame.plane_data_mut(0).unwrap();
        // Okay.I know what I'm doing. I'm sure.
        let ptr = data.as_mut_ptr() as *mut c_void;

        // copy and paste from transform_frame
        let settings = self.settings.lock().unwrap();
        let det = settings.det.as_ref();
        match det {
            Some(det) => {
                let mut out_mat = match unsafe {
                    CvMat::new_rows_cols_with_data(rows, cols, opencv::core::CV_8UC3, ptr, stride)
                } {
                    Ok(mat) => mat,
                    Err(_) => return Err(gst::FlowError::Error),
                };
                let input = det.preprocess(&out_mat).unwrap();
                let (w, h) = (out_mat.cols(), out_mat.rows());
                let targets = det.detect(&input, (w, h), 0.65).unwrap();
                let nms_targets = nms_handle(&targets, 0.45);
                let res = paint_targets(&mut out_mat, &nms_targets, det.classes());
                match res {
                    Ok(_) => return Ok(gst::FlowSuccess::Ok),
                    Err(_) => return Err(gst::FlowError::Error),
                }
            }
            None => {
                return Ok(gst::FlowSuccess::Ok);
            }
        }
    }
}
