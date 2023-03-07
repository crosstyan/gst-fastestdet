use gst::glib;
// use gst::glib::subclass::prelude::*;
use super::fastest_det::{nms_handle, FastestDet, TargetBox};
use gst::prelude::*;
use gst::subclass::prelude::*;
// use gst::{gst_debug, gst_error, gst_info, gst_log, gst_trace, gst_warning};
use gst::{debug, error_msg, info, trace, warning};
use gst_base::subclass::prelude::*;
use gst_video::subclass::prelude::*;
use image::RgbImage;
use rusttype::{Font, Scale};
use serde_derive::{Deserialize, Serialize};

use std::ffi::c_void;

use std::i32;
use std::ops::Index;
use std::sync::Mutex;

use once_cell::sync::Lazy;

static FONT:[u8; include_bytes!("DejaVuSans.ttf").len()] = *include_bytes!("DejaVuSans.ttf");

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
    is_paint: bool,
    det: Option<FastestDet>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model_path: DEFAULT_MODEL_PATH.to_string(),
            param_path: DEFAULT_PARAM_PATH.to_string(),
            classes_path: DEFAULT_CLASSES_PATH.to_string(),
            is_paint: false,
            det: None,
        }
    }
}

pub fn paint_targets(paint_img:&mut RgbImage, targets: &Vec<TargetBox>, classes:&Vec<String>) -> Result<(), anyhow::Error> {
    for target in targets.iter(){
        let (x1, y1, x2, y2) = (target.x1 as u32, target.y1 as u32, target.x2 as u32, target.y2 as u32);
        let rect = imageproc::rect::Rect::at(x1 as i32, y1 as i32).of_size(x2-x1, y2-y1).try_into()?;
        let color = image::Rgb([0, 128, 128]);
        imageproc::drawing::draw_hollow_rect_mut(paint_img, rect, color);
        let class_name = classes.index(target.class as usize);
        let font = Font::try_from_bytes(&FONT).ok_or(anyhow::anyhow!("font error"))?;
        let height = 12.4;
        let scale = Scale{x:height, y:height};
        imageproc::drawing::draw_text_mut(paint_img, color, x1 as i32, y1 as i32, scale, &font, class_name);
    }
    Ok(())
}

#[derive(Default)]
pub struct GstFastestDet {
    settings: Mutex<Settings>,
    /// `text_pad` here should be an output, which outputs the json of the detected objects.
    /// `src` is the output port of the bin.
    /// `sink` is the input port of the bin.
    /// See also `TargetBox` in `fastest_det.rs`.
    /// See also [why call the output port of a element to "src pad" in gstreamer?](https://superuser.com/questions/1400417/why-call-the-output-port-of-a-element-to-src-pad-in-gstreamer)
    text_pad: Option<gst::Pad>,
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
    /// side effect/not pure
    /// the function will paint the targets on the image
    /// and push the targets to the text src
    pub fn detect_push(
        &self,
        det: &FastestDet,
        mat: &mut RgbImage,
        is_paint: bool,
    ) -> Result<(), anyhow::Error> {
        let text_src = self.text_pad.as_ref();
        let input = det.preprocess(&mat).unwrap();
        let (w, h) = (mat.width() as i32, mat.height() as i32);
        let targets = det.detect(&input, (w, h), 0.65).unwrap();
        let nms_targets = nms_handle(&targets, 0.45);
        if let Some(pad) = text_src {
            let serialized = serde_json::to_string(&nms_targets)?;
            let buffer = gst::Buffer::from_mut_slice(serialized.into_bytes());
            // ignore the error
            // if there is no downstream element, the error will be FlowError
            // But we use probe to get the buffer, so no downstream element is ok.
            // No error should be raised.
            let _ = pad.push(buffer);
        }
        if is_paint {
            paint_targets(mat, &nms_targets, &det.classes())
        } else {
            Ok(())
        }
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

    fn with_class(klass: &Self::Class) -> Self {
        let templ = klass.pad_template("text_pad").unwrap();
        let text_pad = gst::Pad::from_template(&templ, Some("text_pad"));
        Self {
            settings: Mutex::new(Settings::default()),
            text_pad: Some(text_pad),
        }
    }
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
                glib::ParamSpecBoolean::builder("is-paint")
                    .nick("Is paint")
                    .blurb("if true, the recognition result will be painted on the image")
                    .default_value(false)
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
    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.settings.lock().unwrap();
                // TODO: proper error detection
                settings.model_path = value.get().unwrap();
                settings.model_path = settings.model_path.trim().to_string();
                if settings.model_path.find(",").is_some() {
                    warning!(CAT, "path should not contain `,`");
                }
                info!(CAT, "Set model path to {}", settings.model_path);
            }
            "config-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.classes_path = value.get().unwrap();
                settings.classes_path = settings.classes_path.trim().to_string();
                info!(CAT, "Set config path to {}", settings.classes_path);
            }
            "param-path" => {
                let mut settings = self.settings.lock().unwrap();
                settings.param_path = value.get().unwrap();
                settings.param_path = settings.param_path.trim().to_string();
                info!(CAT, "Set param path to {}", settings.param_path);
            }
            "is-paint" => {
                let mut settings = self.settings.lock().unwrap();
                settings.is_paint = value.get().unwrap();
                info!(CAT, "Set is_paint to {}", settings.is_paint);
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
                            info!(CAT, "model loaded");
                        }
                        Err(e) => {
                            info!(CAT, "Failed to create det: {}", e);
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

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
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
            "is-paint" => {
                let settings = self.settings.lock().unwrap();
                settings.is_paint.to_value()
            }
            "run" => {
                let settings = self.settings.lock().unwrap();
                settings.det.is_some().to_value()
            }
            _ => unimplemented!(),
        }
    }
    fn constructed(&self) {
        self.parent_constructed();
        // https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/blob/main/text/json/src/jsongstenc/imp.rs#L218
        let obj = self.obj();
        let pad = self.text_pad.as_ref().unwrap();
        obj.add_pad(pad).unwrap();
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
            let src_text_pad_template = gst::PadTemplate::new(
                "text_pad",
                gst::PadDirection::Src,
                gst::PadPresence::Always,
                &text_caps,
            )
            .unwrap();

            vec![src_pad_template, sink_pad_template, src_text_pad_template]
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
        in_frame: &gst_video::VideoFrameRef<&gst::BufferRef>,
        out_frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Keep the various metadata we need for working with the video frames in
        // local variables. This saves some typing below.
        let cols = in_frame.width();
        let rows = in_frame.height();
        let in_data = in_frame.plane_data(0).unwrap();
        let in_format = in_frame.format();
        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_data = out_frame.plane_data_mut(0).unwrap();
        out_data.copy_from_slice(in_data);

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

        match det {
            Some(det) => {
                let mut out_mat = image::RgbImage::from_raw(cols, rows, out_data.to_vec());
                match out_mat {
                    Some(ref mut out_mat) => {
                        match self.detect_push(&det, out_mat, settings.is_paint) {
                            Ok(_) => return Ok(gst::FlowSuccess::Ok),
                            Err(_) => return Err(gst::FlowError::Error),
                        };
                    }
                    None => {
                        return Err(gst::FlowError::Error);
                    }
                };
            }
            None => {
                return Ok(gst::FlowSuccess::Ok);
            }
        }
        // Ok(gst::FlowSuccess::Ok)
    }

    fn transform_frame_ip(
        &self,
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
                unimplemented!();
            }
            None => {
                return Ok(gst::FlowSuccess::Ok);
            }
        }
    }
}
