use gst::glib;
use rand::rngs::StdRng;
// use gst::glib::subclass::prelude::*;
use super::common::{nms_handle, paint_targets, ImageModel, RgbBuffer, TargetBox};
use super::fastest_det::FastestDet;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{debug, info, warning};
use gst_base::subclass::prelude::*;
use gst_video::subclass::prelude::*;
use once_cell::sync::Lazy;
use rand::distributions::{Distribution, Uniform};
use rand::{Rng, SeedableRng};
use serde_derive::Deserialize;
use std::i32;
use std::ops::Not;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

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
    rng: StdRng,
    last_state: Vec<TargetBox>,
    dropout: f32,
    det: Option<FastestDet>,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model_path: DEFAULT_MODEL_PATH.to_string(),
            param_path: DEFAULT_PARAM_PATH.to_string(),
            classes_path: DEFAULT_CLASSES_PATH.to_string(),
            is_paint: false,
            rng: StdRng::from_entropy(),
            last_state: vec![],
            dropout: 0.0,
            det: None,
        }
    }
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

    pub fn send_to_text_pad(&self, targets: &Vec<TargetBox>) -> Result<(), anyhow::Error> {
        let text_src = self.text_pad.as_ref();
        if let Some(pad) = text_src {
            let serialized = serde_json::to_string(targets)?;
            let buffer = gst::Buffer::from_mut_slice(serialized.into_bytes());
            // ignore the error
            // if there is no downstream element, the error will be FlowError
            // But we use probe to get the buffer, so no downstream element is ok.
            // No error should be raised.
            let _ = pad.push(buffer);
        }
        Ok(())
    }

    /// kinda pure
    ///
    /// would return targets filtered by nms
    pub fn detect<
        T: Deref<Target = [u8]> + DerefMut<Target = [u8]> + AsRef<[u8]>,
        M: ImageModel,
    >(
        &self,
        det: &mut M,
        mat: &mut RgbBuffer<T>,
    ) -> Result<Vec<TargetBox>, anyhow::Error> {
        let input = det.preprocess(&mat)?;
        let (w, h) = (mat.width() as i32, mat.height() as i32);
        let targets = det.detect(&input, (w, h), 0.65)?;
        let nms_targets = nms_handle(&targets, 0.45);
        Ok(nms_targets)
    }

    fn transform_impl(&self, cols:u32, rows:u32, data:&mut [u8]) -> Result<gst::FlowSuccess, gst::FlowError>{
        let mut settings = self.settings.lock().unwrap();
        let is_paint = settings.is_paint;
        let distribution = Uniform::from(0..100);
        assert!(settings.dropout >= 0.0 && settings.dropout < 1.0);
        let p = distribution.sample(&mut settings.rng) as f32 / 100.0;
        let is_update = if p <= settings.dropout { false } else { true };
        let last_state = settings.last_state.clone();

        if is_update {
            let det = settings.det.as_mut();
            match det {
                Some(det) => {
                    // Don't use `to_vec` since it will create new buffer by copy
                    let mut out_mat = image::ImageBuffer::from_raw(cols, rows, data);
                    match out_mat {
                        Some(ref mut out_mat) => {
                            if is_update {
                                match self.detect(det, out_mat) {
                                    Ok(targets) => {
                                        if is_paint {
                                            if targets.is_empty().not() {
                                                debug!(CAT, "painting targets:{:?}", targets);
                                            }
                                            let _ = paint_targets(out_mat, &targets, &det.labels());
                                        }
                                        return Ok(gst::FlowSuccess::Ok);
                                    }
                                    Err(_) => return Err(gst::FlowError::Error),
                                };
                            } else {
                                if is_paint {
                                    let _ = paint_targets(out_mat, &last_state, &det.labels());
                                }
                            }
                        }
                        None => {
                            return Err(gst::FlowError::Error);
                        }
                    };
                }
                None => {}
            }
        }
        Ok(gst::FlowSuccess::Ok)
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
                    .default_value(true)
                    .flags(glib::ParamFlags::READWRITE)
                    .build(),
                glib::ParamSpecFloat::builder("dropout")
                    .nick("Dropout rate")
                    .blurb("dropout rate")
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
            "dropout" => {
                let mut settings = self.settings.lock().unwrap();
                settings.dropout = value.get().unwrap();
                info!(CAT, "Set dropout to {}", settings.dropout);
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
            "dropout" => {
                let settings = self.settings.lock().unwrap();
                settings.dropout.to_value()
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
    // src and sink should only support Rgb (image crate)
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst::Caps::builder("video/x-raw")
                .field(
                    "format",
                    gst::List::new([gst_video::VideoFormat::Rgb.to_str()]),
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
                .field("format", gst_video::VideoFormat::Rgb.to_str())
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
            // The sink pad template must be named "sink" for base transform
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
        let out_data = out_frame.plane_data_mut(0).unwrap();
        out_data.copy_from_slice(in_data);
        self.transform_impl(cols, rows, out_data)
    }


    fn transform_frame_ip(
        &self,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let cols = frame.width();
        let rows = frame.height();
        // modify the buffer in place
        let data = frame.plane_data_mut(0).unwrap();
        self.transform_impl(cols, rows, data)
    }
}
