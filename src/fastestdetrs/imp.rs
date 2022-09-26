use gst::glib;
// use gst::glib::subclass::prelude::*;
use gst::prelude::*;
use gst::subclass::prelude::*;
use gst::{gst_debug, gst_error, gst_info, gst_log, gst_trace, gst_warning};
use gst_base::subclass::prelude::*;
use gst_video::subclass::prelude::*;
use opencv::core::Mat as CvMat;
use opencv::core::*;
use opencv::prelude::*;

use std::ffi::c_void;

use std::i32;
use std::sync::Mutex;

use once_cell::sync::Lazy;

// VideoInfo is a struct that contains various fields like width/height,
// framerate and the video format and allows to conveniently with the
// properties of (raw) video formats. We have to store it inside a Mutex in our
// struct as this can (in theory) be accessed from multiple threads at
// the same time.

/// This module contains the private implementation details of our element
static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "fastestdetrs",
        gst::DebugColorFlags::empty(),
        Some("Rust FastestDet Element"),
    )
});

#[derive(Default)]
pub struct FastestDet {}

impl FastestDet {}

// This trait registers our type with the GObject object system and
// provides the entry points for creating a new instance and setting
// up the class data
#[glib::object_subclass]
impl ObjectSubclass for FastestDet {
    const NAME: &'static str = "FastestDetRs";
    type Type = super::FastestDet;
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
impl ObjectImpl for FastestDet {}

impl GstObjectImpl for FastestDet {}

// Implementation of gst::Element virtual methods
impl ElementImpl for FastestDet {
    // there's no set_metadata() in Class now
    // That part is moved to ElementImpl::metadata()
    // The tutorial is outdated
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "FastestDet",
                "Filter/Effect/Converter/Video",
                "Run FastestDet object detection model with opencv and ncnn",
                "Crosstyan <crosstyan@outlook.com>",
            )
        });

        Some(&*ELEMENT_METADATA)
    }

    // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv/gstopencvutils.cpp#L116
    // https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html?gi-language=c
    // copy and paste
    // src and sink should only support BGR
    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            // On the src pad, we can produce BGRx and GRAY8 of any
            // width/height and with any framerate
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

            vec![src_pad_template, sink_pad_template]
        });

        PAD_TEMPLATES.as_ref()
    }
}

// Implementation of gst_base::BaseTransform virtual methods
// https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html?gi-language=c#processing
impl BaseTransformImpl for FastestDet {
    // If the always_in_place flag is set, non-writable buffers will be copied and
    // passed to the transform_ip function, otherwise a new buffer will be created
    // and the transform function called.
    // https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=c#modifications-inplace-input-buffer-and-output-buffer-are-the-same-thing
    const MODE: gst_base::subclass::BaseTransformMode = gst_base::subclass::BaseTransformMode::Both;
    // https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=c#passthrough-mode
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    // Called for converting caps from one pad to another to account for any
    // changes in the media format this element is performing.
    //
    // Optional. Given the pad in this direction and the given caps, what caps
    // are allowed on the other pad in this element?  Possible formats on sink
    // and source pad implemented with custom transform_caps function. By
    // default uses *same format on sink and source*.
    // https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=c#GstBaseTransformClass::transform_caps
    // fn transform_caps(
    //     &self,
    //     element: &Self::Type,
    //     direction: gst::PadDirection,
    //     caps: &gst::Caps,
    //     filter: Option<&gst::Caps>,
    // ) -> Option<gst::Caps>
}

impl VideoFilterImpl for FastestDet {
    // Does the actual transformation of the input buffer to the output buffer
    // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv/gstopencvvideofilter.cpp#L152
    // https://gitlab.freedesktop.org/gstreamer/gstreamer/-/blob/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv/gstopencvvideofilter.cpp#L173
    // https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html?gi-language=c#GstVideoFilterClass::transform_frame
    fn transform_frame(
        &self,
        _element: &Self::Type,
        in_frame: &gst_video::VideoFrameRef<&gst::BufferRef>,
        out_frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        // Keep a local copy of the values of all our properties at this very moment. This
        // ensures that the mutex is never locked for long and the application wouldn't
        // have to block until this function returns when getting/setting property values
        // let settings = *self.settings.lock().unwrap();

        // Keep the various metadata we need for working with the video frames in
        // local variables. This saves some typing below.
        let cols = in_frame.width() as i32;
        let rows = in_frame.height() as i32;
        let in_stride = in_frame.plane_stride()[0] as usize;
        let in_data = in_frame.plane_data(0).unwrap();
        // Okay.I know what I'm doing. I'm sure. and I promise I won't mutate the data.
        let in_ptr = in_data.as_ptr() as *const u8 as *mut c_void;
        let in_format = in_frame.format();
        let out_stride = out_frame.plane_stride()[0] as usize;
        let out_format = out_frame.format();
        let out_data = out_frame.plane_data_mut(0).unwrap();
        let out_ptr = out_data.as_mut_ptr() as *mut c_void;

        assert_eq!(in_format, gst_video::VideoFormat::Bgr);
        assert_eq!(out_format, gst_video::VideoFormat::Bgr);

        let _in_mat = match unsafe {
            CvMat::new_rows_cols_with_data(rows, cols, opencv::core::CV_8UC3, in_ptr, in_stride)
        } {
            Ok(mat) => mat,
            Err(_) => return Err(gst::FlowError::Error),
        };

        let mut out_mat = match unsafe {
            CvMat::new_rows_cols_with_data(rows, cols, opencv::core::CV_8UC3, out_ptr, out_stride)
        } {
            Ok(mat) => mat,
            Err(_) => return Err(gst::FlowError::Error),
        };
        // I guess out_frame has the same data as in_frame.

        let blue = Scalar::new(255.0, 255.0, 0.0, 0.0);
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let thickness = 2;
        let line_type = opencv::imgproc::LINE_8;
        let shift = 0;
        let text = "TEST!";
        // draw something on frame for testing
        let res = opencv::imgproc::put_text(
            &mut out_mat,
            text,
            Point::new(cols / 2, rows / 2),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.75,
            green,
            thickness,
            line_type,
            false,
        );
        if let Err(_) = res {
            return Err(gst::FlowError::Error);
        }

        Ok(gst::FlowSuccess::Ok)
    }

    fn transform_frame_ip(
        &self,
        _element: &Self::Type,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
    ) -> Result<gst::FlowSuccess, gst::FlowError> {
        let cols = frame.width() as i32;
        let rows = frame.height() as i32;
        let stride = frame.plane_stride()[0] as usize;
        let data = frame.plane_data(0).unwrap();
        // Okay.I know what I'm doing. I'm sure. and I promise I won't mutate the data.
        let ptr = data.as_ptr() as *const u8 as *mut c_void;

        let mut out_mat = match unsafe {
            CvMat::new_rows_cols_with_data(rows, cols, opencv::core::CV_8UC3, ptr, stride)
        } {
            Ok(mat) => mat,
            Err(_) => return Err(gst::FlowError::Error),
        };

        let blue = Scalar::new(255.0, 255.0, 0.0, 0.0);
        let green = Scalar::new(0.0, 255.0, 0.0, 0.0);
        let thickness = 2;
        let line_type = opencv::imgproc::LINE_8;
        let shift = 0;
        let text = "TEST!";
        // draw something on frame for testing
        let res = opencv::imgproc::put_text(
            &mut out_mat,
            text,
            Point::new(cols / 2, rows / 2),
            opencv::imgproc::FONT_HERSHEY_SIMPLEX,
            0.75,
            blue,
            thickness,
            line_type,
            false,
        );
        if let Err(_) = res {
            return Err(gst::FlowError::Error);
        }

        Ok(gst::FlowSuccess::Ok)
    }
}
