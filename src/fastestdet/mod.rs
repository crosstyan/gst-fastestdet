use gst::glib;
use gst_base::subclass::prelude::*;
use std::i32;
use std::sync::Mutex;
use once_cell::sync::Lazy;
use gst::prelude::*;
pub mod imp;
mod fastest_det;
mod utils;

// https://gitlab.freedesktop.org/gstreamer/gst-plugins-rs/-/blob/main/video/hsv/src/hsvdetector/imp.rs
// https://gitlab.freedesktop.org/gstreamer/gstreamer-rs/-/blob/main/gstreamer-video/src/auto/video_filter.rs
// https://gitlab.freedesktop.org/gstreamer/gstreamer-rs/-/blob/main/gstreamer-video/src/subclass/video_filter.rs

// https://gitlab.freedesktop.org/gstreamer/gstreamer/-/tree/main/subprojects/gst-plugins-bad/gst-libs/gst/opencv
// https://gitlab.freedesktop.org/gstreamer/gstreamer/-/tree/main/subprojects/gst-plugins-bad/ext/opencv

// https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html?gi-language=c
// https://gstreamer.freedesktop.org/documentation/video/gstvideofilter.html?gi-language=c

glib::wrapper! {
    pub struct GstFastestDet(ObjectSubclass<imp::GstFastestDet>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

// Registers the type for our element, and then registers in GStreamer under
// the name "fastestdetrs" for being able to instantiate it via e.g.
// gst::ElementFactory::make().
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "fastestdetrs",
        gst::Rank::None,
        GstFastestDet::static_type(),
    )
}
