use gst::glib;
pub mod fastestdet;

// Plugin entry point that should register all elements provided by this plugin,
// and everything else that this plugin might provide (e.g. typefinders or device providers).
fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    fastestdet::register(plugin)?;
    Ok(())
}

// Here's the catch
// https://docs.rs/gst-plugin/0.1.1/gst_plugin/macro.plugin_define.html
// the first parameter of `gst::plugin_define` macro is the name of the plugin
// which should NOT be the same as the name of the crate and NOT start with
// `lib` or `gst` But lib.name (the name item under lib section) in `Cargo.toml`
// SHOULD start with "gst" (still without `lib`) The generated library will be
// called `lib<lib.name>.so`
gst::plugin_define!(
    fastestdet,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);

