extern crate winres;

fn main() {
    let mut manifest = winres::WindowsResource::new();
    manifest.set_icon("res/umaai-sm.ico");
    manifest.compile().unwrap();
}
