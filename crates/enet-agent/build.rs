// Embed the app icon + version info into the Windows exe.
fn main() {
    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set("ProductName", "BMW ENET Client");
        res.set("FileDescription", "BMW ENET laptop Client (car side)");
        res.set_icon("../../assets/icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=winres failed: {e}");
        }
    }
}
