// Embed the app icon + version info into the Windows exe.
fn main() {
    #[cfg(windows)]
    {
        let mut res = winres::WindowsResource::new();
        res.set("ProductName", "BMW ENET Gateway");
        res.set("FileDescription", "BMW ENET desktop Host (ISTA bridge)");
        res.set_icon("../../assets/icon.ico");
        if let Err(e) = res.compile() {
            println!("cargo:warning=winres failed: {e}");
        }
    }
}
