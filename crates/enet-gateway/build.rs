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

    // Allow the Host to start without Npcap installed (CI smoke / --simulate).
    // See crates/enet-tunnel/build.rs for the rationale.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") && target.contains("msvc") {
        println!("cargo:rustc-link-arg=/DELAYLOAD:wpcap.dll");
        println!("cargo:rustc-link-arg=delayimp.lib");
    }
}
