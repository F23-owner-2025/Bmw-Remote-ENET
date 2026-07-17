//! Windows link flags so Host/Client can start without Npcap installed.
//!
//! The `pcap` crate links `wpcap.dll` at build time. Without delay-load, Windows
//! refuses to start the process when Npcap is missing — even for `--simulate`
//! (which never calls into pcap). CI runners only have the Npcap SDK, not the
//! runtime DLL, so smoke tests were dying before the Host API could bind.

fn main() {
    // Only MSVC supports /DELAYLOAD. GNU cross-builds still need wpcap at link
    // time for the import lib; they are compile-only in this repo.
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("windows") && target.contains("msvc") {
        println!("cargo:rustc-link-arg=/DELAYLOAD:wpcap.dll");
        println!("cargo:rustc-link-arg=delayimp.lib");
    }
}
