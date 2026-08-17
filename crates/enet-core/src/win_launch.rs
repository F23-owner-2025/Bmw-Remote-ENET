//! Windows `cmd /C start` helpers.
//!
//! `cmd /C start http://127.0.0.1:47903/` is parsed as a UNC path (`//` → `\\`)
//! and shows **Windows cannot find '\\'**. Passing an empty window title
//! (`start "" url`) avoids that. Verbatim `\\?\` exe paths also confuse `start`.

use std::path::Path;

/// `cmd /C start` arguments: empty title, then the URL or file.
///
/// Index 2 must stay `""` so `http://…` is not treated as a UNC path.
pub fn cmd_start_args(target: &str) -> [&str; 4] {
    ["/C", "start", "", target]
}

/// Path string safe to pass to `cmd /C start "" "path"`.
pub fn cmd_start_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    s.strip_prefix(r"\\?\")
        .unwrap_or(&s)
        .replace('/', "\\")
}

/// Open an `http://` / `https://` URL in the default browser.
///
/// On Windows this always uses `start "" <url>` so localhost URLs do not
/// surface the `Windows cannot find '\\'` dialog.
pub fn open_http_url(url: &str) -> std::io::Result<()> {
    if url.trim().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "refusing to open an empty URL",
        ));
    }
    #[cfg(windows)]
    {
        std::process::Command::new("cmd")
            .args(cmd_start_args(url))
            .spawn()?;
        return Ok(());
    }
    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open").arg(url).spawn()?;
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open").arg(url).spawn()?;
        return Ok(());
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "open_http_url is not supported on this OS",
        ))
    }
}

/// Spawn `enet-gui.exe` from `dir` talking to `api`. Detached so the parent can exit.
pub fn spawn_enet_gui(dir: &Path, api: &str) -> std::io::Result<()> {
    let gui = dir.join("enet-gui.exe");
    if !gui.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "enet-gui.exe not found",
        ));
    }
    let mut cmd = std::process::Command::new(&gui);
    cmd.arg("--api").arg(api).current_dir(dir);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const DETACHED_PROCESS: u32 = 0x0000_0008;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP);
    }
    cmd.spawn()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn start_args_include_empty_title() {
        let args = cmd_start_args("http://127.0.0.1:47903/");
        assert_eq!(args[0], "/C");
        assert_eq!(args[1], "start");
        assert_eq!(args[2], "");
        assert_eq!(args[3], "http://127.0.0.1:47903/");
    }

    #[test]
    fn start_path_strips_windows_verbatim_prefix() {
        let p = PathBuf::from(r"\\?\C:\BMW-ENET\Client\enet-agent.exe");
        assert_eq!(
            cmd_start_path(&p),
            r"C:\BMW-ENET\Client\enet-agent.exe"
        );
    }

    #[test]
    fn start_path_leaves_normal_windows_path() {
        let p = PathBuf::from(r"C:\BMW-ENET\Client\enet-gui.exe");
        assert_eq!(cmd_start_path(&p), r"C:\BMW-ENET\Client\enet-gui.exe");
    }

    #[test]
    fn empty_url_is_rejected() {
        let err = open_http_url("").unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }
}
