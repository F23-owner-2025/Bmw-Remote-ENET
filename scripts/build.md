# Build scripts

## Linux / CI

```bash
cargo test --workspace --exclude enet-gui
cargo build --release -p enet-agent -p enet-gateway -p enet-sim
cargo run -p enet-sim --release -- lab --seconds 5 --flaps
```

## Windows

```powershell
cargo build --release
copy target\release\enet-*.exe installer\
cd installer
.\install-gateway.bat
```

Cross-compile from Linux to Windows (optional):

```bash
rustup target add x86_64-pc-windows-gnu
# requires mingw toolchain
cargo build --release --target x86_64-pc-windows-gnu -p enet-agent -p enet-gateway
```
