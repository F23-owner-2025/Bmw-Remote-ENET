@echo off
setlocal
set INSTALLDIR=%ProgramFiles%\BMW-ENET-Gateway
mkdir "%INSTALLDIR%" 2>nul
mkdir "%INSTALLDIR%\config" 2>nul
mkdir "%INSTALLDIR%\logs" 2>nul

copy /Y enet-gateway.exe "%INSTALLDIR%\" >nul
copy /Y enet-gui.exe "%INSTALLDIR%\" >nul
copy /Y ..\config\gateway.toml "%INSTALLDIR%\config\" >nul

echo Checking for Wintun...
where /Q wintun.dll
if errorlevel 1 (
  echo WARNING: wintun.dll not found on PATH. Install Wintun manually.
)

echo Creating firewall rule...
netsh advfirewall firewall add rule name="BMW ENET Tunnel" dir=in action=allow protocol=UDP localport=47900 remoteip=Localsubnet profile=private

echo Installing service...
sc.exe stop BmwEnetGateway >nul 2>&1
sc.exe delete BmwEnetGateway >nul 2>&1
sc.exe create BmwEnetGateway binPath= "\"%INSTALLDIR%\enet-gateway.exe\" --config \"%INSTALLDIR%\config\gateway.toml\"" start= auto
sc.exe description BmwEnetGateway "BMW ENET L2 tunnel gateway for F-Series diagnostics"
sc.exe start BmwEnetGateway

echo Creating desktop shortcut...
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop')+'\BMW ENET Gateway.lnk'); $s.TargetPath='%INSTALLDIR%\enet-gui.exe'; $s.WorkingDirectory='%INSTALLDIR%'; $s.Save()"

echo Done. Install directory: %INSTALLDIR%
endlocal
