@echo off
setlocal
set INSTALLDIR=%ProgramFiles%\BMW-ENET-Agent
mkdir "%INSTALLDIR%" 2>nul
mkdir "%INSTALLDIR%\config" 2>nul
mkdir "%INSTALLDIR%\logs" 2>nul

copy /Y enet-agent.exe "%INSTALLDIR%\" >nul
copy /Y ..\config\agent.toml "%INSTALLDIR%\config\" >nul

echo IMPORTANT: Edit %INSTALLDIR%\config\agent.toml and set peer_addr to your desktop IP.
echo Npcap must be installed for production ENET capture.

sc.exe stop BmwEnetAgent >nul 2>&1
sc.exe delete BmwEnetAgent >nul 2>&1
sc.exe create BmwEnetAgent binPath= "\"%INSTALLDIR%\enet-agent.exe\" --config \"%INSTALLDIR%\config\agent.toml\"" start= auto
sc.exe description BmwEnetAgent "BMW ENET laptop tunnel agent"
sc.exe start BmwEnetAgent

echo Done.
endlocal
