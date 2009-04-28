@echo off

rem Exit with ERRORLEVEL containing the major version number reported by the Microsoft CL compiler.
rem This is useful for doing version-dependent behavior inside Makefiles.
rem An exit value of 0 indicates no CL was found in the PATH.

for %%i in (cl.exe) do if "%%~$PATH:i"=="" goto notfound

set msvcversion=
for /f "eol=. tokens=6-8 usebackq " %%i in (`cmd /c "cl.exe /D 2>&1 | findstr Version"`) do (
  if "%%i"=="Version" set msvcversion=%%j
  if "%%j"=="Version" set msvcversion=%%k
)
if "%msvcversion%"=="" goto notfound
exit %msvcversion:~0,2%

:notfound
exit 0
