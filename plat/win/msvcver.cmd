@echo off

rem Exit with ERRORLEVEL containing the major version number reported
rem by the Microsoft CL compiler.
rem This is useful for doing version-dependent behavior inside Makefiles.
rem An exit value of 0 indicates no CL was found in the PATH.
rem 
rem   cl 12.00.xxxx = Visual Studio 6 (VC6/VC98)
rem   cl 13.00.xxxx = Visual Studio.NET 2002 (VC7.0)
rem   cl 13.10.xxxx = Visual Studio.NET 2003 (VC7.1)
rem   cl 14.00.xxxx = Visual Studio 2005 (VC8)
rem   cl 15.00.xxxx = Visual Studio 2008 (VC9)

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
