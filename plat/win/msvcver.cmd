@echo off

rem Exit with ERRORLEVEL containing the major version number reported
rem by the Microsoft CL compiler.
rem This is useful for doing version-dependent behavior inside Makefiles.
rem An exit value of 0 indicates no CL was found in the PATH.
rem 
rem   12 = cl 12.00.xxxx = Visual Studio 6 (VC6/VC98)
rem   13 = cl 13.00.xxxx = Visual Studio.NET 2002 (VC7.0)
rem   13 = cl 13.10.xxxx = Visual Studio.NET 2003 (VC7.1)
rem   14 = cl 14.00.xxxx = Visual Studio 2005 (VC8)
rem   15 = cl 15.00.xxxx = Visual Studio 2008 (VC9)
rem   16 = cl 16.00.xxxx = Visual Studio 2010 (VC10)
rem   17 = cl 17.00.xxxx = Visual Studio 2012 (VC11)
rem   18 = cl 18.00.xxxx = Visual Studio 2013 (VC12)

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
