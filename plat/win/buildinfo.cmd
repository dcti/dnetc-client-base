@echo off
rem Copyright distributed.net 2026 - All Rights Reserved
rem For use in distributed.net projects only.
rem Any other distribution or use of this source violates copyright.
rem
rem Writes the compiler flags that tell common\cliident.cpp which source
rem revisions the client is built from and when it was built, as a cl
rem response file. Usage: buildinfo.cmd <response file>
rem This is the makefile.vc equivalent of common/buildinfo.sh; see there
rem for what the flags mean. Missing tools (git, PowerShell) just leave
rem the corresponding flags out.

setlocal
set OUT=%~f1
set FLAGS=
cd /d "%~dp0..\.."

call :repo . PUB
if exist "..\private\buffpriv.cpp" call :repo .. PRIV

set NOW=
for /f %%t in ('powershell -NoProfile -Command "[int64](([datetime]::UtcNow - [datetime]'1970-01-01').TotalSeconds)" 2^>nul') do set NOW=%%t
if defined NOW set FLAGS=%FLAGS% /DBUILDINFO_TIME=%NOW%

> "%OUT%" echo.%FLAGS%
exit /b 0

rem :repo <directory> <PUB or PRIV>
:repo
pushd "%~1"
rem Use git only if this directory is the top of its own work tree,
rem so that an enclosing repo (the private one) can't answer for it.
git rev-parse --git-dir >nul 2>&1 || goto repo_done
set CDUP=
for /f "delims=" %%c in ('git rev-parse --show-cdup 2^>nul') do set CDUP=%%c
if defined CDUP goto repo_done
set HASH=
for /f %%h in ('git rev-parse --verify -q HEAD 2^>nul') do set HASH=%%h
if not defined HASH goto repo_done
rem Untracked files don't count, as with "git describe --dirty".
git update-index -q --refresh >nul 2>&1
set DIRTY=1
git diff-index --quiet --ignore-submodules HEAD -- && set DIRTY=0
set FLAGS=%FLAGS% /DBUILDINFO_%2_HASH=0x%HASH:~0,8%UL /DBUILDINFO_%2_DIRTY=%DIRTY%
set WHEN=
for /f %%w in ('git log -1 "--format=%%ct" HEAD 2^>nul') do set WHEN=%%w
if defined WHEN set FLAGS=%FLAGS% /DBUILDINFO_%2_TIME=%WHEN%
:repo_done
popd
exit /b 0
