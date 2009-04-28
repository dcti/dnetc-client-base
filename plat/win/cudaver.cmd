@echo off

rem Exit with ERRORLEVEL containing the CUDA version level.
rem For example:  2010 = CUDA 2.1
rem               2000 = CUDA 2.0

if "%CUDA_INC_PATH%"=="" goto notfound
if not exist %CUDA_INC_PATH%\cuda.h goto notfound

set cudaversion=
for /f "tokens=3 usebackq" %%i in (`findstr CUDA_VERSION %CUDA_INC_PATH%\cuda.h`) do (
  set cudaversion=%%i
)
if "%cudaversion%"=="" goto notfound
echo cudaversion=%cudaversion%
exit %cudaversion%

:notfound
exit 0
