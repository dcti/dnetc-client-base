/*
* Copyright distributed.net 2009 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: cuda_setup.cpp,v 1.5 2009/04/29 09:30:20 andreasb Exp $
*/

#include "cuda_setup.h"
#include "cputypes.h"
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"

#include <cuda.h>
#include <cuda_runtime.h>

// returns 0 on success
// i.e. a supported GPU + driver version + CUDA version was found
int InitializeCUDA()
{
  static int retval = -123;

  if (retval == -123) {
    retval = -1;

    #if ((CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64))
    const char nvcudaFileName[] = "nvcuda.dll";
    char verPath[] = "\\";

    HMODULE module = LoadLibraryEx(&nvcudaFileName[0], NULL, 0);
    if(!module)
    {
      retval = CUDA_SETUP_MISSING_NVCUDA_DLL;
      Log("Unable to locate CUDA module handle");
    }
    else
    {
      TCHAR pszFile[MAX_PATH + 1];
      DWORD dwLen = GetModuleFileName(module, &pszFile[0], MAX_PATH + 1);
      if(dwLen <= 0 || dwLen >= MAX_PATH)
      {
        retval = CUDA_SETUP_INVALID_NVCUDA_PATH;
        Log("Unable to retrieve CUDA module file name");
      }
      else
      {
        DWORD dwHandle;
        DWORD dwLen = GetFileVersionInfoSize(pszFile, &dwHandle);

        if (dwLen > 0)
        {
          LPBYTE pBuffer = new BYTE[dwLen];
          if (pBuffer)
          {
            if (GetFileVersionInfo(&pszFile[0], dwHandle, dwLen, pBuffer))
            {
              UINT uLen;
              VS_FIXEDFILEINFO *lpFfi;
              if(VerQueryValue( pBuffer , &verPath[0] , (LPVOID *)&lpFfi , &uLen ))
              {
                DWORD dwLeftMost = HIWORD(lpFfi->dwFileVersionMS);
                DWORD dwSecondLeft = LOWORD(lpFfi->dwFileVersionMS);
                DWORD dwSecondRight = HIWORD(lpFfi->dwFileVersionLS);
                DWORD dwRightMost = LOWORD(lpFfi->dwFileVersionLS);

                char buffer[64];
                snprintf(buffer, sizeof(buffer), "nvcuda.dll Version: %d.%d.%d.%d" , dwLeftMost, dwSecondLeft, dwSecondRight, dwRightMost);

                Log( buffer );

                #if CUDA_VERSION > 2000
                if (!(dwSecondRight > 11 || (dwSecondRight == 11 && dwRightMost >= 8120)))
                {
                  retval = CUDA_SETUP_INVALID_DRIVER_REVISION;
                  Log("This CUDA 2.1 client requires driver revision 181.20 or later");
                }
                #else
                if (!(dwSecondRight > 11 || (dwSecondRight == 11 && dwRightMost >= 7735)))
                {
                  retval = CUDA_SETUP_INVALID_DRIVER_REVISION;
                  Log("This CUDA 2.0 client requires driver revision 177.35 or later");
                }
                #endif
                else
                {
                  retval = 0;
                }
              }
            }
            else
            {
              retval = CUDA_SETUP_NO_FILE_VERSION;
              Log("Unable to read CUDA file version");
            }

            delete [] pBuffer;
          }

        }
        else
        {
          retval = CUDA_SETUP_NO_FILE_VERSION;
          Log("Unable to read CUDA file version size");
        }
      }
    }

    if(module)
    {
      FreeLibrary(module);
    }
    #else

    retval = 0;

    #endif

    // try creating CUDA stream and event, may fail on library/driver mismatch

    if (retval == 0) {
      cudaStream_t stream = -1;
      if (cudaStreamCreate(&stream) != cudaSuccess) {
        retval = CUDA_SETUP_STREAM_FAILURE;
      } else if (cudaStreamDestroy(stream) != cudaSuccess) {
        retval = CUDA_SETUP_STREAM_FAILURE;
      }
      if (retval)
        Log("Unable to create CUDA stream");
    }

    if (retval == 0) {
      cudaEvent_t event = -1;
      if (cudaEventCreate(&event) != cudaSuccess) {
        retval = CUDA_SETUP_EVENT_FAILURE;
      } else if (cudaEventDestroy(event) != cudaSuccess) {
        retval = CUDA_SETUP_EVENT_FAILURE;
      }
      if (retval)
        Log("Unable to create CUDA event");
    }
  }

  return retval;
}
