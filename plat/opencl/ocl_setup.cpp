/*
* Copyright distributed.net 2009 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
*/

#include <stdlib.h>
#include "cputypes.h"
#include "logstuff.h"
#include "ocl_info.h"
#include "ocl_setup.h"
#include "ocl_context.h"
#include "../rc5-72/opencl/ocl_common.h"

static cl_int numDevices = -12345;
static ocl_context_t *ocl_context;

int getOpenCLDeviceCount(void)
{
  return numDevices;
}

ocl_context_t *ocl_get_context(int device)
{
  return device < numDevices ? &ocl_context[device] : NULL;
}

// To debug on CPU...
// #undef  CL_DEVICE_TYPE_GPU
// #define CL_DEVICE_TYPE_GPU CL_DEVICE_TYPE_ALL

int InitializeOpenCL(void)
{
  if (numDevices != -12345)
    return numDevices;

  numDevices = -1;  /* assume detection failure for now */

  cl_uint devicesDetected = 0;
  cl_uint numPlatforms;
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS)
  {
    Log("Error obtaining number of platforms (clGetPlatformIDs/1)\n");
    ocl_diagnose(status, NULL, NULL);  // decode error code only
  }
  if (status == CL_SUCCESS)
  {
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;

    if (numPlatforms != 0)
    {
      // Allocate enough space for each platform
      platforms = (cl_platform_id *) malloc(numPlatforms * sizeof(cl_platform_id));

      // Fill in platforms with clGetPlatformIDs()
      status = clGetPlatformIDs(numPlatforms, platforms, NULL);
      if (status != CL_SUCCESS)
      {
        Log("Error obtaining list of platforms (clGetPlatformIDs/2)\n");
        ocl_diagnose(status, NULL, NULL);  // decode error code only
      }
      else
      {
        // Use clGetDeviceIDs() to retrieve the number of devices present
        for (cl_uint plat = 0; plat < numPlatforms; plat++)
        {
          cl_uint devcnt;

          status = clGetDeviceIDs(platforms[plat], CL_DEVICE_TYPE_GPU, 0, NULL, &devcnt);
          if (status == CL_DEVICE_NOT_FOUND)  // Special case. No GPU devices but other may exist
          {
            status = CL_SUCCESS;
            devcnt = 0;
          }
          if (status != CL_SUCCESS)
          {
            Log("Error obtaining number of devices on platform %u (clGetDeviceIDs/1)\n", plat);
            ocl_diagnose(status, NULL, NULL);  // decode error code only
            break;
          }
          devicesDetected += devcnt;
        }
      }
    }

    if (status == CL_SUCCESS && devicesDetected != 0)
    {
      // Allocate enough space for each device
      devices = (cl_device_id*) malloc(devicesDetected * sizeof(cl_device_id));

      // Allocate and zero space for ocl_context
      ocl_context = (ocl_context_t*) calloc(devicesDetected, sizeof(ocl_context_t));

      // Fill in devices with clGetDeviceIDs()
      cl_uint offset = 0;
      for (cl_uint plat = 0; plat < numPlatforms; plat++)
      {
        cl_uint devcnt;

        status = clGetDeviceIDs(platforms[plat], CL_DEVICE_TYPE_GPU, devicesDetected - offset, devices + offset, &devcnt);
        if (status == CL_DEVICE_NOT_FOUND)  // Special case. No GPU devices but other may exist
        {
          status = CL_SUCCESS;
          devcnt = 0;
        }
        if (status != CL_SUCCESS)
        {
          Log("Error obtaining list of devices on platform %u (clGetDeviceIDs/2)\n", plat);
          ocl_diagnose(status, NULL, NULL);  // decode error code only
          break;
        }

        // Fill non-zero context fields for each device
        for (cl_uint u = 0; u < devcnt; u++, offset++)
        {
          ocl_context_t *cont = &ocl_context[offset];

          cont->active            = true;
          cont->coreID            = CORE_NONE;
          cont->platformID        = platforms[plat];
          cont->deviceID          = devices[offset];
          cont->firstOnPlatform   = (u == 0);
          cont->clientDeviceNo    = offset;
          cont->runSize           = 65536;
          cont->runSizeMultiplier = 64;
          cont->maxWorkSize       = 2048 * 2048;
        }
      }
    }

    if (status == CL_SUCCESS)
    {
      // Everything is done. Apply configuration.
      numDevices = devicesDetected;
    }

    // Don't need them anymore
    if (devices)
      free(devices);
    if (platforms)
      free(platforms);
  }

  return numDevices;
} 

void OCLReinitializeDevice(ocl_context_t *cont)
{
  if (cont->coreID == CORE_NONE)
    return;
  //Log("Reinializing device %u\n", device);
  cont->coreID = CORE_NONE;

  //Log("Releasing kernel\n");
  if (cont->kernel)
  {
    clReleaseKernel(cont->kernel);
    cont->kernel = NULL;
  }
  
  //Log("Releasing program\n");
  if (cont->program)
  {
    clReleaseProgram(cont->program);
    cont->program = NULL;
  }

  //Log("Releasing CQ\n");
  if (cont->cmdQueue)
  {
    clReleaseCommandQueue(cont->cmdQueue);
    cont->cmdQueue = NULL;
  }

  //Log("Releasing out ptr\n");
  if (cont->out_ptr)
  {
    free(cont->out_ptr);
    cont->out_ptr = NULL;
  }

  //Log("Releasing const ptr\n");
  if (cont->const_ptr)
  {
    free(cont->const_ptr);
    cont->const_ptr = NULL;
  }

  //Log("Releasing const buffer\n");
  if (cont->const_buffer)
  {
    clReleaseMemObject(cont->const_buffer);
    cont->const_buffer = NULL;
  }

  //Log("Releasing out buffer buffer\n");
  if (cont->out_buffer)
  {
    clReleaseMemObject(cont->out_buffer);
    cont->out_buffer = NULL;
  }

  //Log("Releasing context\n");
  if (cont->clcontext)
  {
    clReleaseContext(cont->clcontext);
    cont->clcontext = NULL;
  }

  cont->runSize = 65536;
  cont->maxWorkSize = 2048 * 2048;
  //Log("Reinit OK\n");
}
