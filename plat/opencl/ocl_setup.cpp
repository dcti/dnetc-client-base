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

/*
 * Initialize...() must return:
 *
 *  < 0 - on error (fatal error during initialization/detection, client will shutdown)
 * >= 0 - on success, no matter how many devices were found (even 0).
 *
 * getNumDevices() must return:
 *
 * < 0 - on error (or if previous Initialize...() failed)
 * = 0 - no errors but no devices were found
 * > 0 - number of detected GPUs
 *
 * Although it's expected that Initialize...() will be always called before
 * any getDeviceCount(),  getDeviceCount() must return error and do not touch
 * GPU hardware if Initialize...() wasn't called or was called but failed.
 */

static cl_int numDevices = -12345;
static ocl_context_t *ocl_context;

int getOpenCLDeviceCount(void)
{
  return numDevices;
}

ocl_context_t *ocl_get_context(int device)
{
  if (device >= 0 && device < numDevices)
    return &ocl_context[device];

  Log("INTERNAL ERROR: bad OpenCL device index %d (detected %d)!\n", device, numDevices);
  return NULL;
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

        if (offset >= devicesDetected)   /* Avoid call with bufferSize=0 for last platform without GPU devices */
          break;

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

          /* Assume it working for now */
          cont->active            = true;
          cont->coreID            = CORE_NONE;
          cont->platformID        = platforms[plat];
          cont->deviceID          = devices[offset];
          cont->firstOnPlatform   = (u == 0);
          cont->clientDeviceNo    = offset;
          cont->runSize           = 65536;
          cont->runSizeMultiplier = 64;
          cont->maxWorkSize       = 2048 * 2048;

          /* Sanity check: size_t must be same width for both client and device */
          cl_uint devbits;
          status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_ADDRESS_BITS, sizeof(devbits), &devbits, NULL);
          if (ocl_diagnose(status, "clGetDeviceInfo(CL_DEVICE_ADDRESS_BITS)", cont) != CL_SUCCESS)
            cont->active = false;
          else if (devbits != sizeof(size_t) * 8)
          {
            Log("Error: Bitness of device %u (%u) does not match CPU (%u)!\n", offset, devbits, sizeof(size_t) * 8);
            cont->active = false;
          }
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
  //Log("Reinializing device %u\n", device);
  cont->coreID = CORE_NONE;

  //Log("Releasing kernel\n");
  if (cont->kernel)
  {
    ocl_diagnose( clReleaseKernel(cont->kernel), "clReleaseKernel", cont );
    cont->kernel = NULL;
  }
  
  //Log("Releasing program\n");
  if (cont->program)
  {
    ocl_diagnose( clReleaseProgram(cont->program), "clReleaseProgram", cont );
    cont->program = NULL;
  }

  //Log("Releasing CQ\n");
  if (cont->cmdQueue)
  {
    ocl_diagnose( clReleaseCommandQueue(cont->cmdQueue), "clReleaseCommandQueue", cont );
    cont->cmdQueue = NULL;
  }

  //Log("Releasing const buffer\n");
  if (cont->const_buffer)
  {
    ocl_diagnose( clReleaseMemObject(cont->const_buffer), "clReleaseMemObject(const_buffer)", cont );
    cont->const_buffer = NULL;
  }

  //Log("Releasing out buffer buffer\n");
  if (cont->out_buffer)
  {
    ocl_diagnose( clReleaseMemObject(cont->out_buffer),  "clReleaseMemObject(out_buffer)", cont );
    cont->out_buffer = NULL;
  }

  //Log("Releasing context\n");
  if (cont->clcontext)
  {
    ocl_diagnose( clReleaseContext(cont->clcontext), "clReleaseContext", cont );
    cont->clcontext = NULL;
  }

  cont->runSize = 65536;
  cont->maxWorkSize = 2048 * 2048;
  //Log("Reinit OK\n");
}
