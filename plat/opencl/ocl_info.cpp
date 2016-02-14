/*
* Copyright distributed.net 2009-2016 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: ocl_info.cpp 2016/02/04 19:08:25 zebe Exp $
*/

#include "ocl_info.h"
#include "ocl_setup.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#if (CLIENT_OS == OS_WIN64) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
#include <CL/cl_ext.h>
#endif

#include "logstuff.h"
#include "deviceid.cpp"
#include "../rc5-72/opencl/ocl_common.h"

u32 getOpenCLDeviceFreq(int device)
{
  ocl_context_t *cont = ocl_get_context(device);

  if (cont)
  {
    cl_uint clockrate;
    if (clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockrate), &clockrate, NULL) == CL_SUCCESS)
      return clockrate;
  }

  return 0;
}

/*
 * Run kernel with predefined constants to get device ID
 */
static unsigned GetDeviceID(/* unsigned vendor_id, cl_char *device_name, cl_uint cunits, */ ocl_context_t *cont)
{
  size_t globalWorkSize[1];
  cl_int status;
  cl_uint id = 0;
  cl_uint *outPtr;

  // This code does not set coreID, so device must be implicitly reinited at start
  // and cleaned up at end.
  OCLReinitializeDevice(cont);

  // Create a context and associate it with the device
  cont->clcontext = clCreateContext(NULL, 1, &cont->deviceID, NULL, NULL, &status);
  if (status != CL_SUCCESS)
    goto finished;
  
  cont->cmdQueue = clCreateCommandQueue(cont->clcontext, cont->deviceID, 0, &status);
  if (status != CL_SUCCESS)
    goto finished;

  cont->out_buffer = clCreateBuffer(cont->clcontext, CL_MEM_ALLOC_HOST_PTR, 4, NULL, &status);
  if (status != CL_SUCCESS)
    goto finished;

  if (!BuildCLProgram(cont, deviceid_src, "deviceID"))
    goto finished;

  status = clSetKernelArg(cont->kernel, 0, sizeof(cl_mem), &cont->out_buffer);
  if (status != CL_SUCCESS)
    goto finished;
	 
  globalWorkSize[0] = 1;
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if (status != CL_SUCCESS)
     goto finished;

  outPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, cont->out_buffer, CL_TRUE, CL_MAP_READ, 0, 4, 0, NULL, NULL, &status);
  if (status == CL_SUCCESS)
  {
    id = outPtr[0];
    clEnqueueUnmapMemObject(cont->cmdQueue, cont->out_buffer, outPtr, 0, NULL, NULL);
  }

finished:
  OCLReinitializeDevice(cont);
  return id;
}

long getOpenCLRawProcessorID(int device, const char **cpuname)
{
  static cl_char device_name[256+130];
  strcpy((char*)device_name, "Unknown");

  if (cpuname)
    *cpuname = (const char*)device_name;

  ocl_context_t *cont = ocl_get_context(device);
  if (cont)
  {
    clGetDeviceInfo(cont->deviceID, CL_DEVICE_NAME, sizeof(device_name)-130, device_name, NULL);

    //retrieve card info, if available
    u32 off = strlen((const char*)device_name);
    device_name[off++]=' '; device_name[off++]='\0';
#ifdef CL_DEVICE_BOARD_NAME_AMD
    if (clGetDeviceInfo(cont->deviceID, CL_DEVICE_BOARD_NAME_AMD, sizeof(device_name)-off, &device_name[off], NULL) == CL_SUCCESS)
    {
      device_name[off-1]='(';
      u32 off2 = strlen((const char*)device_name);
      device_name[off2] = ')';
      device_name[off2+1] = '\0';
    }
#endif
// ??? Never used
/*
    cl_uint vendor_id=0;
    clGetDeviceInfo(cont->deviceID, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);

    cl_uint cunits=0;
    clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cunits), &cunits, NULL);
*/
    return GetDeviceID(/* vendor_id, device_name, cunits, */ cont);
  }

  return -1;
}

void OpenCLPrintExtendedGpuInfo(int device)
{
  const char *data;
  cl_int status;

  ocl_context_t *cont = ocl_get_context(device);
  if (cont == NULL)
    return;

  if (cont->firstOnPlatform)
  {
    //Print platform info once
    LogRaw("\nPlatform info:\n");
    LogRaw("--------------\n");
    cl_char str[80];
    status = clGetPlatformInfo(cont->platformID, CL_PLATFORM_NAME, sizeof(str), (void *)str, NULL);
    if (status == CL_SUCCESS) LogRaw("%30s: %s\n", "Platform Name", str);

    status = clGetPlatformInfo(cont->platformID, CL_PLATFORM_VENDOR, sizeof(str), (void *)str, NULL);
    if (status == CL_SUCCESS) LogRaw("%30s: %s\n", "Platform Vendor", str);

    status = clGetPlatformInfo(cont->platformID, CL_PLATFORM_VERSION, sizeof(str), (void *)str, NULL);
    if (status == CL_SUCCESS)  LogRaw("%30s: %s\n", "Platform Version", str);

    cl_char *str2;
    size_t sz;
    status = clGetPlatformInfo(cont->platformID, CL_PLATFORM_EXTENSIONS, 0, NULL, &sz);
    if (sz)
    {
      str2 = (cl_char*)malloc(sz+1);
      if (str2)
      {
        status = clGetPlatformInfo(cont->platformID, CL_PLATFORM_EXTENSIONS, sz+1, (void *)str2, NULL);
        if (status == CL_SUCCESS) LogRaw("%30s: %s\n", "Platform extensions", str2);
        free(str2);
      }
    }
    /* Split platform and device info */
    LogRaw("\nDevice info:\n");
    LogRaw("--------------\n");
  }

  cl_char device_name[1024] = {0};
  cl_device_type type;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
  if (status == CL_SUCCESS)
  {
    if ( type & CL_DEVICE_TYPE_CPU )
      data = "CPU";
    else
    {
      if ( type & CL_DEVICE_TYPE_GPU )
        data = "GPU";
      else
        data = "UNKNOWN";
    }
    LogRaw("%30s: %s\n", "Type", data);
  }

  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
  if (status == CL_SUCCESS) LogRaw("%30s: %s\n", "Name",device_name);

  cl_uint clockrate;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockrate), &clockrate, NULL);
  if (status == CL_SUCCESS) LogRaw("%30s: %u\n", "Max clockrate", clockrate);

  cl_uint cunits;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cunits), &cunits, NULL);
  if (status == CL_SUCCESS) LogRaw("%30s: %u\n", "Max compute units", cunits);

  cl_ulong gmemcache;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(gmemcache), &gmemcache, NULL);
  if (status == CL_SUCCESS) LogRaw("%30s: %lu\n", "Global memory cache size", gmemcache);

  cl_device_mem_cache_type ct;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(ct), &ct, NULL);
  if (status == CL_SUCCESS)
  {
    switch(ct)
    {
      case CL_NONE:
        data = "NONE";
        break;
      case CL_READ_ONLY_CACHE:
        data = "Read Only";
        break;
      case CL_READ_WRITE_CACHE:
        data = "Read/Write";
        break;
      default:
        data = "Not sure";
    }
    LogRaw("%30s: %s\n", "Global memory cache type", data);
  }

  cl_bool um;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(um), &um, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %s\n", "Unified memory subsystem", (um ? "Yes" : "No"));

  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_IMAGE_SUPPORT, sizeof(um), &um, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %s\n", "Image support", (um ? "Yes" : "No"));

  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(gmemcache), &gmemcache, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %lu\n", "Local memory size", gmemcache);

  size_t mwgs;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwgs), &mwgs, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %lu\n", "Max workgroup size", (unsigned long)mwgs);

  cl_uint nvw;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(nvw), &nvw, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %u\n", "native vector width (int)", nvw);
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(nvw), &nvw, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %u\n", "native vector width (float)", nvw);

  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_OPENCL_C_VERSION, sizeof(device_name), device_name, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %s\n", "OpenCL C version",device_name);

  size_t ptres;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(ptres), &ptres, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %lu\n", "Device timer resolution (ns)", (unsigned long)ptres);

  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_VENDOR, sizeof(device_name), device_name, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %s\n", "Device vendor",device_name);

  cl_uint vendor_id;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: 0x%x\n", "Device vendor id",vendor_id);

  status = clGetDeviceInfo(cont->deviceID, CL_DRIVER_VERSION, sizeof(device_name), device_name, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %s\n", "Driver version",device_name);

  cl_uint devbits;
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_ADDRESS_BITS, sizeof(devbits), &devbits, NULL);
  if (status == CL_SUCCESS)
    LogRaw("%30s: %u%s\n", "Device address bits", devbits, (devbits == sizeof(size_t) * 8 ? "" : " - NOT MATCHED -"));

  //TODO: device extensions
}
