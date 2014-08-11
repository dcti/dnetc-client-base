/*
* Copyright distributed.net 2009 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
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

cl_uint numPlatforms = 0;
cl_platform_id *platforms = NULL;
cl_int numDevices = -2;
cl_device_id *devices = NULL;

int getOpenCLDeviceCount(void)
{
  if (numDevices == -2)
    InitializeOpenCL();
  return numDevices;
}

u32 getOpenCLDeviceFreq(unsigned device)
{
  if (getOpenCLDeviceCount() > (int)device)
  {
    cl_uint clockrate;
    if(clGetDeviceInfo(devices[device], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockrate), &clockrate, NULL)==CL_SUCCESS)
		return clockrate;
  }
  return 0;
}

static unsigned GetDeviceID(unsigned vendor_id, cl_char *device_name, cl_uint cunits, unsigned device)
{
  size_t globalWorkSize[1];
  cl_uint *outPtr;

	//Run kernel with predefined constants to get device ID
//  cl_context context = NULL;
  cl_command_queue cmdQueue = NULL;
  cl_mem out_buffer = NULL;
//  cl_program program = NULL;
//  cl_kernel kernel = NULL;
  cl_int status;
  cl_uint id=0;

  if(ocl_context[device].coreID != CORE_NONE)
	  OCLReinitializeDevice(device);

  // Create a context and associate it with the device
  ocl_context[device].clcontext = clCreateContext(NULL, 1, &devices[device], NULL, NULL, &status);
  if(status != CL_SUCCESS)
	  return 0;
  
  cmdQueue = clCreateCommandQueue(ocl_context[device].clcontext, devices[device], 0, &status);
  if(status != CL_SUCCESS)
	  goto finished;

  out_buffer = clCreateBuffer(ocl_context[device].clcontext, CL_MEM_ALLOC_HOST_PTR, 4, NULL, &status);
  if(status != CL_SUCCESS)
	  goto finished;

  if(!BuildCLProgram(device, deviceid_src, "deviceID"))
	  goto finished;

   status |= clSetKernelArg(ocl_context[device].kernel, 0, sizeof(cl_mem), &out_buffer);
   if(status != CL_SUCCESS)
     goto finished;
	 
   globalWorkSize[0] = 1;
   status = clEnqueueNDRangeKernel(cmdQueue, ocl_context[device].kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
   if(status != CL_SUCCESS)
     goto finished;

  outPtr = NULL;
  outPtr = (cl_uint*) clEnqueueMapBuffer(cmdQueue, out_buffer, CL_TRUE, CL_MAP_READ, 0, 4, 0, NULL, NULL, &status);
  if(status == CL_SUCCESS)
  {
	  id = outPtr[0];
      clEnqueueUnmapMemObject(cmdQueue, out_buffer, outPtr, 0, NULL, NULL);
  }

finished:
  if(cmdQueue)
    clReleaseCommandQueue(cmdQueue);
  if(out_buffer)
    clReleaseMemObject(out_buffer);
  OCLReinitializeDevice(device);
  return id;
}

long getOpenCLRawProcessorID(const char **cpuname, unsigned device)
{
  static cl_char device_name[256+130] = {0};
  strcpy((char*)device_name, "Unknown");

  if(cpuname)
	*cpuname = (const char*)device_name;

  if (getOpenCLDeviceCount() > (int)device)
  {
    clGetDeviceInfo(devices[device], CL_DEVICE_NAME, sizeof(device_name)-130, device_name, NULL);

	//retrieve card info, if available
	u32 off = strlen((const char*)device_name);
	device_name[off++]=' '; device_name[off++]='\0';
#ifdef CL_DEVICE_BOARD_NAME_AMD
	if(clGetDeviceInfo(devices[device], CL_DEVICE_BOARD_NAME_AMD, sizeof(device_name)-off, &device_name[off], NULL) == CL_SUCCESS)
	{
	  device_name[off-1]='(';
	  u32 off2 = strlen((const char*)device_name);
	  device_name[off2] = ')';
	  device_name[off2+1] = '\0';
	}
#endif
	
	cl_uint vendor_id=0;
	clGetDeviceInfo(devices[device], CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);

	cl_uint cunits=0;
	clGetDeviceInfo(devices[device], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cunits), &cunits, NULL);
	
	return GetDeviceID(vendor_id, device_name, cunits, device);
  }

  return -1;

}

void OpenCLPrintExtendedGpuInfo(void)
{
  int i;

  if (getOpenCLDeviceCount() <= 0)
  {
    LogRaw("No supported devices found\n");
    return;
  }

  //Print platform info
  LogRaw("\nPlatform info:\n");
  LogRaw("--------------\n");
  cl_char str[80];
  cl_int status;
  status = clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, sizeof(str), (void *)str, NULL);
  if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "Platform Name", str);

  status = clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, sizeof(str), (void *)str, NULL);
  if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "Platform Vendor", str);

  status = clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, sizeof(str), (void *)str, NULL);
  if(status == CL_SUCCESS)  LogRaw("%30s: %s\n", "Platform Version", str);

  cl_char *str2;
  size_t sz;

  status = clGetPlatformInfo(platforms[0], CL_PLATFORM_EXTENSIONS, 0, NULL, &sz);
  if(sz)
  {
    str2 = (cl_char*)malloc(sz+1);
    status = clGetPlatformInfo(platforms[0], CL_PLATFORM_EXTENSIONS, sz+1, (void *)str2, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "Platform extensions",str2);
    free(str2);
  }

  for (i = 0; i < getOpenCLDeviceCount(); i++)
  {
    cl_char device_name[1024] = {0};
    
    LogRaw("\nDevice #%u:\n",i);
    LogRaw("------------\n");

    cl_device_type type;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if(status == CL_SUCCESS)
    {
      LogRaw("%30s: ", "Type");
      if( type & CL_DEVICE_TYPE_CPU )
        LogRaw("CPU\n");                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
      else
        if( type & CL_DEVICE_TYPE_GPU )
          LogRaw("GPU\n");
        else
          LogRaw("UNKNOWN\n");
    }

    status = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "Name",device_name);

    cl_uint clockrate;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockrate), &clockrate, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "Max clockrate", clockrate);

    cl_uint cunits;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cunits), &cunits, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "Max compute units", cunits);

    cl_ulong gmemcache;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(gmemcache), &gmemcache, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "Global memory cache size", gmemcache);

    cl_device_mem_cache_type ct;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(ct), &ct, NULL);
    if(status == CL_SUCCESS) 
    {
      LogRaw("%30s: ", "Global memory cache type");
      switch(ct)
      {
        case CL_NONE:
          LogRaw("NONE\n");
          break;
        case CL_READ_ONLY_CACHE:
          LogRaw("Read Only\n");
          break;
        case CL_READ_WRITE_CACHE:
          LogRaw("Read/Write\n");
          break;
        default:
          LogRaw("Not sure\n");
      }
    }

    cl_bool um;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(um), &um, NULL);
    if(status == CL_SUCCESS) 
    {
      LogRaw("%30s: ", "Unified memory subsystem");
      if(um)
        LogRaw("Yes\n");
      else
        LogRaw("No\n");
    }

    status = clGetDeviceInfo(devices[i], CL_DEVICE_IMAGE_SUPPORT, sizeof(um), &um, NULL);
    if(status == CL_SUCCESS) 
    {
      LogRaw("%30s: ", "Image support");
      if(um)
        LogRaw("Yes\n");
      else
        LogRaw("No\n");
    }

    status = clGetDeviceInfo(devices[i], CL_DEVICE_LOCAL_MEM_SIZE, sizeof(gmemcache), &gmemcache, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "Local memory size", gmemcache);

    size_t mwgs;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(mwgs), &mwgs, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "Max workgroup size", mwgs);

    cl_uint nvw;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(nvw), &nvw, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "native vector width (int)", nvw);
    status = clGetDeviceInfo(devices[i], CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(nvw), &nvw, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "native vector width (float)", nvw);

    status = clGetDeviceInfo(devices[i], CL_DEVICE_OPENCL_C_VERSION, sizeof(device_name), device_name, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "OpenCL C version",device_name);

    size_t ptres;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(ptres), &ptres, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %u\n", "Device timer resolution (ns)",ptres);

    status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, sizeof(device_name), device_name, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "Device vendor",device_name);

    cl_uint vendor_id;
    status = clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: 0x%x\n", "Device vendor id",vendor_id);

    status = clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, sizeof(device_name), device_name, NULL);
    if(status == CL_SUCCESS) LogRaw("%30s: %s\n", "Driver version",device_name);

    //TODO: device extensions
  }

}
