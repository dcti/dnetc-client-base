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

ocl_context_t *ocl_context;

int getNumDevices()
{
	return numDevices;
}

int InitializeOpenCL()
{

  numDevices = -1;
  cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
	
  if(!numPlatforms)
  {
    Log("No OpenCL platforms available!\n");
    return -1;
  }
  // Allocate enough space for each platform
  platforms = (cl_platform_id*)malloc(numPlatforms*sizeof(cl_platform_id));

  // Fill in platforms with clGetPlatformIDs()
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);

  // Use clGetDeviceIDs() to retrieve the number of 
  // devices present
  status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, (cl_uint*)&numDevices);
  if(numDevices)
  {
    // Allocate enough space for each device
    devices = (cl_device_id*)malloc(numDevices*sizeof(cl_device_id));

    // Fill in devices with clGetDeviceIDs()
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
  }

  //allocating space for ocl_context
  ocl_context = (ocl_context_t*)malloc(numDevices*sizeof(ocl_context_t));
  for(int i=0; i < numDevices; i++)
  {
    ocl_context[i].active = true;
    ocl_context[i].coreID = CORE_NONE;
    ocl_context[i].clcontext = NULL;
    ocl_context[i].deviceID = devices[i];
    ocl_context[i].cmdQueue = NULL;
    ocl_context[i].const_buffer = NULL;
    ocl_context[i].out_buffer = NULL;
    ocl_context[i].const_ptr = NULL;
    ocl_context[i].out_ptr = NULL;
    ocl_context[i].kernel = NULL;
    ocl_context[i].program = NULL;
    ocl_context[i].runSize = 65536;	
    ocl_context[i].runSizeMultiplier = 64;
    ocl_context[i].maxWorkSize = 2048 * 2048;
  }
  
  return numDevices;
} 

void OCLReinitializeDevice(int device)
{
  if(ocl_context[device].coreID == CORE_NONE)
	  return;
  //Log("Reinializing device %u\n", device);
  ocl_context[device].coreID = CORE_NONE;

  //Log("Releasing kernel\n");
  if(ocl_context[device].kernel)
  {
	  clReleaseKernel(ocl_context[device].kernel);
	  ocl_context[device].kernel = NULL;
  }
  
  //Log("Releasing program\n");
  if(ocl_context[device].program)
  {
	clReleaseProgram(ocl_context[device].program);
    ocl_context[device].program = NULL;
  }

  //Log("Releasing CQ\n");
  if(ocl_context[device].cmdQueue)
  {
	clReleaseCommandQueue(ocl_context[device].cmdQueue);
	ocl_context[device].cmdQueue = NULL;
  }

  //Log("Releasing out ptr\n");
  if(ocl_context[device].out_ptr)
  {
	  free(ocl_context[device].out_ptr);
	  ocl_context[device].out_ptr = NULL;
  }

  //Log("Releasing const ptr\n");
  if(ocl_context[device].const_ptr)
  {
	  free(ocl_context[device].const_ptr);
	  ocl_context[device].const_ptr = NULL;
  }

  //Log("Releasing const buffer\n");
  if(ocl_context[device].const_buffer)
  {
	  clReleaseMemObject(ocl_context[device].const_buffer);
	  ocl_context[device].const_buffer = NULL;
  }

  //Log("Releasing out buffer buffer\n");
  if(ocl_context[device].out_buffer)
  {
	  clReleaseMemObject(ocl_context[device].out_buffer);
	  ocl_context[device].out_buffer = NULL;
  }

  //Log("Releasing context\n");
  if(ocl_context[device].clcontext)
  {
	  clReleaseContext(ocl_context[device].clcontext);
	  ocl_context[device].clcontext = NULL;
  }

  ocl_context[device].runSize = 65536;	
  ocl_context[device].maxWorkSize = 2048 * 2048;
  //Log("Reinit OK\n");
}
