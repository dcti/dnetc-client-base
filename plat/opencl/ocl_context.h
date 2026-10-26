/*
* Copyright distributed.net 2012-2016 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
*/

#ifndef OCL_CONTEXT_INCLUDED
#define OCL_CONTEXT_INCLUDED

enum {
  CORE_NONE = 0,
  CORE_REF,
  CORE_1PIPE,
  CORE_2PIPE,
  CORE_4PIPE,
  CORE_CL_TOTAL
};

typedef struct {
  u32               coreID;
  bool              active;
  bool              firstOnPlatform; // new platform started here (for logs)
  cl_platform_id    platformID;      // in OpenCL subsystem
  cl_device_id      deviceID;        // in OpenCL subsystem
  int               clientDeviceNo;  // client GPU index (for logs)
  cl_context        clcontext;
  cl_command_queue  cmdQueue; 
  cl_mem            const_buffer; 
  cl_mem            out_buffer; 
  cl_program	    program; 
  cl_kernel         kernel; 
  u32               runSize;
  u32               runSizeMultiplier;
  u32               maxWorkSize;
} ocl_context_t;

ocl_context_t *ocl_get_context(int device);

#endif //OCL_CONTEXT_INCLUDED
