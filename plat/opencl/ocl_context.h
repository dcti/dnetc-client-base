/*
* Copyright distributed.net 2012 - All Rights Reserved
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
  cl_context        clcontext;
  cl_device_id      deviceID;
  cl_command_queue  cmdQueue; 
  cl_mem            const_buffer; 
  cl_mem            out_buffer; 
  cl_uint           *const_ptr; 
  cl_uint           *out_ptr; 
  cl_program	    program; 
  cl_kernel         kernel; 
  u32               runSize;
  u32               runSizeMultiplier;
  u32               maxWorkSize;
}ocl_context_t;

extern ocl_context_t *ocl_context;

#endif //OCL_CONTEXT_INCLUDED