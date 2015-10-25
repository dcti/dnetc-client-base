/*
* Copyright distributed.net 2012 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id: 
*/

#include "ocl_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rc5-ref.cpp"

#define CONST_SIZE (sizeof(cl_uint)*16)
#define OUT_SIZE (sizeof(cl_uint)*64)

static bool init_rc5_72_ocl_ref(ocl_context_t *cont)
{
  if(!cont->active)
  {
    Log("Device %u is not supported\n", cont->clientDeviceNo);
    return false;
  }
  
  if(cont->coreID!=CORE_REF)
    OCLReinitializeDevice(cont);

  cl_int status;
  // Create a context and associate it with the device
  cont->clcontext = clCreateContext(NULL, 1, &cont->deviceID, NULL, NULL, &status);
  if(ocl_diagnose(status, "creating OCL context", cont) != CL_SUCCESS)
	  return false;
  
  //Create a command queue
  cont->cmdQueue = clCreateCommandQueue(cont->clcontext, cont->deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
  if(ocl_diagnose(status, "creating command queue", cont) != CL_SUCCESS)
	  return false;

  //Create device buffers
  cont->const_buffer = clCreateBuffer(cont->clcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, CONST_SIZE, NULL, &status); //CL_MEM_COPY_HOST_WRITE_ONLY ??
  if(ocl_diagnose(status, "creating constants buffer", cont) != CL_SUCCESS)
	  return false;

  cont->out_buffer = clCreateBuffer(cont->clcontext, CL_MEM_ALLOC_HOST_PTR, OUT_SIZE, NULL, &status);
  if(ocl_diagnose(status, "creating output buffer", cont) != CL_SUCCESS)
	  return false;

  if(!BuildCLProgram(cont, ocl_rc572_ref_src, "ocl_rc572_ref"))
	  return false;
  //////////////////////////////////
  /*
	FILE *f;
    cl_char *programSource;

	f=fopen("rc5-ref.cl","rb");
	if(f==NULL) {
		Log("Couldn't load 'rc5-ref.cl'");
		return false;
	}

	fseek (f , 0 , SEEK_END);
	unsigned lSize = ftell (f)+1;

	if(lSize>1000000) { 
		fclose(f);
		Log("Error in 'rc5-ref.cl'");
		return false;
	}

	programSource=(cl_char*)malloc(lSize);
	rewind(f);
	fread(programSource,lSize-1,1,f);
	programSource[lSize-1]=0; 

	fclose(f);
    cont->program = clCreateProgramWithSource(cont->clcontext, 1, (const char**)&programSource, NULL, &status);
    free(programSource);
  //////////////////////////////////
    
  // Build (compile) the program for the devices
  status |= clBuildProgram(cont->program, 1, &cont->deviceID, NULL, NULL, NULL);
  if(ocl_diagnose(status, "building cl program", cont) != CL_SUCCESS)
	return false;

  cont->kernel = clCreateKernel(cont->program, "ocl_rc572_ref", &status);
  if(ocl_diagnose(status, "building kernel", cont) != CL_SUCCESS)
	return false;
  */
  //Get a performance hint
  size_t prefm;
  status = clGetKernelWorkGroupInfo(cont->kernel, cont->deviceID, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
	  sizeof(prefm), &prefm, NULL);
  if(status==CL_SUCCESS)
  {
	  size_t cus;
	  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
	  if(status == CL_SUCCESS)
		cont->runSizeMultiplier = prefm * cus;
  }
  
  /*size_t workitem_size[3];
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &prefm, NULL);
  if(status==CL_SUCCESS)
  {
    cont->maxWorkSize = workitem_size[0] * workitem_size[1] * workitem_size[2];
    Log("max worksize = %u\n", cont->maxWorkSize);
  }*/
  cont->coreID=CORE_REF;
  return true;
}

static bool ClearOutBuffer(const cl_mem buffer, ocl_context_t *cont)
{
  cl_uint *outPtr = NULL;
  cl_int status;

  outPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, buffer, CL_TRUE, CL_MAP_WRITE, 0, 4, 0, NULL, NULL, &status);
  if(ocl_diagnose(status, "mapping output buffer", cont) != CL_SUCCESS)
	return false;
   outPtr[0]=0;

  status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
  if(ocl_diagnose(status, "unmapping output buffer", cont) != CL_SUCCESS)
	return false;
  return true;
}

static bool FillConstantBuffer(const cl_mem buffer, RC5_72UnitWork *rc5_72unitwork, cl_uint keys, ocl_context_t *cont)
{
  cl_uint *constPtr = NULL;
  cl_int status;

  constPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, buffer, CL_TRUE, CL_MAP_WRITE, 0, CONST_SIZE, 0, NULL, NULL, &status);
  if(ocl_diagnose(status, "mapping constants buffer", cont) != CL_SUCCESS)
	return false;

  //key_hi,key_mid,key_lo,granularity
  constPtr[0]=rc5_72unitwork->L0.hi;
  constPtr[1]=rc5_72unitwork->L0.mid;
  constPtr[2]=rc5_72unitwork->L0.lo;
  constPtr[3]=keys;

  //plain_lo,plain_hi,cypher_lo,cypher_hi
  constPtr[4]=rc5_72unitwork->plain.lo;
  constPtr[5]=rc5_72unitwork->plain.hi;
  constPtr[6]=rc5_72unitwork->cypher.lo;
  constPtr[7]=rc5_72unitwork->cypher.hi;
  
  status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, constPtr, 0, NULL, NULL);
  if(ocl_diagnose(status, "unmapping constants buffer", cont) != CL_SUCCESS)
	return false;
  return true;
}

static s32 ReadResults(const cl_mem buffer, u32 *CMC, u32 *iters_done, ocl_context_t *cont)
{
  cl_uint *outPtr = NULL;
  cl_int status;
  cl_uint found;

  *CMC = 0; 
  *iters_done = 0x7fffffff;
  outPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, buffer, CL_TRUE, CL_MAP_READ|CL_MAP_WRITE, 0, OUT_SIZE, 0, NULL, NULL, &status);
  if(ocl_diagnose(status, "mapping output buffer", cont) != CL_SUCCESS)
	return -1;

  found = outPtr[0];
  outPtr[0] = 0;
  if(found)
  {
    u32 fullmatchkeyidx = 0xffffffff;
    if(found>=(OUT_SIZE/sizeof(cl_uint)))
    {
      Log("Internal error reading kernel output\n");
      clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
      return -1;
    }
    for(u32 idx=1; idx<=found; idx++)
    {
      if(outPtr[idx]&0x80000000)
      {
        fullmatchkeyidx = outPtr[idx]&0x7fffffff;
	break;
      }
    }
    //second pass, calculate CMCs
    for(u32 idx=1; idx<=found; idx++)
    {
      u32 data= outPtr[idx]&0x7fffffff;
      if(data<=fullmatchkeyidx)
      {
        (*CMC)++;
        if((data > *iters_done) || (data == fullmatchkeyidx) || (*iters_done == 0x7fffffff))
        {
          *iters_done = data;
        }
      }
    }
    status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
    if(ocl_diagnose(status, "unmapping output buffer", cont) != CL_SUCCESS)
	  return -1;
    if(fullmatchkeyidx < 0xffffffff)
      return 1;
    return 0;
  }
  status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
  if(ocl_diagnose(status, "unmapping output buffer", cont) != CL_SUCCESS)
    return -1;
  return 0; 
}

//Internal test function to make sure core is working properly in user environment
static bool selftest(ocl_context_t *cont)
{
  RC5_72UnitWork tmp_unit;
  u32 CMC, iters_done;
  size_t globalWorkSize[1];    
  cl_int status;

  if(!ClearOutBuffer(cont->out_buffer, cont))
    return false;
	
  //first case: "not found"
  tmp_unit.L0.hi=0xcc;
  tmp_unit.L0.mid=0x55555555;
  tmp_unit.L0.lo=0xaaaaaaaa;
  tmp_unit.plain.lo=0x21436587;
  tmp_unit.plain.hi=0xa9cbed0f;
  tmp_unit.cypher.lo=0x12345678;
  tmp_unit.cypher.hi=0x9abcdef0;

  if(!FillConstantBuffer(cont->const_buffer, &tmp_unit, 1, cont))
    return false;	
  
  status  = clSetKernelArg(cont->kernel, 0, sizeof(cl_mem), &cont->const_buffer);
  status |= clSetKernelArg(cont->kernel, 1, sizeof(cl_mem), &cont->out_buffer);
  if(ocl_diagnose(status, "setting kernel arguments", cont) != CL_SUCCESS)
    return false;

  // Define an index space (global work size) of work items for execution.
  globalWorkSize[0] = 1;

  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
	  return false;

  s32 read_res=ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if((read_res<0)||(CMC>0)||(read_res>0))
	  return false;

  //second case, partial match
  tmp_unit.L0.hi=0xb6;
  tmp_unit.L0.mid=0xb843b603;
  tmp_unit.L0.lo=0x825d8bd0;
  tmp_unit.plain.lo=0x6cf2bd15;
  tmp_unit.plain.hi=0x989e1475;
  tmp_unit.cypher.lo=0xbefcafe7;
  tmp_unit.cypher.hi=0xa6ec745f;

  if(!FillConstantBuffer(cont->const_buffer, &tmp_unit, 1, cont))
    return false;	
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
	return false;
  read_res=ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if((read_res<0)||(CMC!=1)||(read_res>0))
	return false;

  //Third case, full match
  tmp_unit.L0.hi=0x2a;
  tmp_unit.L0.mid=0x47634;
  tmp_unit.L0.lo=0xba6196cc;
  tmp_unit.plain.lo=0xc13fb62;
  tmp_unit.plain.hi=0x7e370fcb;
  tmp_unit.cypher.lo=0x4da0ae1c;
  tmp_unit.cypher.hi=0xd1c60cfb;

  //Log("Self-test passed, device %u\n", deviceID);
  if(!FillConstantBuffer(cont->const_buffer, &tmp_unit, 1, cont))
    return false;	
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if(ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
	return false;
  read_res=ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if((read_res<=0)||(CMC!=1))
	return false;

  return true;
}

#undef CONST_SIZE
#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_ocl_ref (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
#endif
s32 rc5_72_unit_func_ocl_ref(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  ocl_context_t *cont = ocl_get_context(rc5_72unitwork->devicenum);
  RC5_72UnitWork tmp_unit;
  static bool selftestpassed=false;

  if (cont->coreID!=CORE_REF)
  {
    init_rc5_72_ocl_ref(cont);
    if(cont->coreID!=CORE_REF) {
      RaiseExitRequestTrigger();
      return -1;
    }
  }

  if(!selftestpassed)
  {
	if(!selftest(cont))
	{
        Log("Sanity checks failed - OpenCL system is not functional! Device: %d\n", cont->clientDeviceNo);
        RaiseExitRequestTrigger();
        return -1;
	}else
		selftestpassed = true;
  }

  //Log("Got %u iters to do\n", *iterations);
  //Log("%x:%x:%x - %u\n", rc5_72unitwork->L0.hi, rc5_72unitwork->L0.mid, rc5_72unitwork->L0.lo, *iterations);
  memmove(&tmp_unit, rc5_72unitwork, sizeof(RC5_72UnitWork));

  u32 kiter =*iterations;

  if(!ClearOutBuffer(cont->out_buffer, cont))
  {
	  //Log("Couldn't clear out buffer, device:#%u\n", deviceID);
      RaiseExitRequestTrigger();
      return -1;         
  }

  while(kiter) {
    u32 rest0;
	cl_int status;

	if(kiter>=cont->runSize)
      rest0=cont->runSize;
    else
      rest0=kiter;
    kiter-=rest0;
    //fill constant buffer
    if(!FillConstantBuffer(cont->const_buffer, &tmp_unit, rest0, cont))
    {
	  //Log("Error filling constant buffer, device:#%u\n", deviceID);
      RaiseExitRequestTrigger();
      return -1;          //err
    }
    
    status  = clSetKernelArg(cont->kernel, 0, sizeof(cl_mem), &cont->const_buffer);
    status |= clSetKernelArg(cont->kernel, 1, sizeof(cl_mem), &cont->out_buffer);
    if(ocl_diagnose(status, "setting kernel arguments", cont) != CL_SUCCESS)
	{
	  RaiseExitRequestTrigger();
	  return -1;          //err
	}

    // Define an index space (global work size) of work items for execution.
    size_t globalWorkSize[1];    
    globalWorkSize[0] = rest0;

    cl_event ndrEvt;
    status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &ndrEvt);
    if(ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
	{
	  RaiseExitRequestTrigger();
	  return -1;          //err
	}

    // wait for the kernel call to finish execution
	status = clWaitForEvents(1, &ndrEvt);

    cl_ulong startTime;
    cl_ulong endTime;
	status = clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, 0);
    status |= clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, 0);
    status |= clReleaseEvent(ndrEvt);

	double d;
	if(status == CL_SUCCESS)
		d=1e-6 * (endTime - startTime);
	else	
	{
		static bool profilingErr=false;
		if(!profilingErr)
		{
			profilingErr = true;
			ocl_diagnose(status, "getting profile info", cont);
		}
		d = 10;
	}

    if(d>12.)
	{
	  //Decrease worksize by 5%
	  u32 diffm = cont->runSize /20 /cont->runSizeMultiplier;
      if(cont->runSize>(diffm*cont->runSizeMultiplier))
		  cont->runSize -= diffm*cont->runSizeMultiplier;
	  //Log("Down:Time: %f, runsize=%u\n", float(d), cont->runSize); 
	}else
      if((d<8.) &&(rest0 == cont->runSize))
	  {
	    u32 diffm = cont->runSize /20 /cont->runSizeMultiplier;
        if(cont->runSize<cont->maxWorkSize)
		  cont->runSize += diffm*cont->runSizeMultiplier;
  	    //Log("Up:Time: %f, runsize=%u, diff=%u\n", float(d), cont->runSize, diffm*cont->runSizeMultiplier); 
	  }

    //Check the results
    u32 CMC, iters_done;
    s32 read_res=ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
	if(read_res<0)
	{
	  RaiseExitRequestTrigger();
	  return -1;          //err
	}

	if(CMC)
	{
		u32 hi = tmp_unit.L0.hi;
		u32 mid = tmp_unit.L0.mid;
		u32 lo = tmp_unit.L0.lo;

		//Log("%x:%x:%x\n", hi,mid,lo);
		key_incr(&hi, &mid, &lo, iters_done);
//		Log("CMC at:%x:%x:%x, plain:%x:%x, cypher=%x:%x\n", hi,mid,lo,tmp_unit.plain.hi,tmp_unit.plain.lo,tmp_unit.cypher.hi,tmp_unit.cypher.lo);
        
		//update cmc data
		tmp_unit.check.hi=hi;
        tmp_unit.check.mid=mid;
        tmp_unit.check.lo=lo;
        tmp_unit.check.count+=CMC;

		if(read_res>0) //Full match
		{
		  //re-check the result using reference software core
          RC5_72UnitWork t;

          memcpy(&t,rc5_72unitwork,sizeof(RC5_72UnitWork));
          t.L0.hi=hi;
          t.L0.mid=mid;
          t.L0.lo=lo;
          
          if(rc5_72_unit_func_ansi_ref(&t)!=RESULT_FOUND)
          {
            Log("WARNING!!! False positive detected, device:%u!\n", cont->clientDeviceNo);
            Log("Debug info: %x:%x:%x\n",hi,mid,lo);
            RaiseExitRequestTrigger();
            return -1;
		  }

		  tmp_unit.L0.hi = hi;
		  tmp_unit.L0.mid = mid;
		  tmp_unit.L0.lo = lo;

          *iterations -= kiter-iters_done+rest0;
          memmove(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
          return RESULT_FOUND;
		}
	}
	key_incr(&tmp_unit.L0.hi, &tmp_unit.L0.mid, &tmp_unit.L0.lo, rest0);
  }

  /* tell the client about the optimal timeslice increment for this core
     (with current parameters) */
  memmove(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
  rc5_72unitwork->optimal_timeslice_increment = cont->runSize;
  return RESULT_NOTHING;
}

