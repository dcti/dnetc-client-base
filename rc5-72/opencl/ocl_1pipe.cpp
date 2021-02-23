/*
* Copyright distributed.net 2012 - All Rights Reserved
* For use in distributed.net projects only.
* Any other distribution or use of this source violates copyright.
*
* $Id:
*/

#ifndef ANOTHER_PASS

#include "ocl_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __SLEEP_FOR_POLLING__
#include "sleepdef.h"

#include "rc5-1pipe.cpp"
#include "rc5-2pipe.cpp"
#include "rc5-4pipe.cpp"

#define CONST_SIZE (sizeof(cl_uint)*16)
#define OUT_SIZE (sizeof(cl_uint)*128)

static bool init_rc5_72_ocl_npipe(ocl_context_t *cont, unsigned core_ID, const char *core_program, const char *program_entry)
{
  if (!cont->active)
  {
    Log("Device %u is not supported\n", cont->clientDeviceNo);
    return false;
  }

  if (cont->coreID != core_ID)
    OCLReinitializeDevice(cont);

  cl_int status;
  // Create a context and associate it with the device
  cont->clcontext = clCreateContext(NULL, 1, &cont->deviceID, NULL, NULL, &status);
  if (ocl_diagnose(status, "creating OCL context", cont) != CL_SUCCESS)
    return false;

  //Create a command queue
  cont->cmdQueue = clCreateCommandQueue(cont->clcontext, cont->deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
  if (ocl_diagnose(status, "creating command queue", cont) != CL_SUCCESS)
    return false;

  //Create device buffers
  cont->const_buffer = clCreateBuffer(cont->clcontext, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, CONST_SIZE, NULL, &status); //CL_MEM_COPY_HOST_WRITE_ONLY ??
  if (ocl_diagnose(status, "creating constants buffer", cont) != CL_SUCCESS)
    return false;

  cont->out_buffer = clCreateBuffer(cont->clcontext, CL_MEM_ALLOC_HOST_PTR, OUT_SIZE, NULL, &status);
  if (ocl_diagnose(status, "creating output buffer", cont) != CL_SUCCESS)
    return false;

  if (!BuildCLProgram(cont, core_program, program_entry))
    return false;

  //Get a performance hint
  size_t prefm;
  status = clGetKernelWorkGroupInfo(cont->kernel, cont->deviceID, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                    sizeof(prefm), &prefm, NULL);
  if (status == CL_SUCCESS)
  {
    size_t cus;
    status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL);
    if (status == CL_SUCCESS)
      cont->runSizeMultiplier = prefm * cus * 4; //Hack for now. We need 4 wavefronts per CU to hide latency
  }
  //Log("Multiplier = %u\n", cont->runSizeMultiplier);
  unsigned t = cont->runSize/cont->runSizeMultiplier;
  if (t == 0) t = 1;
  cont->runSize = cont->runSizeMultiplier * t; //To be sure runsize is divisible by multiplier

  /*size_t workitem_size[3];
  status = clGetDeviceInfo(cont->deviceID, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(workitem_size), &prefm, NULL);
  if(status==CL_SUCCESS)
  {
    cont->maxWorkSize = workitem_size[0] * workitem_size[1] * workitem_size[2];
    Log("max worksize = %u\n", cont->maxWorkSize);
  }*/
  cont->maxWorkSize = 0xffffffff;
  cont->coreID = core_ID;
  return true;
}

static bool ClearOutBuffer(const cl_mem buffer, ocl_context_t *cont)
{
  cl_uint *outPtr = NULL;
  cl_int status;

  outPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, buffer, CL_TRUE, CL_MAP_WRITE, 0, 4, 0, NULL, NULL, &status);
  if (ocl_diagnose(status, "mapping output buffer", cont) != CL_SUCCESS)
    return false;
  outPtr[0] = 0;

  status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
  if (ocl_diagnose(status, "unmapping output buffer (1)", cont) != CL_SUCCESS)
    return false;
  return true;
}

static bool FillConstantBuffer(const cl_mem buffer, RC5_72UnitWork *rc5_72unitwork, cl_uint iter_offset, ocl_context_t *cont)
{
  cl_uint *constPtr = NULL;
  cl_int status;

  constPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, buffer, CL_TRUE, CL_MAP_WRITE, 0, CONST_SIZE, 0, NULL, NULL, &status);
  if (ocl_diagnose(status, "mapping constants buffer", cont) != CL_SUCCESS)
    return false;

  //key_hi,key_mid,key_lo,granularity
  constPtr[0] = rc5_72unitwork->L0.hi;
  //constPtr[1]=swap32(rc5_72unitwork->L0.mid);
  constPtr[1] = SWAP32(rc5_72unitwork->L0.mid);
  constPtr[2] = rc5_72unitwork->L0.lo;
  constPtr[3] = iter_offset;

  //plain_lo,plain_hi,cypher_lo,cypher_hi
  constPtr[4] = rc5_72unitwork->plain.lo;
  constPtr[5] = rc5_72unitwork->plain.hi;
  constPtr[6] = rc5_72unitwork->cypher.lo;
  constPtr[7] = rc5_72unitwork->cypher.hi;

  cl_uint l0 = ROTL(0xBF0A8B1D+rc5_72unitwork->L0.lo, 0x1d); //L0=ROTL(L0+S0,S0)=ROTL(L0+S0,0x1d)
  constPtr[8] = l0;
  constPtr[9] = ROTL3(l0+0xBF0A8B1D+0x5618cb1c);             //S1=ROTL3(Sc1+S0+L0)

  status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, constPtr, 0, NULL, NULL);
  if (ocl_diagnose(status, "unmapping constants buffer", cont) != CL_SUCCESS)
    return false;
  return true;
}

static s32 ReadResults(const cl_mem buffer, u32 *CMC, u32 *iters_done, ocl_context_t *cont)
{
  cl_uint *outPtr = NULL;
  cl_int status;
  cl_uint found;

  *CMC = 0;
  *iters_done = 0xffffffff;
  outPtr = (cl_uint*) clEnqueueMapBuffer(cont->cmdQueue, buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, OUT_SIZE, 0, NULL, NULL, &status);
  if (ocl_diagnose(status, "mapping output buffer", cont) != CL_SUCCESS)
    return -1;

  //Log("A=%x B=%x\n", outPtr[0], outPtr[1]);
  found = outPtr[0];
  outPtr[0] = 0;
  if (found)
  {
    u32 fullmatchkeyidx = 0xffffffff;
    u32 fullmatch = 0;
    if (found >= (OUT_SIZE/sizeof(cl_uint)/2))
    {
      Log("Internal error reading kernel output\n");
      clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
      return -1;
    }
    for (u32 idx = 0; idx < found; idx++)
    {
      if (outPtr[idx*2+1] & 0x80000000)
      {
        fullmatchkeyidx = outPtr[idx*2+2];
        fullmatch = 1;
        break;
      }
    }

    //second pass, calculate CMCs
    for (u32 idx = 1; idx <= found; idx++)
    {
      u32 data = outPtr[idx*2];
      if (data <= fullmatchkeyidx)
      {
        (*CMC)++;
        if ((data > *iters_done) || (data == fullmatchkeyidx) || (*iters_done == 0xffffffff))
        {
          *iters_done = data;
        }
      }
    }
    status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
    if (ocl_diagnose(status, "unmapping output buffer (2)", cont) != CL_SUCCESS)
      return -1;
    if (fullmatch)
      return 1;
    return 0;
  }
  status = clEnqueueUnmapMemObject(cont->cmdQueue, buffer, outPtr, 0, NULL, NULL);
  if (ocl_diagnose(status, "unmapping output buffer (3)", cont) != CL_SUCCESS)
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

  if (!ClearOutBuffer(cont->out_buffer, cont))
    return false;

  //first case: "not found"
  tmp_unit.L0.hi = 0xcc;
  tmp_unit.L0.mid = 0x55555555;
  tmp_unit.L0.lo = 0xaaaaaaaa;
  tmp_unit.plain.lo = 0x21436587;
  tmp_unit.plain.hi = 0xa9cbed0f;
  tmp_unit.cypher.lo = 0x12345678;
  tmp_unit.cypher.hi = 0x9abcdef0;

  if (!FillConstantBuffer(cont->const_buffer, &tmp_unit, 0, cont))
    return false;

  status = clSetKernelArg(cont->kernel, 0, sizeof(cl_mem), &cont->const_buffer);
  if (status == CL_SUCCESS)
    status = clSetKernelArg(cont->kernel, 1, sizeof(cl_mem), &cont->out_buffer);
  if (ocl_diagnose(status, "setting kernel arguments", cont) != CL_SUCCESS)
    return false;

  // Define an index space (global work size) of work items for execution.
  globalWorkSize[0] = 1;

  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if (ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
    return false;

  s32 read_res = ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if ((read_res < 0) || (CMC > 0) || (read_res > 0))
    return false;

  //second case, partial match
  tmp_unit.L0.hi = 0xb6;
  tmp_unit.L0.mid = 0xb843b603;
  tmp_unit.L0.lo = 0x825d8bd0;
  tmp_unit.plain.lo = 0x6cf2bd15;
  tmp_unit.plain.hi = 0x989e1475;
  tmp_unit.cypher.lo = 0xbefcafe7;
  tmp_unit.cypher.hi = 0xa6ec745f;

  if (!FillConstantBuffer(cont->const_buffer, &tmp_unit, 0, cont))
    return false;
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if (ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
    return false;
  read_res = ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if ((read_res < 0) || (CMC != 1) || (read_res > 0))
    return false;

  //Third case, full match
  tmp_unit.L0.hi = 0x2a;
  tmp_unit.L0.mid = 0x47634;
  tmp_unit.L0.lo = 0xba6196cc;
  tmp_unit.plain.lo = 0xc13fb62;
  tmp_unit.plain.hi = 0x7e370fcb;
  tmp_unit.cypher.lo = 0x4da0ae1c;
  tmp_unit.cypher.hi = 0xd1c60cfb;

  if (!FillConstantBuffer(cont->const_buffer, &tmp_unit, 0, cont))
    return false;
  status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  if (ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
    return false;
  read_res = ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if ((read_res <= 0) || (CMC != 1))
    return false;

  return true;
}

#undef CONST_SIZE
#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_ocl_1pipe (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_2pipe (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_4pipe (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_1pipe_large (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_2pipe_large (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_4pipe_large (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_1pipe_sleep (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_2pipe_sleep (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
extern "C" s32 rc5_72_unit_func_ocl_4pipe_sleep (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
#endif

/* some static flags which are set on per-core basis */
struct core_static_flags
{
  bool selftestpassed;
  bool profilingErr;
};

#define FLOAT_INTERNAL_HELPER(x) x##.
#define FLOAT_HELPER(x) FLOAT_INTERNAL_HELPER(x)
#define MIN_DESIRED_UNIT_MS 8.
#define MAX_DESIRED_UNIT_MS_INT 12
#define MAX_DESIRED_UNIT_MS FLOAT_HELPER(MAX_DESIRED_UNIT_MS_INT)
#define MAX_SLEEP_MKS (2 * (1000 * MAX_DESIRED_UNIT_MS_INT * SCALING_RATIO))
#define MIN_SLEEP_MKS 1000

#endif /* ANOTHER_PASS */

#ifndef SCALING_RATIO
#define SCALING_RATIO 1
#endif /* SCALING_RATIO */

static s32
#if defined(USE_SLEEP)
rc5_72_unit_func_ocl_npipe_sleep
#elif SCALING_RATIO > 1
rc5_72_unit_func_ocl_npipe_large
#else
rc5_72_unit_func_ocl_npipe
#endif
(RC5_72UnitWork *rc5_72unitwork, u32 *iterations,
                                      unsigned core_ID, unsigned pipes_count,
                                      const char *core_program, const char *program_entry,
                                      struct core_static_flags *static_flags)
{
  ocl_context_t *cont = ocl_get_context(rc5_72unitwork->devicenum);
  RC5_72UnitWork tmp_unit;
  cl_int status;

  if (cont == NULL)
  {
    RaiseExitRequestTrigger();
    return -1;
  }

  if (cont->coreID != core_ID)
  {
    init_rc5_72_ocl_npipe(cont, core_ID, core_program, program_entry);
    if (cont->coreID != core_ID) {
      RaiseExitRequestTrigger();
      return -1;
    }
  }
  if (!static_flags->selftestpassed)
  {
    if (!selftest(cont))
    {
      Log("Sanity checks failed - OpenCL system is not functional! Device: %d\n", cont->clientDeviceNo);
      RaiseExitRequestTrigger();
      return -1;
    } else
      static_flags->selftestpassed = true;
  }

  memmove(&tmp_unit, rc5_72unitwork, sizeof(RC5_72UnitWork));

  u32 kiter = *iterations / pipes_count;

//LogTo(LOGTO_FILE, "enter: iter=%X kiter=%X rsz=%X rsm=%X\n",
//       *iterations, kiter, cont->runSize, cont->runSizeMultiplier);

  if (!ClearOutBuffer(cont->out_buffer, cont))
  {
    //Log("Couldn't clear out buffer, device:#%u\n", deviceID);
    RaiseExitRequestTrigger();
    return -1;
  }

  status = clSetKernelArg(cont->kernel, 0, sizeof(cl_mem), &cont->const_buffer);
  if (status == CL_SUCCESS)
    status = clSetKernelArg(cont->kernel, 1, sizeof(cl_mem), &cont->out_buffer);
  if (ocl_diagnose(status, "setting kernel arguments", cont) != CL_SUCCESS)
  {
    RaiseExitRequestTrigger();
    return -1;          //err
  }

  u32 iter_offset = 0;
  while (kiter) {
    u32 rest0;

    if (kiter >= cont->runSize)
      rest0 = cont->runSize;
    else
      rest0 = kiter;
    kiter -= rest0;
    //fill constant buffer
    if (!FillConstantBuffer(cont->const_buffer, &tmp_unit, iter_offset, cont))
    {
      //Log("Error filling constant buffer, device:#%u\n", deviceID);
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    // Define an index space (global work size) of work items for execution.
    size_t globalWorkSize[1];
    globalWorkSize[0] = rest0;

    cl_event ndrEvt;
    status = clEnqueueNDRangeKernel(cont->cmdQueue, cont->kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, &ndrEvt);
    if (ocl_diagnose(status, "launching kernel", cont) != CL_SUCCESS)
    {
      RaiseExitRequestTrigger();
      return -1;          //err
    }

#ifdef USE_SLEEP
    /* Sleep is useless w/o flush. */
    status = clFlush(cont->cmdQueue);
    if (ocl_diagnose(status, "flushing work", cont) != CL_SUCCESS)
    {
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    unsigned sleepTimeMks;
    if (cont->estimatedPerf) {
      sleepTimeMks = (unsigned)(1000000ULL * rest0 / cont->estimatedPerf);
      if (sleepTimeMks > MAX_SLEEP_MKS)
        sleepTimeMks = MAX_SLEEP_MKS;
      else if (sleepTimeMks < MIN_SLEEP_MKS)
        sleepTimeMks = MIN_SLEEP_MKS;
    } else
      sleepTimeMks = MIN_SLEEP_MKS;

    usleep(sleepTimeMks);
#endif /* USE_SLEEP */

    // wait for the kernel call to finish execution
    status = clWaitForEvents(1, &ndrEvt);
    if (ocl_diagnose(status, "waiting for event", cont) != CL_SUCCESS)
    {
      clReleaseEvent(ndrEvt);  // allocated locally, must (try to) free before return
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    cl_ulong startTime;
    cl_ulong endTime;
    cl_int   status_p1, status_p2;
    status_p1 = clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, 0);
    status_p2 = clGetEventProfilingInfo(ndrEvt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, 0);

    // event must be released in any case
    status = clReleaseEvent(ndrEvt);
    if (ocl_diagnose(status, "releasing event", cont) != CL_SUCCESS)
    {
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    double d;
    if (status_p1 == CL_SUCCESS && status_p2 == CL_SUCCESS) // this is status of clGetEventProfilingInfo
      d = 1e-6 * (endTime - startTime);
    else
    {
      if (!static_flags->profilingErr)
      {
        static_flags->profilingErr = true;
        ocl_diagnose((status_p1 != CL_SUCCESS ? status_p1 : status_p2), "getting profile info", cont);
      }
      d = 10;
    }

    if (d > MAX_DESIRED_UNIT_MS * SCALING_RATIO)
    {
      //Decrease worksize by 5%
      u32 diffm = cont->runSize / 20 / cont->runSizeMultiplier;
      if (diffm == 0)
        diffm = 1;
      if (cont->runSize > (diffm*cont->runSizeMultiplier))
        cont->runSize -= diffm*cont->runSizeMultiplier;
      //Log("Down:Time: %f, runsize=%u\n", float(d), cont->runSize);
    } else
    if ((d < MIN_DESIRED_UNIT_MS * SCALING_RATIO) && (rest0 == cont->runSize))
    {
      u32 diffm = cont->runSize / 20 / cont->runSizeMultiplier;
      if (diffm == 0)
        diffm = 1;
      if (cont->runSize < cont->maxWorkSize)
        cont->runSize += diffm*cont->runSizeMultiplier;
      //Log("Up:Time: %f, runsize=%u, diff=%u\n", float(d), cont->runSize, diffm*cont->runSizeMultiplier);
    }

    cont->estimatedPerf = (unsigned long long)(rest0 / (d / 1000));

    key_incr(&tmp_unit.L0.hi, &tmp_unit.L0.mid, &tmp_unit.L0.lo, rest0 * pipes_count);
    iter_offset += rest0 * pipes_count;
  }

  //Check the results
  u32 CMC, iters_done;
  s32 read_res = ReadResults(cont->out_buffer, &CMC, &iters_done, cont);
  if (read_res < 0)
  {
    RaiseExitRequestTrigger();
    return -1;            //err
  }

  if (CMC)
  {
    u32 hi = rc5_72unitwork->L0.hi;
    u32 mid = rc5_72unitwork->L0.mid;
    u32 lo = rc5_72unitwork->L0.lo;

    //Log("%x:%x:%x\n", hi,mid,lo);
    key_incr(&hi, &mid, &lo, iters_done);
    //Log("CMC at:%x:%x:%x, plain:%x:%x, cypher=%x:%x\n", hi,mid,lo,tmp_unit.plain.hi,tmp_unit.plain.lo,tmp_unit.cypher.hi,tmp_unit.cypher.lo);

    //update cmc data
    tmp_unit.check.hi = hi;
    tmp_unit.check.mid = mid;
    tmp_unit.check.lo = lo;
    tmp_unit.check.count += CMC;

    if (read_res > 0)  //Full match
    {
      //re-check the result using reference software core
      RC5_72UnitWork t;

      memcpy(&t, rc5_72unitwork, sizeof(RC5_72UnitWork));
      t.L0.hi = hi;
      t.L0.mid = mid;
      t.L0.lo = lo;

      if (rc5_72_unit_func_ansi_ref(&t) != RESULT_FOUND)
      {
        Log("WARNING!!! False positive detected, device:%u!\n", cont->clientDeviceNo);
        Log("Debug info: %x:%x:%x\n", hi, mid, lo);
        RaiseExitRequestTrigger();
        return -1;
      }

      tmp_unit.L0.hi = hi;
      tmp_unit.L0.mid = mid;
      tmp_unit.L0.lo = lo;

      *iterations = iters_done;
      memmove(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
      return RESULT_FOUND;
    }
  }

  /* tell the client about the optimal timeslice increment for this core
     (with current parameters) */
  memmove(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
  rc5_72unitwork->optimal_timeslice_increment = cont->runSizeMultiplier * pipes_count;
  return RESULT_NOTHING;
}

#ifndef ANOTHER_PASS

#define ANOTHER_PASS

#undef SCALING_RATIO
#define SCALING_RATIO 100 /* Higher performance, especially with USE_SLEEP. */

#include "ocl_1pipe.cpp"

#define USE_SLEEP /* Required for low CPU usage on some hardware/drivers. */

#include "ocl_1pipe.cpp"

s32 rc5_72_unit_func_ocl_1pipe(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe(rc5_72unitwork, iterations, CORE_1PIPE, 1, ocl_rc572_1pipe_src, "ocl_rc572_1pipe", &flags);
}

s32 rc5_72_unit_func_ocl_2pipe(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe(rc5_72unitwork, iterations, CORE_2PIPE, 2, ocl_rc572_2pipe_src, "ocl_rc572_2pipe", &flags);
}

s32 rc5_72_unit_func_ocl_4pipe(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe(rc5_72unitwork, iterations, CORE_4PIPE, 4, ocl_rc572_4pipe_src, "ocl_rc572_4pipe", &flags);
}

s32 rc5_72_unit_func_ocl_1pipe_large(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe_large(rc5_72unitwork, iterations, CORE_1PIPE, 1, ocl_rc572_1pipe_src, "ocl_rc572_1pipe", &flags);
}

s32 rc5_72_unit_func_ocl_2pipe_large(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe_large(rc5_72unitwork, iterations, CORE_2PIPE, 2, ocl_rc572_2pipe_src, "ocl_rc572_2pipe", &flags);
}

s32 rc5_72_unit_func_ocl_4pipe_large(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe_large(rc5_72unitwork, iterations, CORE_4PIPE, 4, ocl_rc572_4pipe_src, "ocl_rc572_4pipe", &flags);
}

s32 rc5_72_unit_func_ocl_1pipe_sleep(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe_sleep(rc5_72unitwork, iterations, CORE_1PIPE, 1, ocl_rc572_1pipe_src, "ocl_rc572_1pipe", &flags);
}

s32 rc5_72_unit_func_ocl_2pipe_sleep(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe_sleep(rc5_72unitwork, iterations, CORE_2PIPE, 2, ocl_rc572_2pipe_src, "ocl_rc572_2pipe", &flags);
}

s32 rc5_72_unit_func_ocl_4pipe_sleep(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  static struct core_static_flags flags;

  return rc5_72_unit_func_ocl_npipe_sleep(rc5_72unitwork, iterations, CORE_4PIPE, 4, ocl_rc572_4pipe_src, "ocl_rc572_4pipe", &flags);
}
#endif /* ANOTHER_PASS */
