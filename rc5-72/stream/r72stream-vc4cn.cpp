/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-vc4cn.cpp,v 1.10 2009/03/29 20:02:27 andreasb Exp $
*/

#include "r72stream-common.h"
#include "r72stream-vc4cn_il.cpp"

bool init_rc5_72_il4_nand(u32 Device)
{
  if(CContext[Device].coreID!=CORE_IL4N)
    AMDStreamReinitializeDevice(Device);

  if(!CContext[Device].active)
  {
    LogScreen("Device is not supported\n");
    return false;
  } else{
    switch(CContext[Device].attribs.target) {
    case CAL_TARGET_600:
      CContext[Device].domainSizeX=56;
      CContext[Device].domainSizeY=56;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_610:
      CContext[Device].domainSizeX=32;
      CContext[Device].domainSizeY=32;
      CContext[Device].maxIters=64;
      break;
    case CAL_TARGET_630:
      CContext[Device].domainSizeX=64;
      CContext[Device].domainSizeY=54;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_670:
      CContext[Device].domainSizeX=64;
      CContext[Device].domainSizeY=16;
      CContext[Device].maxIters=300;
      break;
    case CAL_TARGET_7XX:
      //TODO:domainSize
      break;
    case CAL_TARGET_770:
      CContext[Device].domainSizeX=128;
      CContext[Device].domainSizeY=80;
      CContext[Device].maxIters=512;
      break;
    case CAL_TARGET_710:
      //TODO:domainSize
      CContext[Device].domainSizeX=80;
      CContext[Device].domainSizeY=80;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_730:
      CContext[Device].domainSizeX=128;
      CContext[Device].domainSizeY=64;
      CContext[Device].maxIters=128;
      break;
    default:
      break;
    }
  }

  // Creating context w.r.t. to opened device
  calCtxCreate(&CContext[Device].ctx, CContext[Device].device);

  //-------------------------------------------------------------------------
  // Compiling Device Program
  //-------------------------------------------------------------------------
  if (calclCompile(&CContext[Device].obj, CAL_LANGUAGE_IL, il4_nand_src, CContext[Device].attribs.target) != CAL_RESULT_OK)
  {
    LogScreen("Core compilation failed. Exiting.\n");
    return false;
  }

  if (calclLink(&CContext[Device].image, &CContext[Device].obj, 1) != CAL_RESULT_OK)
  {
    LogScreen("Core linking failed. Exiting.\n");
    return false;
  }

  //-------------------------------------------------------------------------
  // Allocating and initializing resources
  //-------------------------------------------------------------------------
  // Input and output resources
  CContext[Device].outputRes0=0;
  if(CContext[Device].attribs.cachedRemoteRAM>0)
  {
    if (calResAllocRemote2D(&CContext[Device].outputRes0, &CContext[Device].device, 1, CContext[Device].domainSizeX,
                            CContext[Device].domainSizeY, CAL_FORMAT_UINT_1, CAL_RESALLOC_CACHEABLE) == CAL_RESULT_OK)
    {
      //LogScreen("Using cached remote buffer\n");
    }
  }

  if(!CContext[Device].outputRes0) {
    if(calResAllocRemote2D(&CContext[Device].outputRes0, &CContext[Device].device, 1, CContext[Device].domainSizeX,
                           CContext[Device].domainSizeY, CAL_FORMAT_UINT_1, 0)!=CAL_RESULT_OK)
    {
      LogScreen("Failed to allocate output buffer.\n");
      return false;
    } else {
      //LogScreen("Using uncached remote buffer\n");
    }
  }

  // Constant resource
  if(calResAllocLocal1D(&CContext[Device].constRes, CContext[Device].device, 3, CAL_FORMAT_UINT_4, 0)!=CAL_RESULT_OK)
  {
    LogScreen("Failed to allocate constant buffer.\n");
    return false;
  }

  if(amdstream_usePerfCounters)
    if (calCtxCreateCounterExt(&(CContext[Device].idleCounter), CContext[Device].ctx, CAL_COUNTER_IDLE) != CAL_RESULT_OK)
      CContext[Device].idleCounter=0;

  // Creating module using compiled image
  calModuleLoad(&CContext[Device].module, CContext[Device].ctx, CContext[Device].image);

  // Mapping output resource to CPU and initializing values
  // Getting memory handle from resources
  calCtxGetMem(&CContext[Device].outputMem0, CContext[Device].ctx, CContext[Device].outputRes0);
  calCtxGetMem(&CContext[Device].constMem, CContext[Device].ctx, CContext[Device].constRes);

  // Defining entry point for the module
  calModuleGetEntry(&CContext[Device].func, CContext[Device].ctx, CContext[Device].module, "main");
  calModuleGetName(&CContext[Device].outName0, CContext[Device].ctx, CContext[Device].module, "o0");
  calModuleGetName(&CContext[Device].constName, CContext[Device].ctx, CContext[Device].module, "cb0");

  // Setting input and output buffers
  // used in the kernel
  calCtxSetMem(CContext[Device].ctx, CContext[Device].outName0, CContext[Device].outputMem0);
  calCtxSetMem(CContext[Device].ctx, CContext[Device].constName, CContext[Device].constMem);

  CContext[Device].coreID=CORE_IL4N;

  return true;
}

#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_il4_nand (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
#endif

s32 rc5_72_unit_func_il4_nand(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  u32 deviceID=rc5_72unitwork->threadnum;
  if (CContext[deviceID].coreID!=CORE_IL4N)
  {
    init_rc5_72_il4_nand(deviceID);
    if(CContext[deviceID].coreID!=CORE_IL4N) {
      return -1;        //årr
    }
  }

  u32 kiter =(*iterations)/4;

  u32 itersNeeded=kiter;
  u32 width=CContext[deviceID].domainSizeX;
  u32 height=CContext[deviceID].domainSizeY;
  u32 RunSize=width*height;

  //LogScreen("%u ITERS (%u)\n",kiter,kiter/RunSize);

  while(itersNeeded) {
    u32 iters,rest;

    iters=itersNeeded/RunSize;
    if(iters>=CContext[deviceID].maxIters)
    {
      iters=CContext[deviceID].maxIters;
      rest=RunSize;
    } else
    {
      rest=itersNeeded-iters*RunSize;
      iters++;
    }

    //fill constant buffer
    unsigned* constPtr = NULL;
    CALuint pitch = 0;
    calResMap((CALvoid**)&constPtr, &pitch, CContext[deviceID].constRes, 0);

    //cb0[0]					//key_hi,key_mid,key_lo,granularity
    constPtr[0]=rc5_72unitwork->L0.hi;
    constPtr[1]=rc5_72unitwork->L0.mid;
    constPtr[2]=rc5_72unitwork->L0.lo;
    constPtr[3]=RunSize*4;

    //cb0[1]					//plain_lo,plain_hi,cypher_lo,cypher_hi
    constPtr[4]=rc5_72unitwork->plain.lo;
    constPtr[5]=rc5_72unitwork->plain.hi;
    constPtr[6]=rc5_72unitwork->cypher.lo;
    constPtr[7]=rc5_72unitwork->cypher.hi;

    //cb0[2]					//iters,rest,width
    constPtr[8]=iters;
    constPtr[9]=rest;
    float *f;
    f=(float*)&constPtr[10]; *f=(float)width;
    constPtr[11]=iters;

    calResUnmap(CContext[deviceID].constRes);

    CALdomain domain = {0, 0, width, height};

    // Event to check completion of the program
    CALevent e = 0;

    //Start idle counter
    bool perfC = false;
    if(amdstream_usePerfCounters) {
      if (calCtxBeginCounterExt(CContext[deviceID].ctx, CContext[deviceID].idleCounter) == CAL_RESULT_OK)
        perfC=true;
    }

    calCtxRunProgram(&e, CContext[deviceID].ctx, CContext[deviceID].func, &domain);
    calCtxIsEventDone(CContext[deviceID].ctx, e);

    struct timeval tv_ctx_start, tv_ctx_busy, tv_ctx_finish;
    CliTimer(&tv_ctx_start);

    if(iters==CContext[deviceID].maxIters)
      NonPolledUSleep(15000);   //15ms
    else
      NonPolledUSleep(15000*iters/CContext[deviceID].maxIters);

    // Checking whether the execution of the program is complete or not
    u32 busy_counter=0;

    CliTimer(&tv_ctx_busy);

    while (calCtxIsEventDone(CContext[deviceID].ctx, e) == CAL_RESULT_PENDING)
      busy_counter=1;

    CliTimer(&tv_ctx_finish);

    if(perfC)
      calCtxEndCounterExt(CContext[deviceID].ctx, CContext[deviceID].idleCounter);

    //Check the results
    unsigned *o0;
    calResMap((CALvoid**)&o0, &pitch, CContext[deviceID].outputRes0, 0);

    for(u32 i=0; i<height; i++) {
      u32 idx=i*pitch;
      for(u32 j=0; j<width; j++) {
        if(o0[idx+j])           //partial match
        {
          u32 CMC_count=(o0[idx+j]&0x7fffffff)>>18;
          u32 CMC_iter=(((o0[idx+j]>>2)&0x0000ffff)-1)*RunSize;
          u32 CMC_hit=(CMC_iter+i*width+j)*4+(o0[idx+j]&0x00000003);

          rc5_72unitwork->check.count+=CMC_count;
          rc5_72unitwork->check.hi=rc5_72unitwork->L0.hi;
          rc5_72unitwork->check.mid=rc5_72unitwork->L0.mid;
          rc5_72unitwork->check.lo=rc5_72unitwork->L0.lo;
          key_incr(&rc5_72unitwork->check.hi,&rc5_72unitwork->check.mid,&rc5_72unitwork->check.lo,CMC_hit);

          if(o0[idx+j]&0x80000000) {            //full match

            *iterations -= (kiter*4-CMC_hit);

            rc5_72unitwork->L0.hi=rc5_72unitwork->check.hi;
            rc5_72unitwork->L0.mid=rc5_72unitwork->check.mid;
            rc5_72unitwork->L0.lo=rc5_72unitwork->check.lo;

            calResUnmap(CContext[deviceID].outputRes0);

            return RESULT_FOUND;
          }
        }
      }
    }
    calResUnmap(CContext[deviceID].outputRes0);

    if(iters==CContext[deviceID].maxIters)
    {
      CALfloat idlePercentage;
      if(busy_counter==0) {
        if(perfC) {
          if (calCtxGetCounterExt(&idlePercentage, CContext[deviceID].ctx, CContext[deviceID].idleCounter) == CAL_RESULT_OK) {
            if(idlePercentage>0.01f) {
              float delta=(idlePercentage-0.01f)+1.f;
              CContext[deviceID].maxIters*=delta;
            }
          }
        }
      } else {

        CliTimerDiff(&tv_ctx_busy, &tv_ctx_busy, &tv_ctx_finish);
        CliTimerDiff(&tv_ctx_finish, &tv_ctx_start, &tv_ctx_finish);
        double ctx_busywait = (double)tv_ctx_busy.tv_sec * 1000.0 + (double)tv_ctx_busy.tv_usec / 1000.0;
        double ctx_elapsed = (double)tv_ctx_finish.tv_sec * 1000.0 + (double)tv_ctx_finish.tv_usec / 1000.0;
        double delta=ctx_busywait/ctx_elapsed-0.005;

        CContext[deviceID].maxIters*=delta;
        if(CContext[deviceID].maxIters==0)
          CContext[deviceID].maxIters=1;
      }
    }

    u32 itersDone=(iters-1)*RunSize+rest;
    kiter-=itersDone;
    key_incr(&rc5_72unitwork->L0.hi,&rc5_72unitwork->L0.mid,&rc5_72unitwork->L0.lo,itersDone*4);
    itersNeeded-=itersDone;
  }

  /* tell the client about the optimal timeslice increment for this core
     (with current parameters) */
  rc5_72unitwork->optimal_timeslice_increment = RunSize*4*CContext[deviceID].maxIters;
  return RESULT_NOTHING;
}
