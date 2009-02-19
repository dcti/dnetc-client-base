/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev
 * PanAm
 * Alexei Chupyatov
 *
 * $Id: r72stream-vc4cn.cpp,v 1.6 2009/02/19 23:19:29 andreasb Exp $
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
      CContext[Device].domainSizeX=56;
      CContext[Device].domainSizeY=56;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_670:
      CContext[Device].domainSizeX=56;
      CContext[Device].domainSizeY=56;
      CContext[Device].maxIters=256;
      break;
    case CAL_TARGET_7XX:
      //TODO:domainSize
      break;
    case CAL_TARGET_770:
      CContext[Device].domainSizeX=88;
      CContext[Device].domainSizeY=88;
      CContext[Device].maxIters=512;
      break;
    case CAL_TARGET_710:
      //TODO:domainSize
      CContext[Device].domainSizeX=80;
      CContext[Device].domainSizeY=80;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_730:
      //TODO:domainSize
      CContext[Device].domainSizeX=80;
      CContext[Device].domainSizeY=80;
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

  CALresult result;
  result=calclCompile(&CContext[Device].obj, CAL_LANGUAGE_IL, il4_nand_src, CContext[Device].attribs.target);
  if ( result!= CAL_RESULT_OK)
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
  CALresult r;
  if(CContext[Device].attribs.cachedRemoteRAM>0)    //одного мегабайта нам должно хватить
  {
    if((r=calResAllocRemote2D(&CContext[Device].outputRes0, &CContext[Device].device, 1, CContext[Device].domainSizeX,
                              CContext[Device].domainSizeY, CAL_FORMAT_UINT_4, CAL_RESALLOC_CACHEABLE))==CAL_RESULT_OK)
      LogScreen("Using cached remote buffer\n");
  }

  if(!CContext[Device].outputRes0) {
    if(calResAllocRemote2D(&CContext[Device].outputRes0, &CContext[Device].device, 1, CContext[Device].domainSizeX,
                           CContext[Device].domainSizeY, CAL_FORMAT_UINT_4, 0)!=CAL_RESULT_OK)
    {
      LogScreen("Failed to allocate output buffer.\n");
      return false;
    } else {
      LogScreen("Using uncached remote buffer\n");
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
      return -1;        //еrr
    }
  }

  u32 kiter =(*iterations)/4;

  unsigned itersNeeded=kiter;
  unsigned width=CContext[deviceID].domainSizeX;
  unsigned height=CContext[deviceID].domainSizeY;
  unsigned RunSize=width*height;
  u32 maxIters=CContext[deviceID].maxIters;

  while(itersNeeded) {
    unsigned iters,rest;
    bool perfC=0;

    iters=itersNeeded/RunSize;
    if(iters>=maxIters)
    {
      iters=maxIters;
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
    perfC=false;
    if(amdstream_usePerfCounters) {
      if (calCtxBeginCounterExt(CContext[deviceID].ctx, CContext[deviceID].idleCounter) == CAL_RESULT_OK)
        perfC=true;
    }
    if(!perfC)
      LogScreen("Perf counter error!\n");

    calCtxRunProgram(&e, CContext[deviceID].ctx, CContext[deviceID].func, &domain);
    calCtxIsEventDone(CContext[deviceID].ctx, e);
    NonPolledUSleep(30000); //30ms

    // Checking whether the execution of the program is complete or not
    while (calCtxIsEventDone(CContext[deviceID].ctx, e) == CAL_RESULT_PENDING) ;

    if(perfC)
      calCtxEndCounterExt(CContext[deviceID].ctx, CContext[deviceID].idleCounter);

    //Check the results
    unsigned *o0;
    calResMap((CALvoid**)&o0, &pitch, CContext[deviceID].outputRes0, 0);

    for(unsigned i=0; i<height; i++) {
      unsigned idx=i*pitch*4;
      for(unsigned j=0; j<width*4; j+=4) {
        if(o0[idx+j])           //partial match
        {
          rc5_72unitwork->check.count+=o0[idx+j]&0x0fffffff;
          if(cmp72(rc5_72unitwork->check.hi,rc5_72unitwork->check.mid,rc5_72unitwork->check.lo,
                   o0[idx+j+1],o0[idx+j+2],o0[idx+j+3])>0) {
            rc5_72unitwork->check.hi=o0[idx+j+1];
            rc5_72unitwork->check.mid=o0[idx+j+2];
            rc5_72unitwork->check.lo=o0[idx+j+3];
          }
          if(o0[idx+j]&0x80000000) {            //full match

            unsigned res=sub72(o0[idx+j+1],o0[idx+j+2],rc5_72unitwork->L0.hi,rc5_72unitwork->L0.mid);
            *iterations -= (kiter*4-res);

            rc5_72unitwork->check.hi=rc5_72unitwork->L0.hi=o0[idx+j+1];
            rc5_72unitwork->check.mid=rc5_72unitwork->L0.mid=o0[idx+j+2];
            rc5_72unitwork->check.lo=rc5_72unitwork->L0.lo=o0[idx+j+3];

            calResUnmap(CContext[deviceID].outputRes0);

            return RESULT_FOUND;
          }
        }
      }
    }
    calResUnmap(CContext[deviceID].outputRes0);

    if(iters==CContext[deviceID].maxIters)
    {
      if(perfC) {
        CALfloat idlePercentage;
        if (calCtxGetCounterExt(&idlePercentage, CContext[deviceID].ctx, CContext[deviceID].idleCounter) == CAL_RESULT_OK)
          if((idlePercentage>0.02f)||(idlePercentage<0.01f)) {
            float delta=(idlePercentage-0.017f)+1.f;
            CContext[deviceID].maxIters*=delta;
            if(CContext[deviceID].maxIters<2)
              CContext[deviceID].maxIters=2;
          }
      }
    }

    unsigned itersDone=(iters-1)*RunSize+rest;
    kiter-=itersDone;
    key_incr(&rc5_72unitwork->L0.hi,&rc5_72unitwork->L0.mid,&rc5_72unitwork->L0.lo,itersDone*4);
    itersNeeded-=itersDone;
  }
  return RESULT_NOTHING;
}
