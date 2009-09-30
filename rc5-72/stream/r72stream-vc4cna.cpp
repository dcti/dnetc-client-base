/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-vc4cna.cpp,v 1.5 2009/09/30 16:41:05 sla Exp $
*/

#include "r72stream-common.h"
#include "r72stream-vc4nag_il.cpp"

static bool init_rc5_72_il4a_nand(u32 Device)
{
  if(CContext[Device].coreID!=CORE_IL4NA)
    AMDStreamReinitializeDevice(Device);

  if(!CContext[Device].active)
  {
	Log("Thread %u: Device is not supported\n", Device);
	return false;
  } else{
    switch(CContext[Device].attribs.target) {
    case CAL_TARGET_600:
      CContext[Device].domainSizeX=56;
      CContext[Device].domainSizeY=56;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_610:
      CContext[Device].domainSizeX=40;
      CContext[Device].domainSizeY=40;
      CContext[Device].maxIters=10;
      break;
    case CAL_TARGET_630:
      CContext[Device].domainSizeX=24;
      CContext[Device].domainSizeY=24;
      CContext[Device].maxIters=350;
      break;
    case CAL_TARGET_670:
      CContext[Device].domainSizeX=32;
      CContext[Device].domainSizeY=32;
      CContext[Device].maxIters=300;
      break;
    case CAL_TARGET_7XX:
      //TODO:domainSize
      break;
    case CAL_TARGET_770:
      CContext[Device].domainSizeX=600;
      CContext[Device].domainSizeY=600;
      CContext[Device].maxIters=5;
      break;
    case CAL_TARGET_710:
      //TODO:domainSize
      CContext[Device].domainSizeX=80;
      CContext[Device].domainSizeY=80;
      CContext[Device].maxIters=128;
      break;
    case CAL_TARGET_730:
      CContext[Device].domainSizeX=120;
      CContext[Device].domainSizeY=120;
      CContext[Device].maxIters=30;
      break;
    case 8:	//RV870
      CContext[Device].domainSizeX=728;
      CContext[Device].domainSizeY=728;
      CContext[Device].maxIters=4;
      break;
    default:
      break;
    }
  }

  CALresult result;
  result=calCtxCreate(&CContext[Device].ctx, CContext[Device].device);
  if(result!=CAL_RESULT_OK)
  {
    LogScreen("Thread %u: creating context failed! Reason:%u\n",Device,result);
    return false;
  }

  CContext[Device].globalRes0=0;
  if(CContext[Device].attribs.memExport) {
    calResAllocRemote2D(&CContext[Device].globalRes0, &CContext[Device].device, 1, 64,
                        1, CAL_FORMAT_UINT_1, CAL_RESALLOC_GLOBAL_BUFFER);
  }
  
  //-------------------------------------------------------------------------
  // Compiling Device Program
  //-------------------------------------------------------------------------
  result=compileProgram(&CContext[Device].ctx,&CContext[Device].image,&CContext[Device].module0,
                        (CALchar *)il4ag_nand_src,CContext[Device].attribs.target,CContext[Device].globalRes0);
		
  if ( result!= CAL_RESULT_OK)
  {
    Log("Core compilation failed. Exiting.\n");
    return false;
  }

  //-------------------------------------------------------------------------
  // Allocating and initializing resources
  //-------------------------------------------------------------------------

 // Input and output resources
  CContext[Device].outputRes0=0;
  if(CContext[Device].attribs.cachedRemoteRAM>0)
    calResAllocRemote2D(&CContext[Device].outputRes0, &CContext[Device].device, 1, CContext[Device].domainSizeX,
                        CContext[Device].domainSizeY, CAL_FORMAT_UINT_1, CAL_RESALLOC_CACHEABLE);

  if(!CContext[Device].outputRes0) {
    if(calResAllocRemote2D(&CContext[Device].outputRes0, &CContext[Device].device, 1, CContext[Device].domainSizeX,
                           CContext[Device].domainSizeY, CAL_FORMAT_UINT_1, 0)!=CAL_RESULT_OK)
    {
      Log("Failed to allocate output buffer\n");
      return false;
    }
  }

  // Constant resource
  if(calResAllocRemote1D(&CContext[Device].constRes0, &CContext[Device].device, 1, 3, CAL_FORMAT_UINT_4, 0)!=CAL_RESULT_OK)
  {
    Log("Failed to allocate constants buffer\n");
    return false;
  }

  // Mapping output resource to CPU and initializing values
  // Getting memory handle from resources
  result=calCtxGetMem(&CContext[Device].outputMem0, CContext[Device].ctx, CContext[Device].outputRes0);
  if(result==CAL_RESULT_OK)
	result=calCtxGetMem(&CContext[Device].constMem0, CContext[Device].ctx, CContext[Device].constRes0);
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to map resources!\n");
    return false;
  }

  // Defining entry point for the module
  result=calModuleGetEntry(&CContext[Device].func0, CContext[Device].ctx, CContext[Device].module0, "main");
  if(result==CAL_RESULT_OK) {
    result=calModuleGetName(&CContext[Device].outName0, CContext[Device].ctx, CContext[Device].module0, "o0");
      if(result==CAL_RESULT_OK)
        result=calModuleGetName(&CContext[Device].constName0, CContext[Device].ctx, CContext[Device].module0, "cb0");
  }
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to get entry points!\n");
    return false;
  }

  if(CContext[Device].globalRes0) {
    result=calCtxGetMem(&CContext[Device].globalMem0, CContext[Device].ctx, CContext[Device].globalRes0);
    if(result==CAL_RESULT_OK) {
      result=calModuleGetName(&CContext[Device].globalName0, CContext[Device].ctx, CContext[Device].module0, "g[]");
      if(result==CAL_RESULT_OK)
        result=calCtxSetMem(CContext[Device].ctx, CContext[Device].globalName0, CContext[Device].globalMem0);
    }
    if(result!=CAL_RESULT_OK)
    {
      Log("Failed to allocate global buffer!\n");
      return false;
    }
  }

  // Setting input and output buffers
  // used in the kernel
  result=calCtxSetMem(CContext[Device].ctx, CContext[Device].outName0, CContext[Device].outputMem0);
  if(result==CAL_RESULT_OK)
    result=calCtxSetMem(CContext[Device].ctx, CContext[Device].constName0, CContext[Device].constMem0);
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to set buffers!\n");
    return false;
  }

  CContext[Device].coreID=CORE_IL4NA;

  return true;
}

#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_il4a_nand (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
#endif

static bool FillConstantBuffer(CALresource res, u32 runsize, u32 iters, u32 rest, float width, 
							   RC5_72UnitWork *rc5_72unitwork, u32 keyIncrement) 
{
  u32* constPtr = NULL;
  CALuint pitch = 0;

  if(calResMap((CALvoid**)&constPtr, &pitch, res, 0)!=CAL_RESULT_OK)
    return false;

  u32 hi,mid,lo;
  hi=rc5_72unitwork->L0.hi;
  mid=rc5_72unitwork->L0.mid;
  lo=rc5_72unitwork->L0.lo;

  key_incr(&hi,&mid,&lo,keyIncrement*4);

  //cb0[0]					//key_hi,key_mid,key_lo,granularity
  constPtr[0]=hi;
  constPtr[1]=mid;
  constPtr[2]=lo;
  constPtr[3]=runsize*4;

  //cb0[1]					//plain_lo,plain_hi,cypher_lo,cypher_hi
  constPtr[4]=rc5_72unitwork->plain.lo;
  constPtr[5]=rc5_72unitwork->plain.hi;
  constPtr[6]=rc5_72unitwork->cypher.lo;
  constPtr[7]=rc5_72unitwork->cypher.hi;

  //cb0[2]					//iters,rest,width
  constPtr[8]=iters;
  constPtr[9]=rest;
  float *f;
  f=(float*)&constPtr[10]; *f=width;

  if(calResUnmap(res)!=CAL_RESULT_OK)
    return false;
  return true;
}

static s32 ReadResultsFromGPU(CALresource res, CALresource globalRes, u32 width, u32 height, RC5_72UnitWork *rc5_72unitwork, u32 *CMC, u32 *iters_done)
{
  u32 *o0, *g0;
  CALuint pitch = 0;
  bool found=true;
    
  if(globalRes) {
    CALuint result;
    if(calResMap((CALvoid**)&g0, &pitch, globalRes, 0)!=CAL_RESULT_OK)
      return -1;
    result=g0[0];
    g0[0]=0;
    if(calResUnmap(globalRes)!=CAL_RESULT_OK)
      return -1;
    if(result==0)
      found=false;
  }
	
  if(calResMap((CALvoid**)&o0, &pitch, res, 0)!=CAL_RESULT_OK) {
    return -1;
  }

  u32 last_CMC=0;
  *iters_done=(o0[0]&0x7e000000)>>25;
  if(found)
    for(u32 i=0; i<height; i++) {
      u32 idx=i*pitch;
      for(u32 j=0; j<width; j++) {
        if(o0[idx+j]&0x1ffffff)           //partial match
        {
          u32 output=o0[idx+j];
          u32 CMC_count=(output&0x1ffffff)>>18;
          u32 CMC_iter=(((output>>2)&0x0000ffff)-1)*width*height;
          u32 CMC_hit=(CMC_iter+i*width+j)*4+(output&0x00000003);

          //  LogScreen("Partial match found\n");
          u32 hi,mid,lo;
          hi=rc5_72unitwork->L0.hi;
          mid=rc5_72unitwork->L0.mid;
          lo=rc5_72unitwork->L0.lo;

          key_incr(&hi,&mid,&lo,CMC_hit);
          if(last_CMC<=CMC_hit) {
            rc5_72unitwork->check.hi=hi;
            rc5_72unitwork->check.mid=mid;
            rc5_72unitwork->check.lo=lo;
            last_CMC=CMC_hit;
          }

          rc5_72unitwork->check.count+=CMC_count;

          if(output&0x80000000) {            //full match

            rc5_72unitwork->L0.hi=hi;
            rc5_72unitwork->L0.mid=mid;
            rc5_72unitwork->L0.lo=lo;

            calResUnmap(res);

            *CMC=CMC_hit;
            return 1;
          }
        }
      }
    }
    if(calResUnmap(res)!=CAL_RESULT_OK) {
      return -1;
    }
  return 0;
}

s32 rc5_72_unit_func_il4a_nand(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  u32 deviceID=rc5_72unitwork->threadnum;
//  u32 iter_c=*iterations;
//  RC5_72UnitWork rc5_72unitwork_c;
//  memcpy(&rc5_72unitwork_c,rc5_72unitwork,sizeof(RC5_72UnitWork));

  if (CContext[deviceID].coreID!=CORE_IL4NA)
  {
    init_rc5_72_il4a_nand(deviceID);
    if(CContext[deviceID].coreID!=CORE_IL4NA) {
      RaiseExitRequestTrigger();
      return -1;        //arr
    }
  }

  u32 kiter =(*iterations)/4;

  u32 itersNeeded=kiter;
  u32 width=CContext[deviceID].domainSizeX;
  u32 height=CContext[deviceID].domainSizeY;
  u32 RunSize=width*height;

  CALevent e0 = 0;
  u32 iters0=1;
  u32 rest0=0;

//#define VERBOSE 1

#ifdef VERBOSE
  LogScreen("Tread %u: %u ITERS (%u), maxiters=%u\n",deviceID, kiter,kiter/RunSize,CContext[deviceID].maxIters);
#endif
  double fr_d=HiresTimerGetResolution();
  hirestimer_type cstart, cend;

  //Clear global buffer
  if(CContext[deviceID].globalRes0) {
    u32* gPtr = NULL;
    CALuint pitch = 0;
    if(calResMap((CALvoid**)&gPtr, &pitch, CContext[deviceID].globalRes0, 0)==CAL_RESULT_OK)
    {
      gPtr[0]=0;
      calResUnmap(CContext[deviceID].globalRes0);
    }else
    {
      Log("Failed to map global buffer!\n");
      RaiseExitRequestTrigger();
      return -1;        //err
    }
  }

  CALresult result;
  while(itersNeeded) {
    //Make sure there is no overflow in core output
    if(CContext[deviceID].maxIters>65535)
      CContext[deviceID].maxIters=65535;

      iters0=itersNeeded/RunSize;
      if(iters0>=CContext[deviceID].maxIters) {
        iters0=CContext[deviceID].maxIters;
        rest0=RunSize;
      } else  {
        rest0=itersNeeded-iters0*RunSize;
        iters0++;
      }
      itersNeeded-=(iters0-1)*RunSize+rest0;

      //fill constant buffer
      if(!FillConstantBuffer(CContext[deviceID].constRes0,RunSize, iters0, rest0, (float)width, rc5_72unitwork,0))
      {
//      NonPolledUSleep(50000);  //Lost connection to driver
//      *iterations=iter_c;
//      memcpy(rc5_72unitwork,&rc5_72unitwork_c,sizeof(RC5_72UnitWork));

//      return RESULT_WORKING;
        Log("Internal error!\n");
        RaiseExitRequestTrigger();
        return -1;        //err
      }

      CALdomain domain = {0, 0, width, height};
      result=calCtxRunProgram(&e0, CContext[deviceID].ctx, CContext[deviceID].func0, &domain);
      if((result!=CAL_RESULT_OK)&&(result!=CAL_RESULT_PENDING))
      {
        Log("Error running GPU program\n");
        RaiseExitRequestTrigger();
        return -1;        //err
      }
		
      // Checking whether the execution of the program is complete or not
      HiresTimerGet(&cstart);

      u32 busy_c=0;
      if(iters0!=CContext[deviceID].maxIters)
        busy_c=2;
      CALresult result;
      while((result=calCtxIsEventDone(CContext[deviceID].ctx, e0)) == CAL_RESULT_PENDING) {
        if(!busy_c)
          NonPolledUSleep(15000);	//15ms 			
        busy_c++;
      }
      if(result!=CAL_RESULT_OK)
      {
        Log("Error waiting for GPU program to finish!\n");
        RaiseExitRequestTrigger();
        return -1;        //err
      }
      HiresTimerGet(&cend);
      double d=HiresTimerDiff(cend, cstart)/fr_d;
#ifdef VERBOSE 
      LogScreen("Thread %u: Time %lf ms, c=%u\n",deviceID,(double)(cend-cstart)/fr_d, busy_c);
#endif
      if((d>15.5)&&(busy_c>1))	
      {
        u32 delta;
        if(d>60.)
          delta=(u32)CContext[deviceID].maxIters*0.3f;
        else
          delta=(u32)CContext[deviceID].maxIters*0.1f;
        if(delta==0)
          delta=1;
        if(delta>=CContext[deviceID].maxIters)
          CContext[deviceID].maxIters=1;
        else
          CContext[deviceID].maxIters-=delta;

#ifdef VERBOSE
        LogScreen("Thread %u:Busy_c=%u, delta=%u\n",deviceID,busy_c,delta);
#endif
      }else
        if((busy_c<=1)&&(d<15.5))
        {
          u32 delta;
          delta=(u32)CContext[deviceID].maxIters*0.02f;
          if(delta==0)
            delta=1;
          CContext[deviceID].maxIters+=delta;
#ifdef VERBOSE
          LogScreen("Thread %u:idle, delta=%u\n",deviceID,busy_c,delta);
#endif
        } 
        //Check the results
        u32 CMC, iters_finished;
        s32 read_res=ReadResultsFromGPU(CContext[deviceID].outputRes0, CContext[deviceID].globalRes0, width, height, rc5_72unitwork, &CMC, &iters_finished);
        if (read_res==1) {
          *iterations -= (kiter*4-CMC);
          return RESULT_FOUND;
        }
       if (read_res<0)
       {
//			NonPolledUSleep(50000);		//Lost connection to driver
//			*iterations=iter_c;
//			memcpy(rc5_72unitwork,&rc5_72unitwork_c,sizeof(RC5_72UnitWork));

//			return RESULT_WORKING;
         Log("Internal error!\n");
         RaiseExitRequestTrigger();
         return -1;        //err
       }
       if(iters_finished!=((iters0-(rest0==0))&0x3f) /*6 lower bits*/)	//Something bad happend during program execution
       {
         Log("GPU: unexpected program stop!\n");
         Log("Expected: %x, got:%x!\n",iters0,iters_finished);
         RaiseExitRequestTrigger();
         return -1;        //err
       }
       
       unsigned itersDone=(iters0-1)*RunSize+rest0;
       kiter-=itersDone;
       key_incr(&rc5_72unitwork->L0.hi,&rc5_72unitwork->L0.mid,&rc5_72unitwork->L0.lo,itersDone*4);
    }
	
  /* tell the client about the optimal timeslice increment for this core 
     (with current parameters) */
  rc5_72unitwork->optimal_timeslice_increment = RunSize*4*CContext[deviceID].maxIters;
  return RESULT_NOTHING;
}
