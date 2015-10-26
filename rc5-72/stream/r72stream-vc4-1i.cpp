/*
 * Copyright 2008-2010 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
*/

#include "r72stream-common.h"
#include "r72stream-vc4-1i-src.cpp"

//63 elements
#define GLOBAL_B_WIDTH 256 
#define MAX_DOMAIN_SIZE 4096

//rc5-72 test
#define P 0xB7E15163
#define Q 0x9E3779B9

#define SHL(x, s) ((u32) ((x) << ((s) & 31)))
#define SHR(x, s) ((u32) ((x) >> (32 - ((s) & 31))))
#define ROTL(x, s) ((u32) (SHL((x), (s)) | SHR((x), (s))))
#define ROTL3(x) ROTL(x, 3)


static inline u32 swap32(u32 a)
{
  u32 t=(a>>24)|(a<<24);
  t|=(a&0x00ff0000)>>8;
  t|=(a&0x0000ff00)<<8;
  return t;
}

static s32 rc5_72_unit_func_ansi_ref (RC5_72UnitWork *rc5_72unitwork)
{
  u32 i, j, k;
  u32 A, B;
  u32 S[26];
  u32 L[3];
  u32 kiter = 1;
  while (kiter--)
  {
    L[2] = rc5_72unitwork->L0.hi;
    L[1] = rc5_72unitwork->L0.mid;
    L[0] = rc5_72unitwork->L0.lo;
    for (S[0] = P, i = 1; i < 26; i++)
      S[i] = S[i-1] + Q;
      
    for (A = B = i = j = k = 0;
         k < 3*26; k++, i = (i + 1) % 26, j = (j + 1) % 3)
    {
      A = S[i] = ROTL3(S[i]+(A+B));
      B = L[j] = ROTL(L[j]+(A+B),(A+B));
    }
    A = rc5_72unitwork->plain.lo + S[0];
    B = rc5_72unitwork->plain.hi + S[1];
    for (i=1; i<=12; i++)
    {
      A = ROTL(A^B,B)+S[2*i];
      B = ROTL(B^A,A)+S[2*i+1];
    }
    if (A == rc5_72unitwork->cypher.lo)
    {
        return RESULT_FOUND;
    }
  }
  return RESULT_NOTHING;
}

static bool init_rc5_72_il4_1i(stream_context_t *cont)
{
  if(cont->coreID!=CORE_IL4_1I)
    AMDStreamReinitializeDevice(cont);

  if(!cont->active)
  {
    Log("Thread %u: Device is not supported\n", cont->clientDeviceNo);
    return false;
  } else{
    switch(cont->attribs.target) {
    case 8: //RV870
      cont->domainSizeX=1280;
      cont->domainSizeY=1280;
      cont->maxIters=1;
      break;
    case 9: //RV840
      cont->domainSizeX=656;
      cont->domainSizeY=656;
      cont->maxIters=1;
      break;
    case 15: //Cayman
      cont->domainSizeX=1024;
      cont->domainSizeY=1024;
      cont->maxIters=1;
    case 17: //Barts
      cont->domainSizeX=904;
      cont->domainSizeY=904;
      cont->maxIters=1;
    case 20: //Tahiti
      cont->domainSizeX=2048;
      cont->domainSizeY=2048;
      cont->maxIters=1;
    default:
      cont->domainSizeX=512;
      cont->domainSizeY=512;
      cont->maxIters=1;
      break;
    }
  }

  CALresult result;
  result=calCtxCreate(&cont->ctx, cont->device);
  if (ati_verbose(result, "creating context", cont) != CAL_RESULT_OK)
    return false;

  cont->globalRes0=0;
  calResAllocRemote1D(&cont->globalRes0, &cont->device, 1, GLOBAL_B_WIDTH,
                        CAL_FORMAT_UINT_1, CAL_RESALLOC_GLOBAL_BUFFER);
  
  if(!cont->globalRes0)
  {
    Log("Failed to allocate UAV buffer. Reason:%s\n", calGetErrorString());
    return false;
  }

  //-------------------------------------------------------------------------
  // Compiling Device Program
  //-------------------------------------------------------------------------
  result=compileProgram(&cont->ctx,&cont->image,&cont->module0,
                        (CALchar *)il4_1i_src, cont->attribs.target, (cont->globalRes0 != 0), cont);

  if ( result!= CAL_RESULT_OK)
  {
    Log("Core compilation failed. Exiting.\n");
    return false;
  }

  //-------------------------------------------------------------------------
  // Allocating and initializing resources
  //-------------------------------------------------------------------------

  // Constant resource
  if(calResAllocRemote1D(&cont->constRes0, &cont->device, 1, 3, CAL_FORMAT_UINT_4, 0)!=CAL_RESULT_OK)
  {
    Log("Failed to allocate constants buffer\n");
    return false;
  }

  // Mapping output resource to CPU and initializing values
  // Getting memory handle from resources
  result=calCtxGetMem(&cont->constMem0, cont->ctx, cont->constRes0);
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to map resources!\n");
    return false;
  }

  // Defining entry point for the module
  result=calModuleGetEntry(&cont->func0, cont->ctx, cont->module0, "main");
  if(result==CAL_RESULT_OK)
    result=calModuleGetName(&cont->constName0, cont->ctx, cont->module0, "cb0");
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to get entry points!\n");
    return false;
  }

  result=calCtxGetMem(&cont->globalMem0, cont->ctx, cont->globalRes0);
  if(result==CAL_RESULT_OK) {
    result=calModuleGetName(&cont->globalName0, cont->ctx, cont->module0, "uav0");
    if(result==CAL_RESULT_OK)
      result=calCtxSetMem(cont->ctx, cont->globalName0, cont->globalMem0);
  }
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to allocate UAV buffer! Reason: %s\n",calGetErrorString());
    return false;
  }

  // Setting input and output buffers
  // used in the kernel
  result=calCtxSetMem(cont->ctx, cont->constName0, cont->constMem0);
  if(result!=CAL_RESULT_OK)
  {
    Log("Failed to set buffers!\n");
    return false;
  }

  cont->USEcount=0;
  cont->coreID=CORE_IL4_1I;

  return true;
}

#ifdef __cplusplus
extern "C" s32 rc5_72_unit_func_il4_1i (RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *);
#endif

static bool FillConstantBuffer(CALresource res, RC5_72UnitWork *rc5_72unitwork)
{
  u32* constPtr = NULL;
  CALuint pitch = 0;

  if(calResMap((CALvoid**)&constPtr, &pitch, res, 0)!=CAL_RESULT_OK)
    return false;

  //cb0[0]					//key_hi,key_mid,key_lo,granularity
  constPtr[0]=rc5_72unitwork->L0.hi;
  constPtr[1]=swap32(rc5_72unitwork->L0.mid);
  constPtr[2]=rc5_72unitwork->L0.lo;
  constPtr[3]=ROTL(0xBF0A8B1D+rc5_72unitwork->L0.lo,0x1d); //L0=ROTL(L0+S0,S0)=ROTL(L0+S0,0x1d)

  //cb0[1]					//plain_lo,plain_hi,cypher_lo,cypher_hi
  constPtr[4]=rc5_72unitwork->plain.lo;
  constPtr[5]=rc5_72unitwork->plain.hi;
  constPtr[6]=rc5_72unitwork->cypher.lo;
  constPtr[7]=rc5_72unitwork->cypher.hi;

  //cb0[2]
  constPtr[8]=ROTL3(constPtr[3]+0xBF0A8B1D+0x5618cb1c);	//S1=ROTL3(Sc1+S0+L0)

  if(calResUnmap(res)!=CAL_RESULT_OK)
    return false;
  return true;
}

//a<=b?
static bool cmp72(u32 a_h, u32 a_m, u32 a_l, u32 b_h, u32 b_m, u32 b_l)
{
  u32 al=swap32(a_l);
  u32 bl=swap32(b_l);
  if(al<bl)
    return true;
  if(al==bl)
  {
    u32 am=swap32(a_m);
    u32 bm=swap32(b_m);
    if(am<bm)
      return true;
    if(am==bm)
    {
      if(a_h<=b_h)
        return true;
    }
  }
  return false;
}

static s32 ReadResultsFromGPU(CALresource globalRes, RC5_72UnitWork *rc5_72unitwork, u32 *CMC, u32 keysToDo)
{
  CALuint *g0;
  CALuint pitch = 0;
  s32 found=0;

  CALuint result;
  if(calResMap((CALvoid**)&g0, &pitch, globalRes, 0)!=CAL_RESULT_OK)
    return -1;

  result=g0[0];
  //Ќадо провер€ть на случай если уехали дальше, чем заказывали
  //ѕервый проход - ищем fullmatch в диапазоне
  //¬торой проход - собираем CMC
  g0[0]=0;

  if(result)
  {
    u32 idx=1;
    u32 lim_hi=rc5_72unitwork->L0.hi; 
    u32 lim_mid=rc5_72unitwork->L0.mid;
    u32 lim_lo=rc5_72unitwork->L0.lo;

    key_incr(&lim_hi, &lim_mid, &lim_lo, keysToDo);

    for(u32 i=0; i<result; i++)
    {
      if(cmp72(g0[idx], g0[idx+1], g0[idx+2], lim_hi, lim_mid, lim_lo))
      {
        if(g0[idx+3]&0x80000000)
        {
          lim_hi=g0[idx];lim_mid=g0[idx+1];lim_lo=g0[idx+2];
          break;
        }
      }
      idx+=4;
    }

    idx=1;
    for(u32 i=0; i<result; i++)
    {
      u32 hi,mid,lo;
      hi=g0[idx]; mid=g0[idx+1]; lo=g0[idx+2];
      u32 flag=g0[idx+3];

      if(cmp72(hi, mid, lo, lim_hi, lim_mid, lim_lo))
      {
        u32 lastCMC=flag&3;
        u32 CMCc=(flag>>2)&3;
        hi+=lastCMC;
        
        //update cmc data
        rc5_72unitwork->check.hi=hi;
        rc5_72unitwork->check.mid=mid;
        rc5_72unitwork->check.lo=lo;
        rc5_72unitwork->check.count+=CMCc;
        
        if(flag&0x80000000)
        {

	  RC5_72UnitWork t;

          //calculate the number of processed keys
          *CMC=sub72(mid, hi, rc5_72unitwork->L0.mid, rc5_72unitwork->L0.hi);

          rc5_72unitwork->L0.hi=hi;
          rc5_72unitwork->L0.mid=mid;
          rc5_72unitwork->L0.lo=lo;
	  memcpy(&t,rc5_72unitwork,sizeof(RC5_72UnitWork));
          
	  if(rc5_72_unit_func_ansi_ref(&t)!=RESULT_FOUND)
	  {
	    Log("WARNING!!! False positive detected!\n");
	    Log("Debug info: %x:%x:%x\n",hi,mid,lo);
	    RaiseExitRequestTrigger();
	    return -1;
	  }

          found=1;
        }
      }
      idx+=4;
    }
  }
  if(calResUnmap(globalRes)!=CAL_RESULT_OK)
    return -1;

  return found;
}

s32 rc5_72_unit_func_il4_1i(RC5_72UnitWork *rc5_72unitwork, u32 *iterations, void *)
{
  stream_context_t *cont = stream_get_context(rc5_72unitwork->devicenum);
  RC5_72UnitWork tmp_unit;

  if (cont == NULL)
  {
    RaiseExitRequestTrigger();
    return -1;
  }

  if (cont->coreID!=CORE_IL4_1I)
  {
    init_rc5_72_il4_1i(cont);
    if(cont->coreID!=CORE_IL4_1I) {
      RaiseExitRequestTrigger();
      return -1;
    }
  }

  if(checkRemoteConnectionFlag())
  {
    NonPolledUSleep(500*1000);  //sleep 0.5 sec
    *iterations=0;
    return RESULT_WORKING;
  }
  if(cont->coreID==CORE_NONE)
  {
    *iterations=0;
    return RESULT_WORKING;
  }

  memcpy(&tmp_unit, rc5_72unitwork, sizeof(RC5_72UnitWork));

  u32 kiter =(*iterations)/4;

  u32 itersNeeded=kiter;

  CALevent e0 = 0;
  u32 rest0=0;

  //Clear global buffer
  u32* gPtr = NULL;
  CALuint pitch = 0;
  if(calResMap((CALvoid**)&gPtr, &pitch, cont->globalRes0, 0)==CAL_RESULT_OK)
  {
    gPtr[0]=0;
    calResUnmap(cont->globalRes0);
  }else
  {
    if(setRemoteConnectionFlag()) {
      *iterations=0;
      return RESULT_WORKING;
    }
    Log("Failed to map UAV buffer!\n");
    RaiseExitRequestTrigger();
    return -1;          //err
  }

  double fr_d=HiresTimerGetResolution();
  hirestimer_type cstart, cend;

  CALresult result;
  u32 GPUiters=0;
  u32 RunSize=cont->domainSizeX*cont->domainSizeY;
  while(itersNeeded) {
	RunSize=cont->domainSizeX*cont->domainSizeY;
	  
	if(itersNeeded>=RunSize)
      rest0=RunSize;
    else
      rest0=itersNeeded;

    itersNeeded-=rest0;

    //fill constant buffer
    if(!FillConstantBuffer(cont->constRes0, rc5_72unitwork))
    {
      if(setRemoteConnectionFlag()) {
        memcpy(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
        *iterations=0;
        return RESULT_WORKING;
      }
      Log("Internal error!\n");
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    CALprogramGrid g_calProgramGrid;
    g_calProgramGrid.func             = cont->func0;
    g_calProgramGrid.flags            = 0;
    g_calProgramGrid.gridBlock.width  = 64; //needs to be = thread group size as given in IL kernel.
    g_calProgramGrid.gridBlock.height = 1;
    g_calProgramGrid.gridBlock.depth  = 1;
    g_calProgramGrid.gridSize.width   = (rest0 + 
      g_calProgramGrid.gridBlock.width - 1) / g_calProgramGrid.gridBlock.width;
    g_calProgramGrid.gridSize.height  = 1;
    g_calProgramGrid.gridSize.depth   = 1;

    result = calCtxRunProgramGrid(&e0, cont->ctx, &g_calProgramGrid);
    if((result!=CAL_RESULT_OK)&&(result!=CAL_RESULT_PENDING))
    {
      if(setRemoteConnectionFlag()) {
        memcpy(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
        *iterations=0;
        return RESULT_WORKING;
      }
      Log("Error running GPU program\n");
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    // Checking whether the execution of the program is complete or not

    CALresult result;
    HiresTimerGet(&cstart);
    if(isCalCtxWaitForEventsSupported)  //normal case
      result=calCtxWaitForEvents(cont->ctx, &e0, 1, 0);
    else
      while((result=calCtxIsEventDone(cont->ctx, e0)) == CAL_RESULT_PENDING) 
        NonPolledUSleep(15000);  //15ms

    if(result!=CAL_RESULT_OK)
    {
      if(setRemoteConnectionFlag()) {
        memcpy(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
        *iterations=0;
        return RESULT_WORKING;
      }
      Log("Error while waiting for GPU program to finish!\n");
      RaiseExitRequestTrigger();
      return -1;          //err
    }

    HiresTimerGet(&cend);
    double d=HiresTimerDiff(cend, cstart)/fr_d;
    if(isCalCtxWaitForEventsSupported)
    {
      if(d>12.)
	  {
        if(cont->domainSizeX>8)
          cont->domainSizeX-=8;
        if(cont->domainSizeY>8)
          cont->domainSizeY-=8;
	  }
      if(d<8.)
	  {
        if(cont->domainSizeX<MAX_DOMAIN_SIZE)
          cont->domainSizeX+=8;
        if(cont->domainSizeY<MAX_DOMAIN_SIZE)
          cont->domainSizeY+=8;
	  }
    }

    cont->USEcount=0;	//Reset Unexpected Stop Error counter
    u32 itersDone=rest0;
    kiter-=itersDone;
    key_incr(&rc5_72unitwork->L0.hi,&rc5_72unitwork->L0.mid,&rc5_72unitwork->L0.lo,itersDone*4);
    GPUiters+=itersDone;
  }

  //Check results
  u32 CMC, iters_finished;

  memcpy(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
  s32 read_res=ReadResultsFromGPU(cont->globalRes0, rc5_72unitwork, &CMC, GPUiters*4);
  if (read_res==1) {
    *iterations = CMC;
    return RESULT_FOUND;
  }
  key_incr(&rc5_72unitwork->L0.hi,&rc5_72unitwork->L0.mid,&rc5_72unitwork->L0.lo, GPUiters*4);
  if (read_res<0)
  {
    if(setRemoteConnectionFlag()) {
      memcpy(rc5_72unitwork, &tmp_unit, sizeof(RC5_72UnitWork));
      *iterations=0;
      return RESULT_WORKING;
    }
    Log("Internal error!\n");
    RaiseExitRequestTrigger();
    return -1;
  }

  /* tell the client about the optimal timeslice increment for this core
     (with current parameters) */
  rc5_72unitwork->optimal_timeslice_increment = RunSize*4;
  return RESULT_NOTHING;
}
