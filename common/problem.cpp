/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -------------------------------------------------------------------
 * Eagleson's Law:
 *    Any code of your own that you haven't looked at for six or more
 *    months, might as well have been written by someone else.  (Eagleson
 *    is an optimist, the real number is more like three weeks.)
 * -------------------------------------------------------------------
*/
const char *problem_cpp(void) {
return "@(#)$Id: problem.cpp,v 1.119 1999/11/08 02:02:43 cyp Exp $"; }

/* ------------------------------------------------------------- */

#include "cputypes.h"
#include "baseincs.h"
#include "client.h"   //CONTEST_COUNT
#include "clitime.h"  //CliClock()
#include "logstuff.h" //LogScreen()
#include "probman.h"  //GetProblemPointerFromIndex()
#include "selcore.h"  //selcoreGetSelectedCoreForContest()
#include "cpucheck.h" //hardware detection
#include "console.h"  //ConOutErr
#include "triggers.h" //RaiseExitRequestTrigger()
#include "problem.h"  //ourselves
#if (CLIENT_OS == OS_RISCOS)
#include "../platforms/riscos/riscos_x86.h"
extern "C" void riscos_upcall_6(void);
#endif

//#define STRESS_THREADS_AND_BUFFERS /* !be careful with this! */

/* ------------------------------------------------------------- */

#ifdef PIPELINE_COUNT
#error Remove PIPELINE_COUNT from your makefile. It is useless (and 
#error confusing) when different cores have different PIPELINE_COUNTs
#error *Assign it* by core/cputype in LoadState() 
#error .
#error Ideally, the client should not need to know anything about the
#error number of pipelines in use by a core. Create a wrapper function
#error for that core and increment there.
#error .
#error The (projected) ideal prototype for *any* core (RC5/DES/OGR) is 
#error   s32 (*core_unit)( RC5unitWork *, u32 *timeslice, void *membuff );
#error Note that the core wrapper does its own timeslice to nbits 
#error conversion, increments the work space itself, stores the effective
#error timeslice back in the u32 *timeslice, and returns a result code.
#error (result code == RESULT_[FOUND|NOTHING|WORKING] or -1 if error).
#error .
#error Also note that the call is to a function pointer. That too should  
#error be assigned in LoadState() based on contest number and cputype. 
#error .
#error Would some kind user of the ANSI core(s) please rename the cores 
#error from [rc5|des]_unit_func to [rc5|des]_ansi_[x]_unit_func and fix
#error the RC5 cores to use timeslice as a second argument?
#error .
#error ./configure/ users: Please put your nick/cvs id and email address 
#error next to any rules you use in ./configure/.
#endif

#undef PIPELINE_COUNT

#if (CLIENT_CPU == CPU_X86)
  extern "C" u32 rc5_unit_func_486( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_p5( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_p6( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_6x86( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_k5( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_k6( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_p5_mmx( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_k6_mmx( RC5UnitWork * , u32 timeslice );
  extern "C" u32 rc5_unit_func_486_smc( RC5UnitWork * , u32 timeslice );
  extern u32 p1des_unit_func_p5( RC5UnitWork * , u32 nbbits );
  extern u32 p1des_unit_func_pro( RC5UnitWork * , u32 nbbits );
  extern u32 p2des_unit_func_p5( RC5UnitWork * , u32 nbbits );
  extern u32 p2des_unit_func_pro( RC5UnitWork * , u32 nbbits );
  extern u32 des_unit_func_mmx( RC5UnitWork * , u32 nbbits, char *coremem );
  extern u32 des_unit_func_slice( RC5UnitWork * , u32 nbbits );
#elif (CLIENT_OS == OS_AIX)     // this has to stay BEFORE CPU_POWERPC
  #if defined(_AIXALL) || (CLIENT_CPU == CPU_POWER)
  extern "C" s32 rc5_ansi_2_rg_unit_func( RC5UnitWork *rc5unitwork, u32 timeslice );
  #endif
  #if defined(_AIXALL) || (CLIENT_CPU == CPU_POWERPC)
  extern "C" s32 crunch_allitnil( RC5UnitWork *work, u32 iterations);
  extern "C" s32 crunch_lintilla( RC5UnitWork *work, u32 iterations);
  #endif

  extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
#elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_AIX) 
  #if (CLIENT_OS == OS_WIN32)   // NT PPC doesn't have good assembly
  extern u32 rc5_unit_func( RC5UnitWork *  ); //rc5ansi2-rg.cpp
  #else
  extern "C" s32 rc5_unit_func_g1( RC5UnitWork *work, u32 *timeslice /* , void *scratch_area */);
  extern "C" s32 rc5_unit_func_g2_g3( RC5UnitWork *work, u32 *timeslice /* , void *scratch_area */);
  #endif
  extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
#elif (CLIENT_CPU == CPU_68K)
  extern "C" __asm u32 rc5_unit_func_000_030
      ( register __a0 RC5UnitWork *, register __d0 unsigned long timeslice );
  extern "C" __asm u32 rc5_unit_func_040_060
      ( register __a0 RC5UnitWork *, register __d0 unsigned long timeslice );
  extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
#elif (CLIENT_CPU == CPU_ARM)
  extern "C" u32 rc5_unit_func_arm_1( RC5UnitWork * , unsigned long t);
  extern "C" u32 rc5_unit_func_arm_2( RC5UnitWork * , unsigned long t);
  extern "C" u32 rc5_unit_func_arm_3( RC5UnitWork * , unsigned long t);
  extern "C" u32 des_unit_func_arm( RC5UnitWork * , unsigned long t);
  extern "C" u32 des_unit_func_strongarm( RC5UnitWork * , unsigned long t);
#elif (CLIENT_CPU == CPU_PA_RISC)
  extern u32 rc5_unit_func( RC5UnitWork *  );
  extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
#elif (CLIENT_CPU == CPU_SPARC)
  #if (ULTRA_CRUNCH == 1)
  extern "C++" u32 crunch( register RC5UnitWork * , u32 timeslice);
  extern "C++" u32 des_unit_func( RC5UnitWork * , u32 timeslice );
  #else
  extern "C++" u32 rc5_unit_func( RC5UnitWork *  );
  extern "C++" u32 des_unit_func( RC5UnitWork * , u32 timeslice );
  #endif
  // CRAMER // #error Please verify these core prototypes
#elif (CLIENT_CPU == CPU_MIPS)
  #if (CLIENT_OS != OS_ULTRIX)
    #if (MIPS_CRUNCH == 1)
    extern "C++" u32 crunch( register RC5UnitWork * , u32 timeslice);
    extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
    #else
    extern u32 rc5_unit_func( RC5UnitWork *  );
    extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
    #endif
    //FOXYLOXY// #error Please verify these core prototypes
  #else /* OS_ULTRIX */
    extern u32 rc5_unit_func( RC5UnitWork *  );
    extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
  #endif
#elif (CLIENT_CPU == CPU_ALPHA)
  #if (CLIENT_OS == OS_WIN32)
     extern "C" u32 rc5_unit_func( RC5UnitWork *, unsigned long timeslice );
     extern "C" u32 des_unit_func_alpha_dworz( RC5UnitWork * , u32 nbbits );
  #elif (CLIENT_OS == OS_DEC_UNIX)
     #if defined(DEC_UNIX_CPU_SELECT)
       #include <machine/cpuconf.h>
       extern u32 rc5_alpha_osf_ev4( RC5UnitWork * );
       extern u32 rc5_alpha_osf_ev5( RC5UnitWork * );
       extern u32 des_alpha_osf_ev4( RC5UnitWork * , u32 timeslice );
       extern u32 des_alpha_osf_ev5( RC5UnitWork * , u32 timeslice );
     #else
       extern u32 des_unit_func( RC5UnitWork * , u32 timeslice );
     #endif
  #else
     extern "C" u32 rc5_unit_func_axp_bmeyer
         ( RC5UnitWork * , unsigned long iterations);   // note not u32
     extern "C" u32 des_unit_func_alpha_dworz
         ( RC5UnitWork * , u32 nbbits );
  #endif
#else
  extern u32 rc5_unit_func_ansi( RC5UnitWork * , u32 timeslice );
  extern u32 des_unit_func_ansi( RC5UnitWork * , u32 timeslice );
  #error RC5ANSICORE is disappearing. Please declare/prototype cores by CLIENT_CPU and assert PIPELINE_COUNT
#endif

/* ------------------------------------------------------------------- */
#ifdef HAVE_CSC_CORES
extern "C" {
s32 csc_unit_func_1k  ( RC5UnitWork *, u32 *timeslice, void *membuff );
s32 csc_unit_func_1k_i( RC5UnitWork *, u32 *timeslice, void *membuff );
s32 csc_unit_func_6b  ( RC5UnitWork *, u32 *timeslice, void *membuff );
s32 csc_unit_func_6b_i( RC5UnitWork *, u32 *timeslice, void *membuff );
}
#endif
/* ------------------------------------------------------------------- */
#ifdef HAVE_OGR_CORES
extern CoreDispatchTable *ogr_get_dispatch_table(void);
#endif
/* ------------------------------------------------------------------- */



Problem::Problem(long _threadindex /* defaults to -1L */)
{
  threadindex_is_valid = (_threadindex!=-1L);
  threadindex = ((threadindex_is_valid)?((unsigned int)_threadindex):(0));

  /* this next part is essential for alpha, but is probably beneficial to
     all platforms. If it fails for your os/cpu, we may need to redesign 
     how objects are allocated/how rc5unitwork is addressed, so let me know.
                                                       -cyp Jun 14 1999
  */
  RC5UnitWork *w = &rc5unitwork;
  unsigned long ww = ((unsigned long)w);
  #if (CLIENT_CPU == CPU_ALPHA) /* needs to be longword aligned */
  ww &= 0x7; /* (sizeof(longword)-1); */
  #else
  ww &= (sizeof(int)-1); /* int alignment */
  #endif        
  if (ww) 
  {
    Log("rc5unitwork for problem %d is misaligned!\n", threadindex);
    RaiseExitRequestTrigger();
    return;
  }  
//LogScreen("Problem created. threadindex=%u\n",threadindex);

  initialized = 0;
  started = 0;
  
#ifdef STRESS_THREADS_AND_BUFFERS 
  static int runlevel = 0;
  if (runlevel != -12345)
  {
    if ((++runlevel) != 1)
    {
      --runlevel;
      return;
    }
    RaisePauseRequestTrigger();
    LogScreen("Warning! STRESS_THREADS_AND_BUFFERS is defined.\n"
              "Are you sure that the client is pointing at\n"
              "a test proxy? If so, type 'yes': ");
    char getyes[10];
    ConInStr(getyes,4,0);
    ClearPauseRequestTrigger();
    if (strcmpi(getyes,"yes") != 0)
    {
      runlevel = +12345;
      RaiseExitRequestTrigger();
      return;
    }
    runlevel = -12345;
  }
#endif    
}

/* ------------------------------------------------------------------- */

Problem::~Problem()
{
  started = 0; // nothing to do. - suppress compiler warning
#if (CLIENT_OS == OS_RISCOS)
  if (GetProblemIndexFromPointer(this) == 1)
  {
    _kernel_swi_regs r;
    r.r[0] = 0;
    _kernel_swi(RC5PC_RetriveBlock,&r,&r);
    _kernel_swi(RC5PC_Off,&r,&r);
  }
#endif
}

/* ------------------------------------------------------------------- */

// for some odd reasons, the RC5 algorithm requires keys in reversed order
//         key.hi   key.lo
// ie key 01234567:89ABCDEF is sent to rc5_unit_func like that :
//        EFCDAB89:67452301
// This function switches from one format to the other.
//
// [Even if it looks like a little/big endian problem, it isn't. Whatever
//  endianess the underlying system has, we must swap every byte in the key
//  before sending it to rc5_unit_func()]
//
// Note that DES has a similiar but far more complex system, but everything
// is handled by des_unit_func().

static void  __SwitchRC5Format(u64 *_key)                               
{                                                                       
    register u32 tempkeylo = _key->hi; /* note: we switch the order */  
    register u32 tempkeyhi = _key->lo;                                  
                                                                        
    _key->lo =                                                          
      ((tempkeylo >> 24) & 0x000000FFL) |                               
      ((tempkeylo >>  8) & 0x0000FF00L) |                               
      ((tempkeylo <<  8) & 0x00FF0000L) |                               
      ((tempkeylo << 24) & 0xFF000000L);                                
    _key->hi =                                                          
      ((tempkeyhi >> 24) & 0x000000FFL) |                               
      ((tempkeyhi >>  8) & 0x0000FF00L) |                               
      ((tempkeyhi <<  8) & 0x00FF0000L) |                               
      ((tempkeyhi << 24) & 0xFF000000L);                                
}                                                                       

/* ------------------------------------------------------------------- */

// Input:  - an RC5 key in 'mangled' (reversed) format or a DES key
//         - an incrementation count
//         - a contest identifier (0==RC5 1==DES 2==OGR 3==CSC)
//
// Output: the key incremented

static void __IncrementKey(u64 *key, u32 iters, int contest)        
{                                                                   
  switch (contest)                                                  
  {                                                                 
    case RC5:
      __SwitchRC5Format (key);                                      
      key->lo += iters;                                             
      if (key->lo < iters) key->hi++;                               
      __SwitchRC5Format (key);                                      
      break;                                                        
    case DES:
    case CSC:
      key->lo += iters;                                             
      if (key->lo < iters) key->hi++; /* Account for carry */       
      break;                                                        
    case OGR:
      /* This should never be called for OGR */                     
      break;                                                        
  }                                                                 
}

/* ------------------------------------------------------------- */

u32 Problem::CalcPermille() /* % completed in the current block, to nearest 0.1%. */
{ 
  u32 retpermille = 0;
  if (initialized && last_resultcode >= 0)
  {
    if (!started)
      retpermille = startpermille;
    else if (last_resultcode != RESULT_WORKING)
      retpermille = 1000;
    else
    {
      switch (contest)
      {
        case RC5:
        case DES:
        case CSC:
                {
                retpermille = (u32)( ((double)(1000.0)) *
                (((((double)(contestwork.crypto.keysdone.hi))*((double)(4294967296.0)))+
                             ((double)(contestwork.crypto.keysdone.lo))) /
                ((((double)(contestwork.crypto.iterations.hi))*((double)(4294967296.0)))+
                             ((double)(contestwork.crypto.iterations.lo)))) ); 
                break;
                }
        case OGR:
                WorkStub curstub;
                ogr->getresult(ogrstate, &curstub, sizeof(curstub));
                // This is just a quick&dirty calculation that resembles progress.
                retpermille = curstub.stub.diffs[contestwork.ogr.workstub.stub.length]*10
                            + curstub.stub.diffs[contestwork.ogr.workstub.stub.length+1]/10;
                break;
      }
    }
    if (retpermille > 1000)
      retpermille = 1000;
  }
  return retpermille;
}

/* ------------------------------------------------------------------- */

static int __core_picker(Problem *problem, unsigned int contestid)
{                               /* must return a valid core selection # */
  int coresel;
  problem->pipeline_count = 2; /* most cases */

  coresel = selcoreGetSelectedCoreForContest( contestid );
  if (coresel < 0)
    return -1;

  if (contestid == RC5) /* avoid switch */
  {
    #if (CLIENT_CPU == CPU_ARM)
    if (coresel == 0)
    {
      problem->rc5_unit_func = rc5_unit_func_arm_1;
      problem->pipeline_count = 1;
    }
    else if (coresel == 1)
    {
      problem->rc5_unit_func = rc5_unit_func_arm_2;
      problem->pipeline_count = 2;
    }
    else /* (coresel == 2, default) */
    {
      coresel = 2;
      problem->rc5_unit_func = rc5_unit_func_arm_3;
      problem->pipeline_count = 3;
    }
    #elif (CLIENT_CPU == CPU_68K)
    {
      if (coresel < 0 || coresel > 5) /* just to be safe */
        coresel = 0;
      if (coresel == 4 || coresel == 5 ) // there is no 68050, so type5=060
        problem->rc5_unit_func = rc5_unit_func_040_060;
      else //if (coresel == 0 || coresel == 1 || coresel == 2 || coresel == 3)
        problem->rc5_unit_func = rc5_unit_func_000_030;
      problem->pipeline_count = 2;
    }
    #elif (CLIENT_CPU == CPU_ALPHA)
    {
      problem->pipeline_count = 2;
      #if (CLIENT_OS == OS_DEC_UNIX)
      if (coresel == 1) /* EV5, EV56, PCA56, EV6 */
        problem->rc5_unit_func = rc5_alpha_osf_ev5;
      else // EV3_CPU, EV4_CPU, LCA4_CPU, EV45_CPU and default
        problem->rc5_unit_func = rc5_alpha_osf_ev4; 
      #elif (CLIENT_OS == OS_WIN32)
        problem->rc5_unit_func = ::rc5_unit_func;
      #else
        problem->rc5_unit_func = rc5_unit_func_axp_bmeyer;
      #endif
    }
    #elif (CLIENT_OS == OS_AIX)
    {
      static int detectedtype = -1;
      if (detectedtype == -1)
        detectedtype = GetProcessorType(1 /* quietly */);
      #if defined(_AIXALL) || (CLIENT_CPU == CPU_POWERPC)
      switch (detectedtype) 
      {
         case 1:                  // PPC 601
            coresel = 1;
            problem->rc5_unit_func = crunch_allitnil;
            problem->pipeline_count = 1;
            break;
         case 2:                  // other PPC
            coresel = 2;
            problem->rc5_unit_func = crunch_lintilla;
            problem->pipeline_count = 1;
            break;
         case 0:                  // that's POWER
         default:
            coresel = 0;
            #ifdef _AIXALL
            problem->rc5_unit_func = rc5_ansi_2_rg_unit_func ;
            problem->pipeline_count = 2;
            #else                 // no POWER support
            problem->rc5_unit_func = crunch_allitnil;
            problem->pipeline_count = 1;
            #endif
            break;
      } /* endswitch */
      #elif (CLIENT_CPU == CPU_POWER)
      problem->rc5_unit_func = rc5_ansi_2_rg_unit_func;
      problem->pipeline_count = 2;
      coresel = 0;
      #else
      #error "Systemtype not supported"
      #endif
    }
    #elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_AIX)
    {
      problem->rc5_unit_func = rc5_unit_func_g2_g3;
      problem->pipeline_count = 1;
      #if ((CLIENT_OS != OS_BEOS) || (CLIENT_OS != OS_AMIGAOS))
      if (coresel == 0)
        problem->rc5_unit_func = rc5_unit_func_g1;
      #endif
    }
    #elif (CLIENT_CPU == CPU_X86)
    {
      static int detectedtype = -1;
      if (detectedtype == -1)
        detectedtype = GetProcessorType(1 /* quietly */);
      if (coresel < 0 || coresel > 5)
        coresel = 0;
      problem->pipeline_count = 2; /* most cases */
      if (coresel == 1)   // Intel 386/486
      {
        problem->x86_unit_func = rc5_unit_func_486;
        #if defined(SMC) 
        if (problem->threadindex == 0)
          problem->x86_unit_func =  rc5_unit_func_486_smc;
        #endif
      }
      else if (coresel == 2) // Ppro/PII
        problem->x86_unit_func = rc5_unit_func_p6;
      else if (coresel == 3) // 6x86(mx)
        problem->x86_unit_func = rc5_unit_func_6x86;
      else if (coresel == 4) // K5
        problem->x86_unit_func = rc5_unit_func_k5;
      else if (coresel == 5) // K6/K6-2/K7
      {
        problem->x86_unit_func = rc5_unit_func_k6;
        #if defined(MMX_RC5_AMD)
        if ((detectedtype & 0x100) != 0)
        { 
          problem->x86_unit_func = rc5_unit_func_k6_mmx;
          problem->pipeline_count = 4;
        }
        #endif
      }
      else // Pentium (0/6) + others
      {
        problem->x86_unit_func = rc5_unit_func_p5;
        #if defined(MMX_RC5)
        if ((detectedtype & 0x100) != 0)
        { 
          problem->x86_unit_func = rc5_unit_func_p5_mmx;
          problem->pipeline_count = 4; // RC5 MMX core is 4 pipelines
        }
        #endif
        coresel = 0;
      }
    }
    #endif
    return coresel;
  }
  
  #ifdef HAVE_DES_CORES
  if (contestid == DES)
  {
    #if (CLIENT_CPU == CPU_ARM)
    {
      if (coresel == 0)
        problem->des_unit_func = des_unit_func_arm;
      else /* (coresel == 1, default) */
      {
        problem->des_unit_func = des_unit_func_strongarm;
        coresel = 1;
      }
    }
    #elif (CLIENT_CPU == CPU_ALPHA) 
    {
      #if (CLIENT_OS == OS_DEC_UNIX)
      if (coresel == 1) /* EV5, EV56, PCA56, EV6 */
        problem->des_unit_func = des_alpha_osf_ev5;
      else // EV3_CPU, EV4_CPU, LCA4_CPU, EV45_CPU and default
        problem->des_unit_func = des_alpha_osf_ev4;
      #elif (CLIENT_OS == OS_WIN32)
        problem->des_unit_func = des_unit_func_alpha_dworz;
      #else
        problem->des_unit_func = des_unit_func_alpha_dworz;
      #endif
    }
    #elif (CLIENT_CPU == CPU_X86)
    {
      u32 (*slicit)(RC5UnitWork *,u32) = ((u32 (*)(RC5UnitWork *,u32))0);
                        #if defined(CLIENT_SUPPORTS_SMP)
                          slicit = des_unit_func_slice; //kwan
      #endif
      #if defined(MMX_BITSLICER) 
      {
        static long detectedtype = -1;
        if (detectedtype == -1)
          detectedtype = GetProcessorType(1 /* quietly */);
        if ((detectedtype & 0x100) != 0) 
          slicit = (u32 (*)(RC5UnitWork *,u32))des_unit_func_mmx;
      }
      #endif  
                        if (slicit && coresel > 1) /* not standard bryd and not ppro bryd */
      {                /* coresel=2 is valid only if we have a slice core */
        coresel = 2;
        problem->x86_unit_func = slicit;
      }
      else 
      {
        #if defined(CLIENT_SUPPORTS_SMP) 
        // bryd is not thread safe, so make sure that when 
        // running benchmark/test asychronously (ie from a gui), 
        // we pick a core that isn't in use.
        unsigned int thrindex = problem->threadindex;
        if (thrindex == 0 && !problem->threadindex_is_valid)
        { /* !threadindex_is_valid==not probman controlled==benchmark/test*/
          while (GetProblemPointerFromIndex(thrindex))
            thrindex++;
        }
        #endif
        if (coresel == 1) /* movzx bryd */
        {
          problem->x86_unit_func = p1des_unit_func_pro;
          #if defined(CLIENT_SUPPORTS_SMP) 
          if (thrindex > 0)  /* not first thread */
          {
            if (thrindex == 1)  /* second thread */
              problem->x86_unit_func = p2des_unit_func_pro;
            else if (thrindex == 2) /* third thread */
              problem->x86_unit_func = p1des_unit_func_p5;
            else if (thrindex == 3) /* fourth thread */
              problem->x86_unit_func = p2des_unit_func_p5;
            else                    /* fifth...nth thread */
              problem->x86_unit_func = slicit;
          }
          #endif /* if defined(CLIENT_SUPPORTS_SMP)  */
        }
        else             /* normal bryd */
        {
          coresel = 0;
          problem->x86_unit_func = p1des_unit_func_p5;
          #if defined(CLIENT_SUPPORTS_SMP) 
          if (thrindex > 0)  /* not first thread */
          {
            if (thrindex == 1)  /* second thread */
              problem->x86_unit_func = p2des_unit_func_p5;
            else if (thrindex == 2) /* third thread */
              problem->x86_unit_func = p1des_unit_func_pro;
            else if (thrindex == 3) /* fourth thread */
              problem->x86_unit_func = p2des_unit_func_pro;
            else                    /* fifth...nth thread */
              problem->x86_unit_func = slicit;
          }
          #endif /* if defined(CLIENT_SUPPORTS_SMP)  */
        }
      }
    }
    #endif
    return coresel;
  }
  #endif /* #ifdef HAVE_DES_CORES */

  #if defined(HAVE_OGR_CORES)
  if (contestid == OGR)
  {
    return 0;
  }
  #endif

  #ifdef HAVE_CSC_CORES
  if( contestid == CSC ) // CSC
  {
    problem->pipeline_count = 1;
    problem->unit_func = csc_unit_func_1k_i; /* default */
    switch( coresel ) 
    {
      case 0 : problem->unit_func = csc_unit_func_6b_i;
               break;
      case 1 : problem->unit_func = csc_unit_func_6b;
               break;
      default: coresel = 2;
      case 2 : problem->unit_func = csc_unit_func_1k_i;
               break;
      case 3 : problem->unit_func = csc_unit_func_1k;
               break;
    }
    return coresel;
  }
  #endif /* #ifdef HAVE_CSC_CORES */

  return -1; /* core selection failed */
}

/* ------------------------------------------------------------------- */

int Problem::LoadState( ContestWork * work, unsigned int contestid, 
                              u32 _timeslice, int /* was _cputype */ )
{
  unsigned int sz = sizeof(int);

  if (sz < sizeof(u32)) /* need to do it this way to suppress compiler warnings. */
  {
    LogScreen("FATAL: sizeof(int) < sizeof(u32)\n");
    //#error "everything assumes a 32bit CPU..."
    RaiseExitRequestTrigger();
    return -1;
  }
  if (!IsProblemLoadPermitted(threadindex, contestid))
    return -1;

  last_resultcode = -1;
  started = initialized = 0;
  timehi = timelo = 0;
  runtime_sec = runtime_usec = 0;
  last_runtime_sec = last_runtime_usec = 0;
  memset((void *)&profiling, 0, sizeof(profiling));
  startpermille = permille = 0;
  loaderflags = 0;
  contest = contestid;
  tslice = _timeslice;
  coresel = __core_picker(this, contestid );
  if (coresel < 0 || 
     (coresel > 0 && coresel != selcoreValidateCoreIndex(contestid, coresel)))
    return -1;

  //----------------------------------------------------------------

  switch (contest) 
  {
    case RC5:
    case DES:
    case CSC: // HAVE_CSC_CORES

      // copy over the state information
      contestwork.crypto.key.hi = ( work->crypto.key.hi );
      contestwork.crypto.key.lo = ( work->crypto.key.lo );
      contestwork.crypto.iv.hi = ( work->crypto.iv.hi );
      contestwork.crypto.iv.lo = ( work->crypto.iv.lo );
      contestwork.crypto.plain.hi = ( work->crypto.plain.hi );
      contestwork.crypto.plain.lo = ( work->crypto.plain.lo );
      contestwork.crypto.cypher.hi = ( work->crypto.cypher.hi );
      contestwork.crypto.cypher.lo = ( work->crypto.cypher.lo );
      contestwork.crypto.keysdone.hi = ( work->crypto.keysdone.hi );
      contestwork.crypto.keysdone.lo = ( work->crypto.keysdone.lo );
      contestwork.crypto.iterations.hi = ( work->crypto.iterations.hi );
      contestwork.crypto.iterations.lo = ( work->crypto.iterations.lo );

      //determine starting key number. accounts for carryover & highend of keysdone
      u64 key;
      key.hi = contestwork.crypto.key.hi + contestwork.crypto.keysdone.hi + 
         ((((contestwork.crypto.key.lo & 0xffff) + (contestwork.crypto.keysdone.lo & 0xffff)) + 
           ((contestwork.crypto.key.lo >> 16) + (contestwork.crypto.keysdone.lo >> 16))) >> 16);
      key.lo = contestwork.crypto.key.lo + contestwork.crypto.keysdone.lo;

      // set up the unitwork structure
      rc5unitwork.plain.hi = contestwork.crypto.plain.hi ^ contestwork.crypto.iv.hi;
      rc5unitwork.plain.lo = contestwork.crypto.plain.lo ^ contestwork.crypto.iv.lo;
      rc5unitwork.cypher.hi = contestwork.crypto.cypher.hi;
      rc5unitwork.cypher.lo = contestwork.crypto.cypher.lo;

      rc5unitwork.L0.lo = key.lo;
      rc5unitwork.L0.hi = key.hi;
      if (contest == RC5)
        __SwitchRC5Format (&(rc5unitwork.L0));

      refL0 = rc5unitwork.L0;

      if (contestwork.crypto.keysdone.lo!=0 || contestwork.crypto.keysdone.hi!=0 )
      {
        startpermille = (u32)( ((double)(1000.0)) *
        (((((double)(contestwork.crypto.keysdone.hi))*((double)(4294967296.0)))+
                           ((double)(contestwork.crypto.keysdone.lo))) /
        ((((double)(contestwork.crypto.iterations.hi))*((double)(4294967296.0)))+
                        ((double)(contestwork.crypto.iterations.lo)))) );
      }     
      break;

    case OGR:

      #if !defined(HAVE_OGR_CORES)
      return -1;
      #else
      contestwork.ogr = work->ogr;
      contestwork.ogr.nodes.lo = 0;
      contestwork.ogr.nodes.hi = 0;
      ogr = ogr_get_dispatch_table();
      int r = ogr->init();
      if (r != CORE_S_OK)
        return -1;
      r = ogr->create(&contestwork.ogr.workstub, sizeof(WorkStub), ogrstate, sizeof(ogrstate));
      if (r != CORE_S_OK)
        return -1;
      if (contestwork.ogr.workstub.worklength > contestwork.ogr.workstub.stub.length)
      {
        // This is just a quick&dirty calculation that resembles progress.
        startpermille = contestwork.ogr.workstub.stub.diffs[contestwork.ogr.workstub.stub.length]*10
                      + contestwork.ogr.workstub.stub.diffs[contestwork.ogr.workstub.stub.length+1]/10;
      }
      break;
      #endif

  }

  //---------------------------------------------------------------
#if (CLIENT_OS == OS_RISCOS)
  if (threadindex == 1 /*x86 thread*/)
  {
    RC5PCstruct rc5pc;
    _kernel_oserror *err;
    _kernel_swi_regs r;
  
    rc5pc.key.hi = contestwork.key.hi;
    rc5pc.key.lo = contestwork.key.lo;
    rc5pc.iv.hi = contestwork.iv.hi;
    rc5pc.iv.lo = contestwork.iv.lo;
    rc5pc.plain.hi = contestwork.plain.hi;
    rc5pc.plain.lo = contestwork.plain.lo;
    rc5pc.cypher.hi = contestwork.cypher.hi;
    rc5pc.cypher.lo = contestwork.cypher.lo;
    rc5pc.keysdone.hi = contestwork.keysdone.hi;
    rc5pc.keysdone.lo = contestwork.keysdone.lo;
    rc5pc.iterations.hi = contestwork.iterations.hi;
    rc5pc.iterations.lo = contestwork.iterations.lo;
    rc5pc.timeslice = tslice;
  
    err = _kernel_swi(RC5PC_On,&r,&r);
    if (err)
      LogScreen("Failed to start x86 card");
    else
    {
      r.r[1] = (int)&rc5pc;
      err = _kernel_swi(RC5PC_AddBlock,&r,&r);
      if ((err) || (r.r[0] == -1))
      {
        LogScreen("Failed to add block to x86 cruncher\n");
      }
    }
  }
#endif

  last_resultcode = RESULT_WORKING;
  initialized = 1;

  return( 0 );
}

/* ------------------------------------------------------------------- */

int Problem::RetrieveState( ContestWork * work, unsigned int *contestid, int dopurge )
{
  if (!initialized)
    return -1;
  if (work) // store back the state information
  {
    switch (contest) {
      case RC5:
      case DES:
      case CSC:
        // nothing special needs to be done here
        break;
      case OGR:
        ogr->getresult(ogrstate, &contestwork.ogr.workstub, sizeof(WorkStub));
        break;
    }
    memcpy( (void *)work, (void *)&contestwork, sizeof(ContestWork));
  }
  if (contestid)
    *contestid = contest;
  if (dopurge)
    initialized = 0;
  if (last_resultcode < 0)
    return -1;
  return ( last_resultcode );
}

/* ------------------------------------------------------------- */

u32 rc5_singlestep_core_wrapper( RC5UnitWork * rc5unitwork, u32 timeslice,
                int pipeline_count, auto u32 (*unit_func)( RC5UnitWork *) )
{                                
  u32 kiter = 0;
  int keycount = timeslice;
  //LogScreenf ("rc5unitwork = %08X:%08X (%X)\n", rc5unitwork.L0.hi, rc5unitwork.L0.lo, keycount);
  while ( keycount-- ) // timeslice ignores the number of pipelines
  {
    u32 result = (*unit_func)( rc5unitwork );
    if ( result )
    {
      kiter += result-1;
      break;
    }
    else
    {
      // "mangle-increment" the key number by the number of pipelines
      __IncrementKey (&(rc5unitwork->L0), pipeline_count, 0 );
      kiter += pipeline_count;
    }
  }
  return kiter;
}  

/* ------------------------------------------------------------- */

int Problem::Run_RC5(u32 *timesliceP, int *resultcode)
{
  u32 kiter = 0;
  u32 timeslice = *timesliceP;

  // align the timeslice to an even-multiple of pipeline_count and 2 
  u32 alignfact = pipeline_count + (pipeline_count & 1);
  timeslice = ((timeslice + (alignfact - 1)) & ~(alignfact - 1));

  // don't allow a too large of a timeslice be used ie (>(iter-keysdone)) 
  // (technically not necessary, but may save some wasted time)
  if (contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi)
  {
    u32 todo = contestwork.crypto.iterations.lo-contestwork.crypto.keysdone.lo;
    if (todo < timeslice)
    {
      timeslice = todo;
      timeslice = ((timeslice + (alignfact - 1)) & ~(alignfact - 1));
    }
  }

#if 0
LogScreen("alignTimeslice: effective timeslice: %lu (0x%lx),\n"
          "suggested timeslice: %lu (0x%lx)\n"
          "pipeline_count = %lu, timeslice%%pipeline_count = %lu\n", 
          (unsigned long)timeslice, (unsigned long)timeslice,
          (unsigned long)tslice, (unsigned long)tslice,
          pipeline_count, timeslice%pipeline_count );
#endif

  timeslice /= pipeline_count;

  #if (CLIENT_CPU == CPU_X86)
    kiter = (*x86_unit_func)( &rc5unitwork, timeslice );
  #elif ((CLIENT_CPU == CPU_SPARC) && (ULTRA_CRUNCH == 1)) || \
        ((CLIENT_CPU == CPU_MIPS) && (MIPS_CRUNCH == 1)) 
    kiter = crunch( &rc5unitwork, timeslice );
  #elif (CLIENT_CPU == CPU_68K) || (CLIENT_OS == OS_AIX) || \
        (CLIENT_CPU == CPU_POWER)
    kiter = (*rc5_unit_func)( &rc5unitwork, timeslice );
  #elif (CLIENT_CPU == CPU_POWERPC)
    kiter = timeslice;
    *resultcode = (*rc5_unit_func)( &rc5unitwork, &kiter );
  #elif (CLIENT_CPU == CPU_ARM)
    kiter = rc5_unit_func(&rc5unitwork, timeslice);
  #elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
    kiter = (timeslice * pipeline_count) - 
      rc5_unit_func(&rc5unitwork,timeslice);
  #elif (CLIENT_CPU == CPU_ALPHA)
    kiter = rc5_unit_func(&rc5unitwork, timeslice);
  #else
    kiter = rc5_singlestep_core_wrapper( &rc5unitwork, timeslice,
                pipeline_count, rc5_unit_func );
  #endif

  // Mac OS needs to yield here, since yielding works differently
  // depending on the core
  #if (CLIENT_OS == OS_MACOS)
    if (MP_active == 0) YieldToMain(1);
  #endif

  timeslice *= pipeline_count;
  *timesliceP = timeslice;

  __IncrementKey (&refL0, timeslice, contest);
    // Increment reference key count

  if (((refL0.hi != rc5unitwork.L0.hi) ||  // Compare ref to core
      (refL0.lo != rc5unitwork.L0.lo)) &&  // key incrementation
      (kiter == timeslice))
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #23: Please contact help@distributed.net\n"
        "Debug Information: %08x:%08x - %08x:%08x\n",
        rc5unitwork.L0.hi, rc5unitwork.L0.lo, refL0.hi, refL0.lo);
    #endif
    *resultcode = -1;
    return -1;
  };

  contestwork.crypto.keysdone.lo += kiter;
  if (contestwork.crypto.keysdone.lo < kiter)
    contestwork.crypto.keysdone.hi++;
    // Checks passed, increment keys done count.

  if (kiter < timeslice)
  {
    // found it!
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo) 
      contestwork.crypto.key.hi++; // wrap occured ?
    *resultcode = RESULT_FOUND;
    return RESULT_FOUND;
  }
  else if (kiter != timeslice)
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #24: Please contact help@distributed.net\n"
        "Debug Information: k: %x t: %x\n"
        "Debug Information: %08x:%08x - %08x:%08x\n", kiter, timeslice,
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1;
    return -1;
  };

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
       ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
       ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING;    // Done with this round
}  

/* ------------------------------------------------------------- */

int Problem::Run_CSC(u32 *timesliceP, int *resultcode)
{
#ifndef HAVE_CSC_CORES
  timesliceP = timesliceP;
  *resultcode = -1;
  return -1;
#else  
  s32 rescode = (*unit_func)( &rc5unitwork, timesliceP, core_membuffer );

  if (rescode < 0) /* "kiter" error */
  {
    *resultcode = -1;
    return -1;
  }
  *resultcode = (int)rescode;

  // Increment reference key count
  __IncrementKey (&refL0, *timesliceP, contest);

  // Compare ref to core key incrementation
  if ((refL0.hi != rc5unitwork.L0.hi) || (refL0.lo != rc5unitwork.L0.lo))
  { 
    #ifdef DEBUG_CSC_CORE /* can you spell "thread safe"? */
    Log("CSC incrementation mismatch:\n"
        "expected %08x:%08x, got %08x:%08x\n",
        refL0.lo, refL0.hi, rc5unitwork.L0.lo, rc5unitwork.L0.hi );
    #endif
    *resultcode = -1;
    return -1;
  }

  // Checks passed, increment keys done count.
  contestwork.crypto.keysdone.lo += *timesliceP;
  if (contestwork.crypto.keysdone.lo < *timesliceP)
    contestwork.crypto.keysdone.hi++;

  // Update data returned to caller
  if (*resultcode == RESULT_FOUND)
  {
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo) 
      contestwork.crypto.key.hi++; // wrap occured ?
    return RESULT_FOUND;
  }

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
       ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
       ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
  {
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }
  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING; // Done with this round
#endif  
}

/* ------------------------------------------------------------- */

int Problem::Run_DES(u32 *timesliceP, int *resultcode)
{
#ifndef HAVE_DES_CORES
  *timesliceP = 0;  /* no keys done */
  *resultcode = -1; /* core error */
  return -1;
#else
  u32 kiter = 0;
  u32 timeslice = *timesliceP;
  
  #if (CLIENT_CPU == CPU_X86)
  u32 min_bits = 8;  /* bryd and kwan cores only need a min of 256 */
  u32 max_bits = 24; /* these are the defaults if !MEGGS && !DES_ULTRA */

  #if defined(MMX_BITSLICER)
  if (((u32 (*)(RC5UnitWork *,u32, char *))(x86_unit_func) == des_unit_func_mmx))
  {
    #if defined(BITSLICER_WITH_LESS_BITS)
    min_bits = 16;
    #else
    min_bits = 20;
    #endif
    max_bits = min_bits; /* meggs driver has equal MIN and MAX */
  }
  #endif
  u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

  if (nbits < min_bits) nbits = min_bits;
  else if (nbits > max_bits) nbits = max_bits;
  timeslice = (1ul << nbits);

  #if defined(MMX_BITSLICER)
  if (((u32 (*)(RC5UnitWork *,u32, char *))(x86_unit_func) == des_unit_func_mmx))
    kiter = des_unit_func_mmx( &rc5unitwork, nbits, core_membuffer );
  else
  #endif
  kiter = (*x86_unit_func)( &rc5unitwork, nbits );
  #elif (CLIENT_CPU == CPU_ALPHA) && (CLIENT_OS == OS_WIN32)
  u32 nbits = 20;  // FROM des-slice-dworz.cpp
  timeslice = (1ul << nbits);
  kiter = des_unit_func ( &rc5unitwork, nbits );
  #else
  u32 nbits=1; while (timeslice > (1ul << nbits)) nbits++;

  if (nbits < MIN_DES_BITS) nbits = MIN_DES_BITS;
  else if (nbits > MAX_DES_BITS) nbits = MAX_DES_BITS;
  timeslice = (1ul << nbits);

  kiter = des_unit_func ( &rc5unitwork, nbits );
  #endif

  *timesliceP = timeslice;

  __IncrementKey (&refL0, timeslice, contest);
  // Increment reference key count

  if (((refL0.hi != rc5unitwork.L0.hi) ||  // Compare ref to core
      (refL0.lo != rc5unitwork.L0.lo)) &&  // key incrementation
      (kiter == timeslice))
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #23: Please contact help@distributed.net\n"
        "Debug Information: %08x:%08x - %08x:%08x\n",
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1;
    return -1;
  };

  contestwork.crypto.keysdone.lo += kiter;
  if (contestwork.crypto.keysdone.lo < kiter)
    contestwork.crypto.keysdone.hi++;
    // Checks passed, increment keys done count.

  // Update data returned to caller
  if (kiter < timeslice)
  {
    // found it!
    u32 keylo = contestwork.crypto.key.lo;
    contestwork.crypto.key.lo += contestwork.crypto.keysdone.lo;
    contestwork.crypto.key.hi += contestwork.crypto.keysdone.hi;
    if (contestwork.crypto.key.lo < keylo) 
      contestwork.crypto.key.hi++; // wrap occured ?
    *resultcode = RESULT_FOUND;
    return RESULT_FOUND;
  }
  else if (kiter != timeslice)
  {
    #if 0 /* can you spell "thread safe"? */
    Log("Internal Client Error #24: Please contact help@distributed.net\n"
        "Debug Information: k: %x t: %x\n"
        "Debug Information: %08x:%08x - %08x:%08x\n", kiter, timeslice,
        rc5unitwork.L0.lo, rc5unitwork.L0.hi, refL0.lo, refL0.hi);
    #endif
    *resultcode = -1; /* core error */
    return -1;
  };

  if ( ( contestwork.crypto.keysdone.hi > contestwork.crypto.iterations.hi ) ||
     ( ( contestwork.crypto.keysdone.hi == contestwork.crypto.iterations.hi ) &&
     ( contestwork.crypto.keysdone.lo >= contestwork.crypto.iterations.lo ) ) )
  {
    // done with this block and nothing found
    *resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
  }

  // more to do, come back later.
  *resultcode = RESULT_WORKING;
  return RESULT_WORKING; // Done with this round
#endif /* #ifdef HAVE_DES_CORES */
}

/* ------------------------------------------------------------- */

int Problem::Run_OGR(u32 *timesliceP, int *resultcode)
{
#if !defined(HAVE_OGR_CORES)
  timesliceP = timesliceP;
#else
  int r, nodes;

  if (*timesliceP > 0x100000UL)
    *timesliceP = 0x100000UL;

  nodes = (int)(*timesliceP);
  r = ogr->cycle(ogrstate, &nodes);
  *timesliceP = (u32)nodes;

  u32 newnodeslo = contestwork.ogr.nodes.lo + nodes;
  if (newnodeslo < contestwork.ogr.nodes.lo) {
    contestwork.ogr.nodes.hi++;
  }
  contestwork.ogr.nodes.lo = newnodeslo;

  switch (r) 
  {
    case CORE_S_OK:
    {
      r = ogr->destroy(ogrstate);
      if (r == CORE_S_OK) 
      {
        *resultcode = RESULT_NOTHING;
        return RESULT_NOTHING;
      }
      break;
    }
    case CORE_S_CONTINUE:
    {
      *resultcode = RESULT_WORKING;
      return RESULT_WORKING;
    }
    case CORE_S_SUCCESS:
    {
      if (ogr->getresult(ogrstate, &contestwork.ogr.workstub, sizeof(WorkStub)) == CORE_S_OK)
      {
        //Log("OGR Success!\n");
        contestwork.ogr.workstub.stub.length = 
                  (u16)(contestwork.ogr.workstub.worklength);
        *resultcode = RESULT_FOUND;
        return RESULT_FOUND;
      }
      break;
    }
  }
  /* Something bad happened */
#endif
 *resultcode = -1; /* this will cause the problem to be discarded */
 return -1;
}

/* ------------------------------------------------------------- */

int Problem::Run(void) /* returns RESULT_*  or -1 */
{
  struct timeval stop, start;
  int retcode, core_resultcode;
  u32 timeslice;

  if ( !initialized )
    return ( -1 );

  if ( last_resultcode != RESULT_WORKING ) /* _FOUND, _NOTHING or -1 */
    return ( last_resultcode );
    
  CliClock(&start);
  if (!started)
  {
    timehi = start.tv_sec; timelo = start.tv_usec;
    runtime_sec = runtime_usec = 0;
    memset((void *)&profiling, 0, sizeof(profiling));
    started=1;

#ifdef STRESS_THREADS_AND_BUFFERS 
    contest = RC5;
    contestwork.crypto.key.hi = contestwork.crypto.key.lo = 0;
    contestwork.crypto.keysdone.hi = contestwork.crypto.iterations.hi;
    contestwork.crypto.keysdone.lo = contestwork.crypto.iterations.lo;
    runtime_usec = 1; /* ~1Tkeys for a 2^20 packet */
    last_resultcode = RESULT_NOTHING;
    return RESULT_NOTHING;
#endif    
  }

  /* 
    On return from the Run_XXX contestwork must be in a state that we
    can put away to disk - that is, do not expect the loader (probfill 
    et al) to fiddle with iterations or key or whatever.
    
    The Run_XXX functions do *not* update problem.last_resultcode, they use
    core_resultcode instead. This is so that members of the problem object
    that are updated after the resultcode has been set will not be out of
    sync when the main thread gets it with RetrieveState(). 
    
    note: although the value returned by Run_XXX is usually the same as 
    the core_resultcode it is not always so. For instance, if 
    post-LoadState() initialization  failed, but can be deferred, Run_XXX 
    may choose to return -1, but keep core_resultcode at RESULT_WORKING.
  */

  timeslice = tslice;
  last_runtime_usec = last_runtime_sec = 0;
  core_resultcode = last_resultcode;
  retcode = -1;

  switch (contest)
  {
    case RC5: retcode = Run_RC5( &timeslice, &core_resultcode );
              break;
    case DES: retcode = Run_DES( &timeslice, &core_resultcode );
              break;
    case OGR: retcode = Run_OGR( &timeslice, &core_resultcode );
              break;
    case CSC: retcode = Run_CSC( &timeslice, &core_resultcode );
              break;
    default: retcode = core_resultcode = last_resultcode = -1;
       break;
  }

  
  if (retcode < 0) /* don't touch tslice or runtime as long as < 0!!! */
  {
    return -1;
  }
  
  core_run_count++;
  CliClock(&stop);
  if ( core_resultcode != RESULT_WORKING ) /* _FOUND, _NOTHING */
  {
    if (((u32)(stop.tv_sec)) > ((u32)(timehi)))
    {
      u32 tmpdif = timehi - stop.tv_sec;
      tmpdif = (((tmpdif >= runtime_sec) ?
        (tmpdif - runtime_sec) : (runtime_sec - tmpdif)));
      if ( tmpdif < core_run_count )
      {
        runtime_sec = runtime_usec = 0;
        start.tv_sec = timehi;
        start.tv_usec = timelo;
      }
    }
  }
  if (stop.tv_sec < start.tv_sec || 
     (stop.tv_sec == start.tv_sec && stop.tv_usec <= start.tv_usec))
  {
    //AIEEE! clock is whacky (or unusably inaccurate if ==)
  }
  else
  {
    if (stop.tv_usec < start.tv_usec)
    {
      stop.tv_sec--;
      stop.tv_usec+=1000000L;
    }
    runtime_usec += (last_runtime_usec = (stop.tv_usec - start.tv_usec));
    runtime_sec  += (last_runtime_sec = (stop.tv_sec - start.tv_sec));
    if (runtime_usec > 1000000L)
    {
      runtime_sec++;
      runtime_usec-=1000000L;
    }
  }

  tslice = timeslice;

  last_resultcode = core_resultcode;
  return last_resultcode;
}

/* ----------------------------------------------------------------------- */

int IsProblemLoadPermitted(long prob_index, unsigned int contest_i)
{
  prob_index = prob_index; /* possibly unused */
  switch (contest_i)
  {
    case RC5: 
    {
      #if (CLIENT_OS == OS_RISCOS) /* RISC OS x86 thread only supports RC5 */
      if (prob_index == 1 && contest_i != RC5)
        return 0;
      #endif
      return 1;
    }
    case DES:
    {
      #ifdef HAVE_DES_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case OGR:
    {
      #ifdef HAVE_OGR_CORES
      return 1;
      #else
      return 0;
      #endif
    }
    case CSC:
    {
      #ifdef HAVE_CSC_CORES
      return 1;
      #else
      return 0;
      #endif
    }
  }
  return 0;
}
