/* #define DEBUG */
/*
 * This file contains startup code similar to what crt0 does elsewhere.
 * Written August 1997 Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwpre3x.c,v 1.1.2.1 2001/01/21 15:10:31 cyp Exp $
 *
 * Unlike Novell's prelude, its in source, is backwards compatible with 
 * NetWare 3.x, zeroes the BSS segment, and supports Watcom c/cpp lib 
 * startup/exit code.
 *
 * Unlike Watcom's prelude, its in source and it doesn't have half a ton 
 * of gunk in it to get most of the pseudo-static clib pulled in.
 * (This prelude helps make Watcom's static lib (plib[mt]3s) unneccesary)
 * ***************************************************************
 * THIS MUST BE COMPILED WITH WATCOM C IF LINKING WITH WATCOM LIBs 
 * ***************************************************************
 *
 * An NLM's startup/shutdown sequence is something like this:
 * 
 * (All function names in quotes are in prelude. "_Prelude" and "_Stop" are 
 * simply the default names and may be redefined with the appropriate linker 
 * command. Everything else (except functionality marked "(OS)") is in CLIB)
 *
 * The sequence DestroyThreads->DestroyThreadGroup->atexit->AtUnload as
 * depicted below is what it says in the API. I personally believe that
 * atexit procedures are called before the thread group is destroyed.
 *
 *       NLM Loader (OS)                      NLM Unloader (OS) KillMe() (OS)
 *        |         |                              |     |          |  |
 *       "_Prelude(...)"                          "_Stop()"         |  |
 *        |         |                              |     |          ^  ^
 *       _StartNLM(...)                           _TerminateNLM()   |  |
 *           |                                     |          |     |  |
 *    (BeginThreadGroup())                         |          |     |  |    
 *           |                                     |          |     |  | 
 *           |   ,-----exit(),------------------, raise(6);   |     |  |
 *           |   |     |     |                  |  |          ^     ^  ^
 *      "_cstart_()"   |     `--<--, ,-> (DestroyThreads)     |     |  |
 *        |      |     `-----<----,| |    |     |  |          |     |  |
 *        |      |                || |    |  (flushbuffers)   |     |  |
 *        |      `---<---,        || | (DestroyThreadGroup)   ^     ^  ^
 *      "main()"         |        || |    |     |  |          |     |  |
 *        |              |        || |    |    atexit         |     |  |
 *        `-(by return)--'        || |    |     |  |          |     |  |
 *         ---->- ExitThread() ->-'| |    |     |  `-AtUnload-'     |  |
 *         ---->- exit() ------->--' |    |     `-------------------'  |
 *         ---->- _exit() ----->-----'    `----------------------------'
 * 
 *
 * however, I have modified the flow slightly. exit() as well as atexit()
 * are in prelude rather than in CLIB so that atexit() handling is called
 * before ExitThread(). That way destructors (if any) are handled within
 * thread group context. Naturally, this means that calling ExitThread() 
 * directly will bypass atexit(). #define EXIT_IN_CLIB if you want it
 * the otherway (destructors won't have threadgroup context)
 * 
 * Furthermore, this prelude provides a PreludeSetOnStopHook(void (*)(void)),
 * that can be used for customizing signal handling (or whatever). The
 * hook will be called *before* prelude passes the request-to-unload to
 * CLIB's _TerminateNLM().
 *
*/

#ifdef __cplusplus
  #error Error: This needs to be a 'C' not 'CPP' file.
#endif   

extern void CYieldUntilIdle(void);
extern void CYieldWithDelay(void);
extern void ConsolePrintf(const char *,...);
extern int GetThreadGroupID(void); 
extern int SetThreadGroupID(int );
typedef long wchar_t; /* unicode char */

#ifdef DEBUG
#define DEBUGPRINTF(x) ConsolePrintf x
#else
#define DEBUGPRINTF(x) /* nothing */
#endif

/* --------------------------------------------------------------------- */

struct __exitcallback { void (*func)(void); int threadGroupID; };  
static struct
{
  void *nlmHandle;
  struct __exitcallback OnStopHook;
  int main_threadGroupID;
  int in_stop;
} preludestatics={0, {((void (*)(void))0), 0}, -1, 0 };

/* --------------------------------------------------------------------- */

#ifdef EXIT_IN_CLIB
#define __atexitstuff(m, f) /* nothing */
extern int atexit(void (*)(void)); 
extern void exit(int);
#else
static int __atexitstuff(int mode, void (*func) (void))
{
  static struct __exitcallback hookList[32];
  static int hookCount = -1;

  DEBUGPRINTF(("beginning atexit mode: %d\r\n", mode));

  if (hookCount < 0)
    hookCount = 0;

  if (mode < 0)
  {
    while (hookCount > 0)
    {
      hookCount--;
      if (hookList[hookCount].func)
      {
        int tgid = hookList[hookCount].threadGroupID;
        func = (void (*)(void))(hookList[hookCount].func);
        hookList[hookCount].func = (void (*)(void))0;
        hookList[hookCount].threadGroupID = 0;
        tgid = SetThreadGroupID(tgid);
        (*func)();
        SetThreadGroupID(tgid);
      }
    }
  }  
  else if (mode > 0)
  {
    if (hookCount < ((int)(sizeof(hookList)/sizeof(hookList[0]))) )
    {
      if (func)
      {
        hookList[hookCount].threadGroupID = GetThreadGroupID();
        hookList[hookCount].func = func;
        hookCount++;
      }
      DEBUGPRINTF(("ended atexit mode: func added ok\r\n"));
      return 0;
    }
    DEBUGPRINTF(("ended atexit mode: func add failed\r\n"));
    return -1; /* EFAILURE (32 functions are already registered) */
  }
  DEBUGPRINTF(("ended atexit mode: %d\r\n", mode));
  return hookCount;
}  
int atexit(void (*func) (void)) 
{ 
  return __atexitstuff(+1, func); 
}
extern void ExitThread( int action_code, int status );
extern int GetThreadID(void);
extern int SuspendThread(int);
void exit(int retcode)
{
  int thrid = GetThreadID();
  DEBUGPRINTF(("exit(): calling atexit() and c++ destructors\r\n"));
  __atexitstuff( -1, 0 );
  DEBUGPRINTF(("exit(): done with atexit() and c++ destructors\r\n"));
  preludestatics.main_threadGroupID = -1;
  DEBUGPRINTF(("exit(): thread group considered dead\r\n"));
  if (preludestatics.in_stop)
  {
    DEBUGPRINTF(("exit(): ok, noted that we are being UNLOADed, so I'm\r\n"
                 "        going to suspend myself... (should not return)\r\n"));
    while (thrid == GetThreadID())
    {
      if (SuspendThread(thrid)!=0)
        thrid = -1;
    }
    DEBUGPRINTF(("exit(): PANIC! I woke up! what do I do now?\r\n"));
    return;
  }
  DEBUGPRINTF(("exit(): doing ExitThread(EXIT_NLM,..). (should not return)\r\n"));
  ExitThread( 1 /* EXIT_NLM */, retcode );
  DEBUGPRINTF(("exit(): PANIC! why am I still here? what do I do now?\r\n"));
}
#endif

/* ---------------------------------------------------------------------- */

#if defined(__WATCOMC__) /* for compatability with Watcom's libraries. */
   
void __InitRtns(unsigned);
#pragma aux __InitRtns "*" parm [eax]
void __FiniRtns(unsigned,unsigned);
#pragma aux __FiniRtns "*" parm [eax] [edx]

#pragma pack(1)
struct ib_data { unsigned char resfield, prio; void (*proc)(void); };
#pragma pack()

static void _initfini_nop(void) {}
#pragma data_seg ( "XIB" );
struct ib_data _Start_XI = { 0, 0, _initfini_nop };
#pragma data_seg ( "XI" );
#pragma data_seg ( "XIE" );
struct ib_data _End_XI = { 0, 0, _initfini_nop };
#pragma data_seg ( "YIB" );
struct ib_data _Start_YI = { 0, 0, _initfini_nop };
#pragma data_seg ( "YI" );
#pragma data_seg ( "YIE" );
struct ib_data _End_YI = { 0, 0, _initfini_nop };
#pragma data_seg ( "_DATA" );

static void __InitFiniRtns(volatile struct ib_data *vectb,  
                           volatile struct ib_data *vecte,
                           volatile unsigned int priob,
                           volatile unsigned int prioe )
{
  if (priob > 255) priob = 0;
  if (prioe > 255) prioe = 255;
  while (priob <= prioe)
  {
    volatile struct ib_data *ibb = vectb, *ibe = vecte;
    //DEBUGPRINTF(("begin initrtns %d->%d\r\n", priob, prioe));
    while (ibb < ibe)
    {
      if (ibb->resfield != 2 && ibb->prio == priob)
      {
        void (*proc)(void) = ibb->proc;
        ibb->resfield = 2;
        if (proc)
        {
          //DEBUGPRINTF(("  begin initrtns %d (ib=%p) proc=%p\r\n", priob, ibb, proc));
          CYieldUntilIdle(); 
          (*proc)();
          CYieldUntilIdle(); 
          //DEBUGPRINTF(("  end initrtns %d %p\r\n", priob, proc));
        }
        ibb->resfield = 2;
      }
      ibb++;
    }
    //DEBUGPRINTF(("end initrtns %d->%d\r\n", priob, prioe));
    priob++;
  }
  return;
}  
  
void __FiniRtns(unsigned priob, unsigned prioe)
{
  if (priob > 255) priob = 0;
  if (prioe > 255) prioe = 255;
  __InitFiniRtns(&_Start_YI,&_End_YI,
                 ((unsigned char)priob),((unsigned char)prioe));
}
static void _initfini_exit(void) { __FiniRtns(0,255); }
void __InitRtns(unsigned prioe)
{
  DEBUGPRINTF(("begin initrtns (c++ contructors)\r\n"));
  if (prioe > 255) prioe = 255;
  atexit(_initfini_exit);
  __InitFiniRtns(&_Start_XI,&_End_XI,0,((unsigned char)prioe));
  DEBUGPRINTF(("end initrtns (c++ constructors)\r\n"));
}

#else
#define __InitRtns(pe)    /* nothing */
#define __FiniRtns(pb,be) /* nothing */
#endif

/* ---------------------------------------------------------------------- */

/*
 * Try very hard to minimize #including anything. 
 * (Just in case Novell changes even this simple api)
*/

#define TRADITIONAL_NLM_INFO_SIGNATURE 0x00000000
#define NLM_INFO_SIGNATURE             0x494D4C4E /* 'NLMI' */

#define TRADITIONAL_FLAVOR             0

/* versions within the traditional flavor: */
#define TRADITIONAL_VERSION            0
#define LIBERTY_VERSION                1

typedef struct NLMInfo
{
  union { unsigned long signature; int ID; } sig;
                                           /* 0 = old NLM, 0x494D4C4E = new */
  unsigned long flavor;                    /* major modality differences */
  unsigned long version;                   /* minor deviation within flavor */
  unsigned int sizeof_long_double; /* size_t */
  unsigned int sizeof_wchar_t;     /* size_t */
} NLMInfo;

static NLMInfo kNLMInfo = 
                {{TRADITIONAL_NLM_INFO_SIGNATURE}, /* or 0x494D4C4E 'NLMI' */
                  TRADITIONAL_FLAVOR,   /* 0 */
                  LIBERTY_VERSION,   /* 1, 0=TRADITIONAL_VERSION */
                  sizeof(long double),  /* 10 */
                  sizeof(wchar_t) };    /* 4 */

/* ---------------------------------------------------------------------- */

/* _cstart_ (called userStartFunc() in Novell documentation) is called 
   from _StartNLM. It in turn calls the first program function, which 
   in the ANSI C "hosted" execution model is main(int argc, char *argv[]).

   Since the NetWare CLIB has a function for setting up argc and argv, 
   there is no need to set the command line here, and we call the CLIB
   function (_SetupArgV()) to do the work for us. There is a special
   _SetupArgV() in NetWare 4.11 that handles STDIN/STDOUT redirection 
   called _SetupArgV_411, otherwise we have to handle redirection ourselves. 
   
   _SetupArgV() calls main() (or whatever) with argc and argv and then
   returns the errorcode returned by main().
*/

extern int _SetupArgv ( int (*main)(int arg, char *argv[]) );
extern int _SetupArgV_411 ( int (*main)(int arg, char *argv[]) );
extern int main( int argc, char *argv[] );

/* The Watcom compiler adds references to an '_argc' to every obj file. */
int _argc = 0;  char **_argv = 0;


static int _pre_main( int argc, char *argv[] )
{
  int rc;
  preludestatics.main_threadGroupID = GetThreadGroupID();
  DEBUGPRINTF(("premain: got here\r\n"));
  _argc = argc; _argv = argv;
  __InitRtns( 255 );
  DEBUGPRINTF(("premain: begin main()\r\n"));
  rc = main(argc, argv);
  DEBUGPRINTF(("premain: end main()\r\npremain: begin exit()\r\n"));
  exit(rc);
  DEBUGPRINTF(("premain: PANIC! exit() came back!\r\n"));
  preludestatics.main_threadGroupID = -1;
  return 0; /* shaddup compiler */
}  

int _cstart_()  
{
  //DEBUGPRINTF(("beginning _cstart_\r\n"));
  return _SetupArgv( _pre_main );
  //DEBUGPRINTF(("end _cstart_\r\n"));
}

/* ---------------------------------------------------------------------- */

/* _Prelude is the default "name" of start function called from the OS 
   The OS doesn't load by name, but by whatever the linker set the "start"
   (see linker OPTION START) function address to. 
   
   Special note on "custom data":  NetWare allows modules to read custom 
   data into system memory during initialization. This data can be anything 
   that might be required. For example, the driver may need to read in 
   firmware to be loaded into a co-processor board. To define the custom 
   file, use the CUSTOM keyword in the driver's definition file followed by 
   the file's name (custom files are simply appended to the NLM/DSK/NAM/LAN
   file). NetWare passes the custom data file's handle, starting offset, 
   size, and the Read routine address to the Initialize routine, where it 
   should be saved upon entry. Initialize Driver can read the file into 
   memory by calling the Read routine using the syntax shown below:
   ReadCustomDataRoutine( LONG CustomDataFileHandle, LONG CustomDataOffset,
                           BYTE *CustomDataDestination, LONG CustomDataSize);
   Note: the CustomDataFileHandle is the same as the NLM file handle.

   Of Interest: The stack frame size is [ebp-esp] (at the start of _Prelude).
*/

extern unsigned long _StartNLM( void *NLMHandle, void *initErrorScreenID, 
   unsigned char *cmdLineP, unsigned char *loadDirectoryPath,
   unsigned long uninitializedDataLength, unsigned long NLMFileHandle,
   unsigned long (*readRoutineP)(), unsigned long customDataOffset,
   unsigned long customDataSize, NLMInfo *NLMInformation,
   int (*userStartFunc)() );

unsigned long _Prelude( void *NLMHandle, void *initErrorScreenID,
   unsigned char *cmdLineP, unsigned char *loadDirectoryPath,
   unsigned long uninitializedDataLength, unsigned long NLMFileHandle,
   unsigned long (*readRoutineP)(), unsigned long customDataOffset,
   unsigned long customDataSize )
{
   extern char _edata, _end;
   register unsigned char *q = &_end;   /* end of BSS (start of stack?) */
   register unsigned char *p = &_edata; /* end of DATA (start of BSS) */
   while (p < q) *p++=0; 
   p = (char *)(&preludestatics);
   q = p+sizeof(preludestatics);
   while (p < q) *p++=0;

   preludestatics.nlmHandle = NLMHandle;

   return _StartNLM( NLMHandle, initErrorScreenID, cmdLineP, 
     loadDirectoryPath,  uninitializedDataLength, NLMFileHandle, readRoutineP,
     customDataOffset, customDataSize, &kNLMInfo, _cstart_ );
}

/* ---------------------------------------------------------------------- */

/* _Stop is the default "name" of stop function called from the OS 
   The OS doesn't load by name, but by whatever the linker set the "stop"
   (see linker OPTION STOP) function address to. */

#define TERMINATE_BY_EXTERNAL_THREAD   0
#define TERMINATE_BY_UNLOAD            5

extern unsigned long _TerminateNLM( int NLMID, int threadID, int status );

void (*PreludeSetOnStopHook(void (*func) (void)))(void)
{
  register void (*oldfunc)(void) = preludestatics.OnStopHook.func;
  preludestatics.OnStopHook.func = func;
  preludestatics.OnStopHook.threadGroupID = GetThreadGroupID();
  return oldfunc;
}

extern void *signal(int, void *); 
extern int BeginThread( void (*)(void *), void *, unsigned int, void *);
extern unsigned long GetCurrentTime(void);

void _Stop( void ) 
{
  int tgid;
  preludestatics.in_stop = 1;

#ifndef EXIT_IN_CLIB
  DEBUGPRINTF(("_stop(): beginning. entry from OS console process\r\n" ));
  if (preludestatics.main_threadGroupID != -1 && 
      preludestatics.main_threadGroupID != 0)
  {
    void *sighdlr;
    tgid = SetThreadGroupID(preludestatics.main_threadGroupID);
    sighdlr = signal( 6 /*SIGTERM*/, ((void *)0x01) /* SIG_IGN */);
    DEBUGPRINTF(("_stop(): signal(6) == %p\r\n", sighdlr ));
    if ( sighdlr < ((void *)10) )
      sighdlr = ((void *)0);
    else if (-1 == BeginThread( ((void (*)(void *))sighdlr), 0, 16384, ((void *)6) ))
      sighdlr = ((void *)0);
    SetThreadGroupID(tgid);
    if (sighdlr)
    {
      unsigned long lasttime = 0;
      unsigned int ticks = (20*182)/10; /* 20 seconds */
      DEBUGPRINTF(("_stop(): beginning \"wait for exit\" loop\r\n"));
      while (ticks && preludestatics.main_threadGroupID != -1 && 
                      preludestatics.main_threadGroupID != 0)
      {
        unsigned long timenow = GetCurrentTime();
        if (timenow != lasttime)
          ticks--;
        lasttime = timenow;
        CYieldWithDelay(); /* let it start (and possibly finish) */
      }
      DEBUGPRINTF(("_stop(): end \"wait for exit\" loop. time remaining: %u.%02us (%u ticks)\r\n", (ticks*10)/182, (ticks*10)%182, ticks));
    }
  }

  if (preludestatics.OnStopHook.func &&
      preludestatics.main_threadGroupID != -1 && 
      preludestatics.main_threadGroupID != 0) /* main didn't return */
  {
    int tgid = SetThreadGroupID(preludestatics.OnStopHook.threadGroupID);
    (*preludestatics.OnStopHook.func)();
    SetThreadGroupID(tgid);
  }
  if (preludestatics.main_threadGroupID != -1 && 
      preludestatics.main_threadGroupID != 0) /* main didn't return */
  {
    __atexitstuff( -1, 0 );
  }
#endif /* EXIT_IN_CLIB */
  DEBUGPRINTF(("_stop(): calling _TerminateNLM(). (Should come back to us)\r\n"));
  _TerminateNLM( kNLMInfo.sig.ID, TERMINATE_BY_EXTERNAL_THREAD /* 0 */ , 
                                  TERMINATE_BY_UNLOAD  /* 5 */ );
  DEBUGPRINTF(("_stop(): finished. Returning to OS.\r\n"));
  return;
}                                      

/* --------------------------------------------------------------------- */

void *GetNLMHandleFromPrelude(void)
{
  return preludestatics.nlmHandle;
}
