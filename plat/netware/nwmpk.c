/*
 * Stubs/Emulation for the MPK* API as first released with NetWare 5.0
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: nwmpk.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $
 *
*/

//#define DEBUG_TABLELOAD

#include "nwlemu.h"

#ifdef __cplusplus
extern "C" {
#endif
//#include <conio.h>
//#include <process.h>
#include <nwsemaph.h>
#include <errno.h>
#include <string.h>
#include <nwmpk.h>
#ifdef __cplusplus
}
#endif

#if 0

push    offset i_thread_?_?
push    0
push    2000h

mov     eax, NLMHandle
mov     eax, [eax+278h]
push    dword ptr [eax+2E0h]

lea     eax, [ebp-8]
push    eax
push    0
call    thread_create
add     esp, 18h

thread_context[0x000] = fast signature ('m' or 'M')
thread_context[0x008] = mutex thread link
thread_context[0x018] = mutex waiters
thread_context[0x020] = mutex owner 
thread_context[0x028] = mutex name
thread_context[0x034] = mutex SPL
thread_context[0x03c] = mutex lock count
thread_context[0x040] = mutex fail count
thread_context[0x044] = mutex readers
thread_context[0x050-0x67] = 6 ptrs, to 6 mutex stacks

thread_context[0x000] = fast signature ('t' or 'T') 
thread_context[0x00C] = stack base
thread_context[0x010] = current stack ptr (SP reg) [saved at switch time]
thread_context[0x014] = signature (0x12345678)
thread_context[0x018] = Global_PRSC_Resource_Tag
thread_context[0x01C] = state flags? (0x02=sleep)
thread_context[0x01E-0x2D] = thread name
thread_context[0x3A8] = "run forward link"
thread_context[0x3A8] = "run back link"
thread_context[0x3B0] = thread_state: 
                        0x01=wait, 0x02=suspended, 0x04=run, 0x08=uninit,
                        (0x08 is the type when created by thread_start)
                        0x10=halted, 0x20=sleep, 0x40=event, 0x80=idle,
                        0x100=swapped, 0x400=sema, 0x800=low prio poll
                        0x1000=high prio poll, 0x2000=??
                        0x0C000000=delayed
                        0x10000000=migrate, 0x20000000=barrier, 
                        0x40000000=rwlock, 0x80000000=mutex,
                        0xC0000000=rundown
thread_context[0x3B4] = event link
thread_context[0x3B8] = ? explicit zero on creation
thread_context[0x3BC] = prio (dword). Initially 100.
thread_context[0x3c0] = copy of prio when going to mutex_[try]lock|wait
thread_context[0x3C8] = scheduler flags (0x04 = preemption on), 0x02 on creation
thread_context[0x3D0] = proc * (value passed in 6th arg at creation)
thread_context[0x3E4] = "task"
thread_context[0x3E8] = "pset forward link"
thread_context[0x3EC] = "pset back link"
thread_context[0x3F0] = marshall mutex for thread
thread_context[0x400] = spin lock mutex for thread
thread_context[0x404] = ? explicit zero on creation ("references")
thread_context[0x408] = cpu time for thread (microsecs?)
thread_context[0x40C] = time at last switch (absolute)
thread_context[0x410] = "pset" (what is that?)
thread_context[0x414] = current thread binding (PCB *) [zero on creation]
thread_context[0x418] = switch count (by isr only?)
thread_context[0x41C] = current thread assignment (PCB *)
thread_context[0x420] = last engine
thread_context[0x424] = ?? (value passed in 5th arg at creation (always zero?))
thread_context[0x428] = ?? (value passed in 4th arg at creation (always 2000?))
thread_context[0x430] = thread-type/smp-state (0 == not-smp-thread)
thread_context[0x434] = ptr to 8 byte struct. [0x00]=thread_context, [0x04] = thread_t
thread_context[0x440] = exit val *
thread_context[0x4AC] = "migrating" (long value) what is this?
thread_context[0x4d8] = fp reg saved flag
thread_context[0x4dc] = fp reg save area
        1          2         3             4                  5    6
        14h      18h         1Ch          20h                24h   28h
create( unused, thread_t *, name?, int type? (always2000), 0, proc *)

int thr_create(?,?,?,?,?);                /* 0=success, -1=error */
unsigned int thr_minstack(void);          /* currently 8192 */
int thr_join(thread_t, void **exitinfoP); /* 0=success, -1=error */
void thr_exit(thread_t, void *exitinfo);  /* never returns */
thread_t thr_self(void);                  /* thread_context[0x434]->[0x04] */
int thr_suspend(thread_t);                /* 0=ok (always?!) */
int thr_continue(thread_t);               /* 0=ok (always?!) */
int thr_kill(thread_t, void *exitinfo);   /* 0=ok (always?!) */
int thr_setscheduler(thread_t, sched_t *) /* 0=ok (always?!) */
int thr_getscheduler(thread_t, sched_t ); /* 0=ok (always?!) */
int thr_getprio(thread_t);                /* returns curr prio */
int thr_setprio(thread_t, int );          /* 0=ok (always?!) */
void thr_yield(void);         /* is non-MP safe. same as thread_yield */
int thr_active(thread_t);     /* returns thread_active(thread_context) */
int thr_type(void);           /* =>0=non-smp, same as thread_type */
void thr_delay(unsigned int millisecs);   /* non-MP safe? */
int thr_preempt_on(void);     /* returns old flags (0x04=preemption bit) */
int thr_preempt_off(void);    /* returns old flags (0x04=preemption bit) */

/* ------------------------------------- */

void *thread_context(void); /* (NetWareContext)?(RunningProcess):(v_psmtable[currcpu])) */
                   14    18           1C
int thread_create( 0, thread_t *, resourcetag?/name?, int type? (always2000), 0, proc * );
                                 /* returns 0=ok, -1=err */
int thread_terminate(t_context); /* 0=ok, -1=err */
int thread_active(t_context);    /* zero if thread_context[0x3B0]==0x100 */
int thread_kill(t_context);      /* 0=ok, -1=err */
int thread_start(some id?)       /* */
void thread_yield(void);         /* identical to thr_yield. Non-MP safe */
int thread_assign(t_context,cpuPCB *);  
int thread_bind(t_context, cpuPCB *);
int thread_unbind(t_context, cpuPCB *); /* -1=err, 0=ok, 1=notbound at all, 2=not bound to that */
int thread_resume(t_context);    /* 0=ok, -1=err */
int thread_suspend(t_context);   /* 0=ok, -1=err **BUG** suspends currthrd */
int thread_type(t_context);      /* identical to thr_type() */
int thread_switch(void);         /* NOT MP SAFE */
thread_[add|remove]_[hi|low]pri_poll
#endif

/* -------------------------------------------------------------------- */

static void *__get_symbol_ptr(unsigned int __sym_xxx)
{
  static int mpk_sym_count = -1;

  /* The following MPK* functions no longer exist in NetWare 5.1:
     MPKExitThread, MPKGetThreadExitCode, MPKThreadCheckFromSuspendKill,
     MPKGetThreadAttributes, MPKSetThreadAttributes, MPKGetThreadList,
     MPKGetThreadUserData, MPKSetThreadUserData, kIsThreadInCNB
     There may be others that don't exist anymore, but those are 
     the confirmed dead.
  */      
  static struct { const char *symname; void *symptr; int mpk_set; } symtab[] =
  {
    /* searching the symbol table is computationally expensive on NetWare 5,
       so mpk_set identifies the function set to use. 1/-1=NetWare 5, 
       2/-2=NetWare 4.11, 0=any/all. Negative numbers means "optional".
       The entire set is considered invalid if *any* non-optional symbols 
       can't be found.
    */
                #define __sym_MPKCurrentThread         0
                { "\x10""MPKCurrentThread",              (void *)0, 1 },
                #define __sym_MPKCreateThread          1
                { "\x0f""MPKCreateThread",               (void *)0, 1 },
                #define __sym_MPKStartThread           2
                { "\x0e""MPKStartThread",                (void *)0, 1 },
                #define __sym_MPKScheduleThread        3
                { "\x11""MPKScheduleThread",             (void *)0, 1 },
                #define __sym_MPKYieldThread           4
                { "\x0e""MPKYieldThread",                (void *)0, 1 },
                #define __sym_MPKSuspendThread         5    
                { "\x10""MPKSuspendThread",              (void *)0, 1 },
                #define __sym_MPKResumeThread          6
                { "\x0f""MPKResumeThread",               (void *)0, 1 },
                #define __sym_MPKGetThreadName         7
                { "\x10""MPKGetThreadName",              (void *)0, 1 },
                #define __sym_MPKSetThreadName         8
                { "\x10""MPKSetThreadName",              (void *)0, 1 },
                #define __sym_MPKDestroyThread         9
                { "\x10""MPKDestroyThread",              (void *)0, 1 },
                #define __sym_MPKGetThreadPriority    10
                { "\x14""MPKGetThreadPriority",          (void *)0, -1 },
                #define __sym_MPKSetThreadPriority    11
                { "\x14""MPKSetThreadPriority",          (void *)0, -1 },
                #define __sym_MPKEnterNetWare         12
                { "\x0f""MPKEnterNetWare",               (void *)0, 1 },
                #define __sym_MPKExitClassicNetWare   13
                { "\x15""MPKExitClassicNetWare",         (void *)0, -1 },
                #define __sym_MPKExitNetWare          14
                { "\x0e""MPKExitNetWare",                (void *)0, 1 },
                #define __sym_ThreadSwitchLowPriority 15  /* >= 4.x */
                { "\x17""ThreadSwitchLowPriority",       (void *)0, 0 },
                #define __sym_thr_yield_to_mp         16
                { "\x0f""thr_yield_to_mp",               (void *)0, 2 },
                #define __sym_thr_yield_to_netware    17
                { "\x14""thr_yield_to_netware",          (void *)0, 2 },
                #define __sym_thr_yield               18
                { "\011""thr_yield",                     (void *)0, 2 },
                #define __sym_GetProcessorControlBlock 19
                { "\x18""GetProcessorControlBlock",      (void *)0, 2 },
                #define __sym_thread_bind             20
                { "\013""thread_bind",                   (void *)0, 2 },
                #define __sym_kBindThread             21
                { "\013""kBindThread",                   (void *)0, 1 },
                #define __sym_RunningProcess          22
                { "\016""RunningProcess",                (void *)0, 2 },
                #define __sym_kIsThreadInCNB          23     /* not in 5.1 */
                { "\016""kIsThreadInCNB",                (void *)0, -1 },
                #define __sym_CurrentProcess          24
                { "\016""CurrentProcess",                (void *)0, 1 },
                #define __sym_thread_type             25
                { "\013""thread_type",                   (void *)0, 2 },
                #define __sym_thr_delay               26
                { "\011""thr_delay",                     (void *)0, 2 },
                #define __sym_thr_preempt_on          27
                { "\016""thr_preempt_on",                (void *)0, 2 },
                #define __sym_thr_preempt_off         28
                { "\017""thr_preempt_off",               (void *)0, 2 },
                #define __sym_thr_self                29
                { "\010""thr_self",                      (void *)0, 2 },
                #define __sym_thr_getprio             30
                { "\013""thr_getprio",                   (void *)0, 2 },
                #define __sym_thr_setprio             31
                { "\013""thr_setprio",                   (void *)0, 2 },
                #define __sym_CPU_Utilization         32
                { "\017""CPU_Utilization",               (void *)0, 2 }
  };
  unsigned int pos;
  if (mpk_sym_count < 0)
  {
    int nwver = (GetFileServerMajorVersionNumber()*100)+
                 GetFileServerMinorVersionNumber();
    unsigned int mpk_found = 0, mpk_count = 0;
    int nlmHandle = GetNLMHandleFromPrelude();
    #ifdef DEBUG_TABLELOAD
    unsigned long ticksnow = GetCurrentTicks();  
    ConsolePrintf("\rbeginning sym import, %d\r\n", nwver);
    #endif

    for (pos=0; pos<(sizeof(symtab)/sizeof(symtab[0])); pos++)
    {    
      int tryit = 0;                                                              
      #ifdef DEBUG_TABLELOAD
      if (strlen(symtab[pos].symname+1) != symtab[pos].symname[0])
        ConsolePrintf("badlen for %s: Should be 0x%x\r\n", symtab[pos].symname+1, strlen(symtab[pos].symname+1));
      #endif
      if (nwver < 400)
        tryit = 0;
      else if (symtab[pos].mpk_set == 0)
        tryit = 1;
      else if (mpk_found != mpk_count)
        tryit = 0;
      else if (nwver >= 500 && (symtab[pos].mpk_set == 1 || symtab[pos].mpk_set == -1))
        tryit = 1;
      else if (nwver  < 500 && (symtab[pos].mpk_set == 2 || symtab[pos].mpk_set == -2))
        tryit = 1;
      symtab[pos].symptr = (void *)0;
      if (tryit)
      {
        symtab[pos].symptr = ImportPublicSymbol( nlmHandle, symtab[pos].symname );
        #ifdef DEBUG_TABLELOAD
        ConsolePrintf("\r%s => %08x\r\n", symtab[pos].symname+1, symtab[pos].symptr );
        #endif
        if (!symtab[pos].symptr)
        { 
          int altsym = 1;
          #if 0
          if (pos == __sym_MPKGetThreadAttributes) /* 5.0 bug */
            symtab[pos].symname = "\x16""MPKGetThreadAttribtues";
          else if (pos == __sym_MPKSetThreadAttributes) /* 5.0 bug */
            symtab[pos].symname = "\x16""MPKSetThreadAttribtues";
          else
          #endif
            altsym = 0;
          if (altsym)  
            symtab[pos].symptr = ImportPublicSymbol( nlmHandle, symtab[pos].symname );
        }
        if (symtab[pos].mpk_set)
        {
          mpk_count++;
          if (symtab[pos].symptr || symtab[pos].mpk_set < 0)
            mpk_found++;
        }
      }
    }
    if (mpk_found != mpk_count)
    {
      for (pos=0; pos<(sizeof(symtab)/sizeof(symtab[0])); pos++)
      {
        if (symtab[pos].symptr && symtab[pos].mpk_set)
        {
          symtab[pos].symptr = (void *)0;
          UnImportPublicSymbol( nlmHandle, symtab[pos].symname );
        }  
      }
      mpk_found = 0;
    }  
    mpk_sym_count = (int)mpk_found;

    #ifdef DEBUG_TABLELOAD
    ConsolePrintf("\rend sym import. %d found. Elapsed time: %u ticks\r\n", mpk_found, GetCurrentTicks()-ticksnow);
    #endif
  }                
  if (__sym_xxx < (sizeof(symtab)/sizeof(symtab[0])) )
  {
    return symtab[__sym_xxx].symptr;
  }
  return (void *)0;
}

static MPKError __validate_thread_id(int threadid) /* FIXME! */
{
  if (threadid != 0 && threadid != -1)
    return (MPKError)0;
  return (MPKError)1; /* ESRCH */
}

/* ----------------------------------------------------------------- */

static int __is_thread_in_cnb(void) /* < 0 = error */
{
  int (*_kIsThreadInCNB)(void *) = 
                 (int (*)(void *))__get_symbol_ptr(__sym_kIsThreadInCNB);
  if (_kIsThreadInCNB)
  {
    void * (*_CurrentProcess)(void) = 
                 (void * (*)(void))__get_symbol_ptr(__sym_CurrentProcess);
    if (_CurrentProcess)
    {
      if ( (*_kIsThreadInCNB)( (*_CurrentProcess)() ) )
        return 1;
      return 0;
    }
  }
  else
  {
    int (*_thread_type)(void) = 
                (int (*)(void))__get_symbol_ptr(__sym_thread_type);
    if (_thread_type)
    {
      if ( (*_thread_type)() )
        return 0;
      return 1;
    }
  }
  return -1;
}

/* ----------------------------------------------------------------- */

MPKThread MPKCurrentThread(void)
{
  MPKThread thread = (MPKThread)0;
  MPKThread (*proc)(void) = 
     (MPKThread (*)(void))__get_symbol_ptr(__sym_MPKCurrentThread);
  #if 0 /* can't do this! */
  if (!proc)
  {
    proc = (MPKThread (*)(void))__get_symbol_ptr(__sym_thr_self);
  }
  #endif
  if (proc)
  {
    thread = (*proc)();
  }
  else
  {
    thread = (MPKThread)GetThreadID();
  }
  return thread;
}

/* ----------------------------------------------------------------- */

struct __thread_helper_struct 
{ 
  /* int needfree; */
  void (*start)(void *);
  void *arg;
};
  
static void __thread_helper( void *boot_arg )
{
  struct __thread_helper_struct *ths = (__thread_helper_struct *)boot_arg;
  void (*tomp)(void) = (void (*)(void))__get_symbol_ptr(__sym_thr_yield_to_mp);
  void (*start)(void *) = ths->start;
  void *arg = ths->arg;
  /* int need_free = ths->needfree; */

  ths->start = (void (*)(void *))0; /* signal start */

  /*
  if (need_free) 
    free(boot_arg);
  */

  if (tomp)
    (*tomp)();
    
  (*(start))(arg);
  return;
}  

#if 0
static int __oldstyle_begin_thread(void (*start)(), void *stackAddr,
                           size_t stackSize, void *arg )
{
  int threadid = 0;
  struct __thread_helper_struct *ths = 
      (struct __thread_helper_struct *)
      malloc(sizeof(struct __thread_helper_struct));
  if (ths)
  {
    /* ths->needfree = 1; */
    ths->start = (void (*)(void *))start;
    ths->arg = arg;
    threadid = BeginThreadGroup( __thread_helper, stackAddr, 
                            (unsigned int)stackSize, (void *)ths );
    if (threadid == -1)
      threadid = 0;
    if (!threadid)
      free((void *)ths);
  }  
  return threadid;
}  
#endif

/* ----------------------------------------------------------------- */

MPKThread MPKCreateThread( const char *name, void (*start)(), void *stackAddr,
                           size_t stackSize, void *arg )
{
  MPKThread thread = (MPKThread)0;
  MPKThread (*proc)( const char *, void (*)(), void *, size_t, void *) =
     (MPKThread (*)( const char *, void (*)(), void *, size_t, void *))
        __get_symbol_ptr(__sym_MPKCreateThread);
  if (proc)
  {
    thread = (*proc)(name, start, stackAddr, stackSize, arg);
  }
  else
  {
    errno = 0;
//ConsolePrintf("beginthread: %s\r\n", strerror(errno));
    int threadid = BeginThreadGroup( (void (*)(void *))start, stackAddr, 
                                    (unsigned int)stackSize, arg );
    if (threadid == -1)
    {
      threadid = 0;
//ConsolePrintf("beginthread: %s\r\n", strerror(errno));
    }
    if (threadid)
    {
      thread = (MPKThread)threadid;
      if (name)
        MPKSetThreadName(thread, (void *)name );
    }
  }
  return thread;
}                                   

/* ----------------------------------------------------------------- */

MPKThread MPKStartThread( const char *name, void (*start)(), void *stackAddr,
                          size_t stackSize, void *arg )
{
  MPKThread thread = (MPKThread)0;
  MPKThread (*proc)( const char *, void (*)(), void *, size_t, void *) = 
     (MPKThread (*)( const char *, void (*)(), void *, size_t, void *))
        __get_symbol_ptr(__sym_MPKStartThread);
  if (proc)
  {
    thread = (*proc)(name, start, stackAddr, stackSize, arg);
  }
  else
  {
    int threadid;
    struct __thread_helper_struct ths;
    /* ths.needfree = 0; */
    ths.start = (void (*)(void *))start;
    ths.arg = arg;

    threadid = BeginThread( __thread_helper, stackAddr, 
                            (unsigned int)stackSize, (void *)&ths );
    if (threadid == -1)
      threadid = 0;
    if (threadid)
    {
      thread = (MPKThread)threadid;
      if (name)
        MPKSetThreadName(thread, (void *)name );
      while (ths.start) /* wait for spin up */
        MPKYieldThread();
    }
  }
  return thread;
}                                 

/* ----------------------------------------------------------------- */

MPKError MPKScheduleThread( MPKThread thread )
{
  MPKError retcode = (MPKError)1; /* ENOTSUPPORTED */
  MPKError (*proc)(MPKThread) = (MPKError (*)(MPKThread))__get_symbol_ptr(__sym_MPKScheduleThread);
  if (proc)
  {
    retcode = (*proc)( thread );
  }
  return (MPKError)retcode;
}  

/* ----------------------------------------------------------------- */

static void __pre_yield_fixup_cpuutil_411(void)
{
  /* this is a hack to get around SMP.NLM's thread_switch() dead 
     spinning (well, on a spin lock) until cpu utilization falls 
     below 98% (and while CPU_Combined is below 80%)
  */
  unsigned int (*_CPU_Utilization) = 
             (unsigned int (*))__get_symbol_ptr(__sym_CPU_Utilization);
  if (_CPU_Utilization)
  {
    int cpunum, numcpus = GetNumberOfRegisteredProcessors();
    for (cpunum=0; cpunum < numcpus; cpunum++)
    {
      if ((*_CPU_Utilization) >= 98)
      {
        (*_CPU_Utilization) = 97;
      }   
      _CPU_Utilization++;
    }
  }
  return;
}

void MPKYieldThread( void )
{
  void (*proc)(void) = (void (*)(void))__get_symbol_ptr(__sym_MPKYieldThread);
  if (proc)
  {
    (*proc)();
  }
  else
  {  
    int incnb = __is_thread_in_cnb();
    if (incnb == 0)
    {
      proc = (void (*)(void))__get_symbol_ptr(__sym_thr_yield);
      if (proc)
      {
        __pre_yield_fixup_cpuutil_411();
        (*proc)();
        return;
      }
    }
    ThreadSwitchLowPriority(); 
    #if 0
    proc = (void (*)(void))__get_symbol_ptr(__sym_ThreadSwitchLowPriority);
    if (!proc)
    {
      proc = ThreadSwitch;
      ConsolePrintf("\rUsing threadswitch!\r\n");
    }
    (*proc)();
    if (incnb > 0)
    {
      //MPKExitNetWare();
    }
    #endif
  }
  return;
}

/* ----------------------------------------------------------------- */

void MPKSuspendThread( MPKThread thread )
{
  void (*proc)(MPKThread) = (void (*)(MPKThread))__get_symbol_ptr(__sym_MPKSuspendThread);
  if (proc)
  {
    (*proc)( thread );
  }
  else
  {
    int threadid = (int)thread;
    MPKError retval = __validate_thread_id(threadid);
    if (retval == 0 && threadid == GetThreadID()) 
      retval = (MPKError)1;
    if (retval == 0)
      SuspendThread( threadid );
  }
  return;
}  

/* ----------------------------------------------------------------- */

MPKError MPKResumeThread( MPKThread thread )
{
  MPKError retval = (MPKError)0;
  MPKError (*proc)(MPKThread) = (MPKError (*)(MPKThread))__get_symbol_ptr(__sym_MPKResumeThread);
  if (proc)
  {
    retval = (*proc)( thread );
  }
  else
  {
    int threadid = (int)thread;
    retval = __validate_thread_id(threadid);
    if (retval == 0 && threadid == GetThreadID()) 
      retval = (MPKError)1;
    if (retval == 0)
    {
      int rc = ResumeThread( threadid );
      if (rc != 0) retval = (MPKError)1; /* ??? */
    }
  }
  return retval;
}  

/* ----------------------------------------------------------------- */

MPKError MPKGetThreadName( MPKThread thread, void *name, size_t maxLen )
{
  MPKError retval = (MPKError)1; /* EINVAL */
  if (name && maxLen)
  {
    MPKError (*proc)(MPKThread, void *, size_t) = 
       (MPKError (*)(MPKThread, void *, size_t))
             __get_symbol_ptr(__sym_MPKGetThreadName);
    if (proc)
    {
      retval = (*proc)( thread, name, maxLen );
    }
    else
    {
      int threadid = (int)thread;
      retval = __validate_thread_id(threadid);
      if (retval == 0)
      {
        char threadname[32];
        if (GetThreadName(threadid,threadname) != 0)
          retval = (MPKError)1; /* ESRCH */
        else
        {
          int i = 0;
          char *p = (char *)name;
          while (maxLen > 1 && threadname[i])
          {
            *p++ = threadname[i++];
            maxLen--;
          }
          *p = '\0';
        }
      }
    }
  }
  return retval;
}  

/* ----------------------------------------------------------------- */

MPKError MPKSetThreadName( MPKThread thread, const void *name )
{
  MPKError retval = (MPKError)1; /* EINVAL */
  if (name)
  {
    MPKError (*proc)(MPKThread, const void *) =
       (MPKError (*)(MPKThread, const void *))
             __get_symbol_ptr(__sym_MPKSetThreadName);
    if (proc)
    {
      retval = (*proc)( thread, name );
    }
    else
    {
      int threadid = (int)thread;
      retval = __validate_thread_id(threadid);
      if (retval == 0)
      { 
        char threadname[18];
        unsigned int i = 0; char *p = (char *)name;
        while (*p && i < (sizeof(threadname)-1))
          threadname[i] = *p++;
        threadname[i] = '\0';
        if (RenameThread( threadid, threadname ) != 0)
        {
          retval = (MPKError)1;
        }
      }
    }
  }
  return retval;
}

/* ----------------------------------------------------------------- */

MPKError MPKDestroyThread( MPKThread thread )
{
  MPKError retval = (MPKError)0;
  MPKError (*proc)(MPKThread) = 
     (MPKError (*)(MPKThread))__get_symbol_ptr(__sym_MPKDestroyThread);
  if (proc)
  {
    retval = (*proc)( thread );
  }
  else
  {
    static int shownonce = -1;
    if (shownonce < 0)
    {
      shownonce = 1;
      ConsolePrintf("\r\nMPKDestroyThread is not supported in this version.\r\n");
    }
    retval = (MPKError)1; /* ENOTSUPPORTED */
  }
  return retval;
}    


/* ----------------------------------------------------------------- */

MPKError MPKExitThread( void *status ) /* no longer supported on 5.1 */
{
  status = status;
  return (MPKError)1; /* ENOTSUPPORTED */
}

/* ----------------------------------------------------------------- */

MPKError MPKGetThreadExitCode( MPKThread thread, void **status )
{                                          /* no longer supported on 5.1 */
  thread = thread; status = status;
  return (MPKError)1; /* ENOTSUPPORTED */
}  

/* ----------------------------------------------------------------- */

MPKError MPKThreadCheckFromSuspendKill( void )
{                                         /* no longer supported on 5.1 */
  return (MPKError)1; /* ENOTSUPPORTED */
}
  
/* ----------------------------------------------------------------- */

MPKError MPKGetThreadAttributes( MPKThread thread, LONG *attributes )
{                                         /* no longer supported on 5.1 */
  thread = thread; attributes = attributes; 
  return (MPKError)1; /* ENOTSUPPORTED */
}

/* ----------------------------------------------------------------- */

MPKError MPKSetThreadAttributes( MPKThread thread, LONG attributes )
{                                        /* no longer supported on 5.1 */
  thread = thread; attributes = attributes;
  return (MPKError)1; /* ENOTSUPPORTED */
}

/* ----------------------------------------------------------------- */

LONG MPKGetThreadPriority( MPKThread thread )
{
  LONG retval = 0;
  LONG (*proc)(MPKThread) = 
     (LONG (*)(MPKThread))__get_symbol_ptr(__sym_MPKGetThreadPriority);
  #if 0 /* doesn't work! */
  if (!proc)  
    proc = (LONG (*)(MPKThread))__get_symbol_ptr(__sym_thr_getprio);
  #endif
  if (proc)
  {
    retval = (*proc)(thread);
  }
  return retval;
}
  
/* ----------------------------------------------------------------- */

MPKError MPKSetThreadPriority( MPKThread thread, LONG priority )
{
  MPKError retval = (MPKError)1; /* ENOTSUPPORTED */
  MPKError (*proc)(MPKThread, LONG) = 
     (MPKError (*)(MPKThread, LONG))
     __get_symbol_ptr(__sym_MPKSetThreadPriority);
  if (proc)
  {
    retval = (*proc)(thread, priority);
  }
  #if 0 /* doesn't work */
  else
  {
    proc = (MPKError (*)(MPKThread, LONG))__get_symbol_ptr(__sym_thr_setprio);
    if (proc)
    {
      retval = (*proc)(thread, priority); /* returns 0 or -1 */
      if (retval != 0)
        retval = (MPKError)1; /* FIXME */
    }
  }
  #endif
  return retval;
}

/* ----------------------------------------------------------------- */

MPKError MPKGetThreadList( MPKAppl app, MPKThread *buffer, LONG slots,
            LONG *slotsUsed )               /* no longer supported on 5.1 */
{
  app = app; buffer = buffer; slots = slots; slotsUsed = slotsUsed;
  return (MPKError)1; /* ENOTSUPPORTED */
}

/* ----------------------------------------------------------------- */

MPKError MPKSetThreadUserData( MPKThread thread, void *data )
{                                          /* no longer supported on 5.1 */
  thread = thread; data = data;
  return (MPKError)1; /* ENOTSUPPORTED */
}

/* ----------------------------------------------------------------- */

void *MPKGetThreadUserData( MPKThread thread )
{                                          /* no longer supported on 5.1 */
  thread = thread;
  return 0; /* ENOTSUPPORTED */
}

/* ----------------------------------------------------------------- */

static MPKError __enterexitnetware(int sym)
{
  MPKError retval = (MPKError)1; /* ENOTSUPPORTED */
  MPKError (*proc)(void) =
     (MPKError (*)(void)) __get_symbol_ptr(sym);
  if (proc)
  {
    retval = (*proc)();
  }  
  else
  {
    if (sym == __sym_MPKEnterNetWare)
      sym = __sym_thr_yield_to_netware;
    else /* __sym_MPKExitNetWare or __sym_MPKExitClassicNetWare */
      sym = __sym_thr_yield_to_mp;
    proc = (MPKError (*)(void)) __get_symbol_ptr(sym);
    if (proc)
    {
      (*proc)();
      retval = (MPKError)0;
    }
  }
  return retval;
}

void MPKEnterNetWare( void )
{
  __enterexitnetware(__sym_MPKEnterNetWare);
}
MPKError MPKExitClassicNetWare( void )
{
  return __enterexitnetware(__sym_MPKExitClassicNetWare);
}
void MPKExitNetWare( void )
{
  __enterexitnetware(__sym_MPKExitNetWare);
}  

/* ----------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" int __MPKEnableThreadPreemption(void);
extern "C" int __MPKDisableThreadPreemption(void);
#endif

int __MPKEnableThreadPreemption(void)
{
  int (*proc)(void) = (int (*)(void))__get_symbol_ptr(__sym_thr_preempt_on);
  if (proc)
  {
    int i = (*proc)();
    if ((i & 0x04)!=0)
      return 0;
    i = (*proc)();
    if ((i & 0x04)!=0)
      return 0;
    return -2;
  }
  return -1;
}

int __MPKDisableThreadPreemption(void)
{
  int (*proc)(void) = (int (*)(void))__get_symbol_ptr(__sym_thr_preempt_off);
  if (proc)
  {
    int i = (*proc)();
    if ((i & 0x04)==0)
      return 0;
    i = (*proc)();
    if ((i & 0x04)==0)
      return 0;
    return -2;
  }
  return -1;
}

/* ----------------------------------------------------------------- */

#ifdef __cplusplus 
extern "C" void __MPKDelayThread( unsigned int millisecs );
#endif

void __MPKDelayThread( unsigned int millisecs )
{
  void (*proc)(unsigned int) = (void (*)(unsigned int))
                               __get_symbol_ptr(__sym_thr_delay);
  if (proc)
  {
    (*proc)(millisecs);
    return;
  }
  delay(millisecs);
  return;

  #if 0
  static int need_migrate = -1;
  if (need_migrate == -1)
  {
    if (GetFileServerMajorVersionNumber() == 4 && 
        GetNumberOfRegisteredProcessors() > 1)
      need_migrate = 1; 
    else
    {
      need_migrate = 0;
      MPKExitNetWare();
    }
  }
  if (need_migrate)
  {
    LONG semp = OpenLocalSemaphore(0);
    if (semp)
    {
      if (millisecs < 55)
        millisecs = 55;
      TimedWaitOnLocalSemaphore(semp, millisecs);
      CloseLocalSemaphore(semp);  
      return;
    }
  }
  delay( millisecs );
  return;
  #endif
}

/* ------------------------------------------------------------------ */

#ifdef __cplusplus
extern "C" int __MPKSetThreadAffinity( int cpunum );
#endif

int __MPKSetThreadAffinity( int cpunum ) /* 0... */
{
  if (cpunum >= 0 && cpunum < GetNumberOfRegisteredProcessors())
  {
    MPKExitNetWare();
//ConsolePrintf("isincnb: %d\r\n", __is_thread_in_cnb());
    {
      int (*_kBindThread)(int thrid, int cpuid) = 
              (int (*)(int, int))__get_symbol_ptr(__sym_kBindThread);
      int (*_RunningProcess) = 
                          (int *)__get_symbol_ptr(__sym_RunningProcess);
      if (_kBindThread && _RunningProcess)
      {
        /* documentation is wrong. cpunum is zero based, not one based */
        if (0 == ((*_kBindThread)( (*_RunningProcess), cpunum )))
          return 0;
      }    
      else /* its *CRITICAL* that cpunum is in range here! */
      { 
        void *(*_GetProcessorControlBlock)(unsigned int) = 
         (void *(*)(unsigned int))__get_symbol_ptr(__sym_GetProcessorControlBlock);
        int (*_thread_bind)(int, void *) = 
         (int (*)(int, void *))__get_symbol_ptr(__sym_thread_bind);

        if (_GetProcessorControlBlock && _thread_bind)
        {
          void *pcb = (*_GetProcessorControlBlock)( cpunum ); /* 0... */
          if (pcb)
          {
            (*_thread_bind)( GetThreadID(), pcb );
            return 0;
          }  
        }
      }
    }
  }
  return -1;
}

/* ------------------------------------------------------------------ */

/* there is something whacky about the void prototypes. whats the point of
   inc/dec/add/sub without a way to *read* the value? Ah, well.
*/ 

void atomic_inc(LONG *addr)
{
  _asm mov ecx,addr
  _asm lock inc dword ptr [ecx]
}

void atomic_dec(LONG *addr)
{
  _asm mov ecx,addr
  _asm lock dec dword ptr [ecx]
}

void atomic_add(LONG *addr,LONG value)
{
  _asm mov ecx,addr
  _asm mov eax,value
  _asm lock add [ecx],eax
}

void atomic_sub(LONG *addr,LONG value)
{
  _asm mov ecx,addr
  _asm mov eax,value
  _asm lock sub [ecx],eax
}

LONG atomic_bts(LONG *base,LONG bitoffset)
{
  LONG result;
  _asm mov edx,base
  _asm mov ecx,bitoffset
  _asm xor eax,eax
  _asm lock bts [edx],ecx
  _asm setb al
  _asm mov result,eax
  return result;
}

LONG atomic_btr(LONG *base,LONG bitoffset)
{
  LONG result;
  _asm mov edx,base
  _asm mov ecx,bitoffset
  _asm xor eax,eax
  _asm lock btr [edx],ecx
  _asm setb al
  _asm mov result,eax
  return result;
}

LONG atomic_xchg(LONG *addr,LONG value)
{
  _asm mov ecx,addr
  _asm mov eax,value
  _asm lock xchg [ecx],eax
  _asm mov value,eax
  return value;
}

/* ------------------------------------------------------------------ */
