/*
 * Copyright distributed.net 1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This module encapsulates functions for the creation and destruction of
 * a single thread - used by client.cpp and piproxy.cpp
 * implementation and porting notes are in threadcd.h             - cyp
 *
*/
const char *threadcd_cpp(void) {
return "@(#)$Id: threadcd.cpp,v 1.16 1999/04/15 21:57:42 trevorh Exp $"; }

#include <stdio.h>      //NULL
#include "threadcd.h"   //includes implementation and porting notes.
#include "baseincs.h"
#include "sleepdef.h"   //sleep() and usleep()

/* ---------------------------------------------------------------------- */

/* destroy a thread (block until dead) */
int CliDestroyThread( THREADID cliThreadID )
{
  int rescode = 0;
  #if (!defined(OS_SUPPORTS_THREADING) || !defined(THREADING_IS_AVAILABLE))
     rescode = ((cliThreadID)?(1):(0)); //suppress compiler warning
     rescode = -1;
  #elif (CLIENT_OS == OS_OS2)
     DosWaitThread( &cliThreadID, DCWW_WAIT);
  #elif (CLIENT_OS == OS_WIN32)
     WaitForSingleObject((HANDLE)cliThreadID, INFINITE);
  #elif (CLIENT_OS == OS_BEOS)
     {
     status_t be_exit_value;
     wait_for_thread( cliThreadID, &be_exit_value );
     }
  #elif (CLIENT_OS == OS_NETWARE)
     CliWaitForThreadExit( cliThreadID );
  #else
     pthread_join( cliThreadID, (void**)NULL);
  #endif
  return rescode;
}

/* --------------------------------------------------------------------- */

// The thread is wrapped in a shell for two reasons:
// 1. we can block until the thread has started
// 2. from the caller's perspective, the thread proc is called like a
//    local function (no exit() etc required)

struct __thread_shell_data
{
  int started;
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
    void (FAR *proc)(void *);
    void FAR *param;
  #else
    void (*proc)(void *);
    void *param;
  #endif
};

#if (defined(OS_SUPPORTS_THREADING) && defined(THREADING_IS_AVAILABLE))
static void __thread_shell( void *param )
{
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) //what is FLAT then?
    struct __thread_shell_data FAR *data = (__thread_shell_data FAR *)param;
    void (FAR *tproc)(void *) = data->proc;
    void FAR *tparam = data->param;
  #else
    struct __thread_shell_data *data = (__thread_shell_data *)param;
    register void (*tproc)(void *) = data->proc;
    void *tparam = data->param;
  #endif

  data->started = 1; //the pointers are safe, so let the caller go on

  (*tproc)( tparam );

  #if (CLIENT_OS == OS_BEOS)
    exit(0);
  #endif
  return;
}
#endif // OS_SUPPORTS_THREADING

//-----------------------------------------------------------------------

// create a thread (block until running) return threadID or NULL if error
THREADID CliCreateThread( register void (*proc)(void *), void *param )
{
  THREADID cliThreadID;

  #if (!defined(OS_SUPPORTS_THREADING) || !defined(THREADING_IS_AVAILABLE))
    cliThreadID = (param==NULL || proc==NULL); //suppress compiler warns
    cliThreadID = (THREADID) NULL;
  #else
    {
    struct __thread_shell_data shelldata;
    void *shellparam = (void *)(&shelldata);
    register void (*shellproc)(void *) = __thread_shell;

    shelldata.started = 0;
    shelldata.proc = proc;
    shelldata.param = param;

    #if (CLIENT_OS == OS_WIN32)
       {
       cliThreadID = _beginthread( shellproc, 8192, shellparam );
       //if ( cliThreadID == 0) cliThreadID = NULL; //0
       }
    #elif (CLIENT_OS == OS_OS2)
       {
       cliThreadID = _beginthread( shellproc, NULL, 8192, shellparam );
       if ( cliThreadID == -1) cliThreadID = NULL; //0
       }
    #elif (CLIENT_OS == OS_NETWARE)
       {
       cliThreadID = BeginThread( shellproc, NULL, 8192, shellparam );
       if ( cliThreadID == -1) cliThreadID = NULL; //0
       }
    #elif (CLIENT_OS == OS_BEOS)
       {
       static int threadindex = 0;
       char thread_name[32];
       long be_priority = B_LOW_PRIORITY;

       sprintf(thread_name, "RC5DES#%d ", ++threadindex );
       cliThreadID = spawn_thread((long (*)(void *))shellproc,
                             thread_name, be_priority, shellparam );
       if (cliThreadID < B_NO_ERROR || resume_thread(cliThreadID)!=B_NO_ERROR)
         cliThreadID = NULL; //0
       }
    #elif defined(_POSIX_THREAD_PRIORITY_SCHEDULING)
       {
       pthread_attr_t thread_sched;

       pthread_attr_init( &thread_sched );
       pthread_attr_setscope( &thread_sched, PTHREAD_SCOPE_SYSTEM );
       pthread_attr_setinheritsched( &thread_sched, PTHREAD_INHERIT_SCHED );

       if (pthread_create( &cliThreadID, &thread_sched,
                       (void *(*)(void*))shellproc, shellparam))
         cliThreadID = (pthread_t) NULL; //0
       }
    #else
       {
       if (pthread_create( &cliThreadID, NULL,
                        (void *(*)(void*))shellproc, shellparam))
         cliThreadID = (pthread_t) NULL; //0
       }
    #endif

    if ((void*)cliThreadID != NULL)
      {
      while (!shelldata.started)
        usleep(10000);
      }
    }
  #endif //OS_SUPPORTS_THREADING

  return cliThreadID;
}

//-----------------------------------------------------------------------

