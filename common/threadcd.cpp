// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: threadcd.cpp,v $
// Revision 1.5  1998/06/14 08:26:57  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.4  1998/06/14 08:13:12  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


/* module history:
   04 June 1998 - slapped together - Cyrus Patel <cyp@fb14.uni-mainz.de>
*/

// This module encapsulates functions for the creation and destruction of
// a single thread - used by client.cpp and piproxy.cpp

static char *id="@(#)$Id: threadcd.cpp,v 1.5 1998/06/14 08:26:57 friedbait Exp $";

#include "threadcd.h"   //includes implementation and porting notes.
#include "sleepdef.h"   //sleep() and usleep()

//-----------------------------------------------------------------------
// ************** implementation and porting notes are in threadcd.h ****
//-----------------------------------------------------------------------

    //destroy a thread (block until dead)
int CliDestroyThread( THREADID cliThreadID )
{
  int rescode = 0;
  #if !defined(OS_SUPPORTS_THREADING)
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
     pthread_join( cliThreadID, NULL);
  #endif
  return rescode;
}

//-----------------------------------------------------------------------




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

#ifdef OS_SUPPORTS_THREADING
static void __thread_shell( void *param )
{
  #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2) //what is FLAT then?
    struct __thread_shell_data FAR *data = (__thread_shell_data FAR *)param;
    void (FAR *tproc)(void *) = data->proc;
    void FAR *tparam = data->param;
  #else
    struct __thread_shell_data *data = (__thread_shell_data *)param;
    void (*tproc)(void *) = data->proc;
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

  //create a thread (block until running) return threadID or NULL if error
THREADID CliCreateThread( void (*proc)(void *), void *param )
{
  THREADID cliThreadID;

  #ifndef OS_SUPPORTS_THREADING
    cliThreadID = NULL;
  #else
    {
    struct __thread_shell_data shelldata;
    void *shellparam = (void *)(&shelldata);
    void (*shellproc)(void *) = __thread_shell;

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

