/* Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * --------------------------------------------------------------------
 * The polling process is essentially an event handler that gets called 
 * when "idle events" get queued, ie the main process sleeps. Procedures
 * registered with the polling process then get called synchronously 
 * (subject to priority and round-robin order).
 *
 * The advantages, from the client's perspective, are many-fold: 
 * For one, the line between "multi-threaded" and "single-threaded" clients 
 * vanishes (silly terminology from the client's perspective anyway; a
 * threaded client is not faster that a non-threaded one on a single
 * processor machine), which cuts down on the need to maintain separate
 * source or binaries. Secondly, its more efficient on platforms that do
 * not support threading: Multiple 'small problems' running back-to-back 
 * have less overhead that 'big problems' (still <3 seconds) since the
 * time-till-stop can be controlled precisely. Thirdly, it is more efficient 
 * on platforms that do support threading, but are non-preemptive. That is, 
 * the number of context switches can be kept at a minimum while still 
 * maintaining exact control over when when the client _should_ yield.
 * Fourthly, it is useful even in preemptive environments since routine 
 * work can be delegated (with a run-at time stamp if necessary) without
 * having to spin off another thread and without having oodles of 
 * platform-specific code mucking about in real client code. Fifth: it
 * does away with the 'timeslice factor' crutch.
 *
 * PolledSleep() and PolledUSleep() are automatic/default replacements for 
 * sleep() and usleep() (see sleepdef.h) and allow the polling process to run.
 *
 * NonPolledSleep() and NonPolledUSleep() are "real" sleepers. This are 
 * required for real threads (a la Go_mt()) that need to yield control to 
 * other threads.
 *
 * Created by Cyrus Patel (cyp@fb14.uni-mainz.de) 
 * --------------------------------------------------------------------
*/
const char *pollsys_cpp(void) {
return "@(#)$Id: pollsys.cpp,v 1.13 2000/06/02 06:24:58 jlawson Exp $"; }

#include "baseincs.h"  /* NULL, malloc */
#include "clitime.h"   /* CliTimer() */
#include "pollsys.h"   /* ourselves to keep prototypes in sync */
#define __SLEEP_FOR_POLLING__
#include "sleepdef.h"

#ifdef USLEEP_IS_SLEEP
#include "network.h"
#undef usleep
void usleep( unsigned int usecs )
{
  struct timeval tv;
  tv.sec = 0; tv.usec = usecs;
  select(0,NULL,NULL,NULL,&tv);
}
#endif

/* -------------------------------------------------------------------- */

#define MAX_POLL_RUNLEVEL 32 /* ie priority, 0 is lowest/slowest */

struct polldata
{
  struct polldata *next;
  unsigned int priority; /* 0 is lowest/slowest */
  int chainhead;
  int inuse;
  int fd;
  void (*proc)(void *);
  void *arg;
  struct timeval execat; 
};

static struct
{
  struct polldata *runlist;
  unsigned int regcount;
  struct polldata *nextrun[MAX_POLL_RUNLEVEL+1];
} pollsysdata = { NULL, 0, {NULL}};


/* ---------------------------------------------------------------------- */
/*
   RegPolledProcedure(). Procedures are auto unregistered when executed.
*/   

int UnregPolledProcedure( int fd )
{
  struct polldata *thisp;
  int rc = -1;
  
  thisp = pollsysdata.runlist;
  while ( thisp )
  {
    if (thisp->fd == fd)
    {
      if (thisp->inuse)
      {
        thisp->inuse = 0;
        rc = 0;
      }
      pollsysdata.regcount--;
      break;
    }
    thisp = thisp->next;
  }
  return rc;
}  

/*
  RegPolledProcedure() adds a procedure to be called from the polling loop.
  Procedures may *not* use sleep() or usleep() directly! (Its a stack issue, 
  not a reentrancy problem). Procedures are automatically unregistered 
  when called (they can re-register themselves). The 'interval' argument 
  specifies how much time must elapse before the proc is scheduled to run - 
  the default is {0,0}, ie schedule as soon as possible. Returns a non-zero 
  handle on success or -1 if error. Care should be taken to ensure that
  procedures registered with a high priority have an interval long enough
  to allow procedures with a low(er) priority to run.
*/  

int RegPolledProcedure( auto void (*proc)(void *), void *arg, 
                        struct timeval *interval, unsigned int priority ) 
{
  struct polldata *thatp, *thisp, *chaintail;
  unsigned int i, mcount;
  int fd = -1;

  if (proc)
  {
    thisp = pollsysdata.runlist;
    chaintail = NULL;
    while ( thisp )
    {
      fd = thisp->fd;
      if (!thisp->inuse)
        break;
      chaintail = thisp;
      thisp = thisp->next;
    }
    if ( !thisp )
    {
      if (!pollsysdata.runlist)
      {
        for (i = 0; i < (sizeof(pollsysdata.nextrun)/
            sizeof(pollsysdata.nextrun[0])); i++)
          pollsysdata.nextrun[i]=NULL;
      }
      mcount = 1024/(sizeof(struct polldata));
      thisp = (struct polldata *)(malloc( mcount * sizeof(struct polldata) ));
      if ( thisp )
      {
        thatp = thisp;
        if (fd == -1)
          fd = 0;
        for (i = 0; i < mcount; i++)
        {
          thatp->next = (i<(mcount-1))?(thatp+1):(NULL);
          thatp->chainhead = (i==0);
          thatp->proc = NULL;  
          thatp->inuse = 0;  
          thatp->fd = ++fd;
          thatp++;
        }
        if (pollsysdata.runlist == NULL)
          pollsysdata.runlist = thisp;
        else
          chaintail->next = thisp;
      }
    }
    if ( !thisp )
      fd = -1;
    else
    {
      pollsysdata.regcount++;
      fd = thisp->fd;
      thisp->inuse = 1;
      thisp->proc = proc;
      thisp->arg  = arg;
      thisp->priority = priority;
      if (priority >= MAX_POLL_RUNLEVEL)
        thisp->priority = MAX_POLL_RUNLEVEL-1;
      CliTimer( &thisp->execat );
      if (interval)
      {
        thisp->execat.tv_sec += interval->tv_sec;
        thisp->execat.tv_usec += interval->tv_usec;
        thisp->execat.tv_sec += thisp->execat.tv_usec/1000000;
        thisp->execat.tv_usec %= 1000000;
      }
    }
  }
  return (fd);
}  


static void __initchk(void *dummy) { dummy = dummy; }
int InitializePolling(void)
{ 
  int fd;
  if (pollsysdata.runlist != NULL)
    return 0;
  if ((fd = RegPolledProcedure( __initchk, NULL, NULL, 0 )) != -1)
  {
    UnregPolledProcedure( fd ); /* remove it from the queue */
    return 0;
  }
  return -1;
}


int DeinitializePolling(void)
{
  struct polldata *thatp, *thisp;

  thisp = pollsysdata.runlist;
  thatp = pollsysdata.runlist = NULL;

  while ( thisp )
  {
    if ( thisp->chainhead )
    {
      if ( thatp )
        free((void *)(thatp));
      thatp = thisp;
    }
    thisp = thisp->next;
    if (!thisp && thatp)
      free((void *)(thatp));
  }
  return 0;
}  

void __RunPollingLoop( unsigned int secs, unsigned int usecs )
{
  static unsigned int isrunning = 0;
  struct timeval now, until;
  struct polldata *thisp, *nextp = NULL;
  void *arg = NULL;
  unsigned int runprio;
  register void (*proc)(void *) = NULL;
  int reclock, loopend, dorun;

  if ((++isrunning) > 1)
  {
    fprintf(stderr, "call to sleep when no sleep allowed!");
  }
  else if (!pollsysdata.runlist || pollsysdata.regcount==0)
  {
    if ( secs )
    {
      sleep( secs );
    }
    else
    {
      usleep( usecs );
    }
  }
  else
  {
    CliTimer( &now );
    until.tv_usec = now.tv_usec;
    until.tv_sec = now.tv_sec;
    until.tv_usec += usecs;
    until.tv_sec += secs;
    until.tv_sec += until.tv_usec / 1000000;
    until.tv_usec %= 1000000;
  
    runprio = MAX_POLL_RUNLEVEL;
    
    do
    {
      thisp = NULL;
      reclock = 0;
//printf("o%d", runprio);

      do
      {
        dorun = 0;
//printf("i");
      
        //could lock MUTEX here 
        if ( !pollsysdata.runlist )
        {
          loopend = 1;
          reclock = 0;
        }
        else
        {
          //lock MUTEX here
          if ( !thisp )
          {
            if ( !pollsysdata.nextrun[runprio] )
              pollsysdata.nextrun[runprio] = pollsysdata.runlist;
            thisp = pollsysdata.nextrun[runprio];
          }
          if ((nextp = thisp->next) == NULL)
            nextp = pollsysdata.runlist;
          loopend = ( !nextp || 
                (pollsysdata.nextrun[runprio])->fd == nextp->fd );
          if (thisp->inuse)
          {
            if ((thisp->priority == runprio) &&
              (( now.tv_sec > thisp->execat.tv_sec ) ||
              (( thisp->execat.tv_sec == now.tv_sec ) && 
              ( now.tv_usec >= thisp->execat.tv_usec ))))
            {
              arg = thisp->arg;
              proc = thisp->proc;
              thisp->inuse = 0;
              pollsysdata.regcount--;
              dorun = 1;
              reclock = 1;
              pollsysdata.nextrun[runprio] = nextp;
              loopend = 1;
//printf("r%dp%d", thisp->fd, thisp->priority );
            }
          }
        }
        //could unlock MUTEX here
        
        if (dorun)
          (*proc)(arg);
        thisp = nextp;
      } while (!loopend);
      
      if (reclock) /* ran at that level */
        runprio = MAX_POLL_RUNLEVEL; /* start over */
      else if (runprio > 0)
        runprio--;
      else
      {
        runprio = MAX_POLL_RUNLEVEL;
        if (( now.tv_sec < until.tv_sec ) || (( now.tv_sec == until.tv_sec ) 
          && ( now.tv_usec < until.tv_usec )))
          usleep(100); 
        reclock = 1;
      }

      if (reclock)
        CliTimer( &now );
           
    } while (( now.tv_sec < until.tv_sec ) || 
              (( now.tv_sec == until.tv_sec ) && 
               ( now.tv_usec < until.tv_usec )));
  }
    
  --isrunning;
  return;           
}

// PolledSleep() and PolledUSleep() are automatic/default replacements for 
// sleep() and usleep() (see sleepdef.h) and allow the polling process to run.

void PolledSleep( unsigned int seconds )   
{ __RunPollingLoop( seconds, 0 ); }

void PolledUSleep( unsigned int useconds)  
{ __RunPollingLoop( 0, useconds ); }

// NonPolledSleep() and NonPolledUSleep() are "real" sleepers. This are 
// required for real threads (a la Go_mt()) that need to yield control to 
// other threads.

void NonPolledSleep( unsigned int seconds) 
{ sleep( seconds ); }

void NonPolledUSleep(unsigned int useconds)
{ usleep( useconds ); }

