/* Created by Oliver Roberts <oliver@futaura.co.uk>
**
** $Id: amTime.c,v 1.1.2.1 2001/01/21 15:10:27 cyp Exp $
**
** ----------------------------------------------------------------------
** This file contains all the Amiga specific code related to time
** functions, including obtaining system time, GMT offset, generating
** delays, obtaining monotonic & linear time.
** ----------------------------------------------------------------------
*/

/*
** Using PPC timers for short delays should be ok (and decreases overhead).
** However, there appears to be a bug in PowerUp which causes the kernel to
** hang, so don't use for PowerUp (ok with 46.30+)
*/
//#ifndef __POWERUP__
#define USE_PPCDELAY
//#endif

/*
** Native PowerUP and WarpOS timers are not guaranteed to increment as a wall
** clock would, and are inaccurate especially on overclocked systems.  PowerUp
** also treats a 66.66666MHz bus speed as 66 MHz, which causes timers to not
** be in sync with real time seconds!  Therefore, we better use 68k timers (and
** workaround the slowdown involved because of context switching overhead)
*/
//#define USE_PPCTIMER

/*
** Use PPC timers to calculate system time?  Best not!
*/
//#define USE_PPCSYSTIME


#include "amiga.h"

#ifdef __PPC__
#pragma pack(2)
#endif

#include <exec/types.h>
#include <proto/locale.h>
#include <proto/timer.h>
#include <devices/timer.h>
#include <stdlib.h>

#ifdef __PPC__
#pragma pack()
#endif

#if (defined(__PPC__)) && (defined(__POWERUP__))
#include <powerup/ppclib/time.h>
#endif

static int GMTOffset = -9999;
struct Device *TimerBase = NULL;

#ifdef __PPC__

#ifdef USE_PPCSYSTIME
static struct timeval BaseTime;
#endif

#if defined(__POWERUP__) && (defined(USE_PPCTIMER) || defined(USE_PPCSYSTIME))
static void *TimerObj = NULL;
#endif

#endif /* __PPC__ */

/* allocated on a per task basis */
struct TimerResources
{
   struct timerequest *tr_TimeReq;
   struct MsgPort     *tr_TimePort;
   BOOL                tr_TimerDevOpen;
   UWORD               tr_pad;

   #if defined(__PPC__) && defined(__POWERUP__) && defined(USE_PPCDELAY)
   ULONG               tr_DelaySig;
   #endif
};

static int GetGMTOffset(void);

VOID CloseTimer(VOID)
{
   struct TimerResources *res;

   #ifndef __PPC__
   struct Task *task = FindTask(NULL);
   res = (struct TimerResources *)task->tc_UserData;
   #elif !defined(__POWERUP__)
   struct TaskPPC *task = FindTaskPPC(NULL);
   res = (struct TimerResources *)task->tp_Task.tc_UserData;
   #else
   res = (struct TimerResources *)PPCGetTaskAttr(PPCTASKTAG_EXTUSERDATA);
   #endif

   if (res) {
      #if defined(__PPC__) && defined(__POWERUP__) && defined(USE_PPCDELAY)
      if (res->tr_DelaySig != (ULONG)-1) PPCFreeSignal(res->tr_DelaySig);
      #endif

      if (res->tr_TimeReq) {
         if (res->tr_TimerDevOpen) CloseDevice((struct IORequest *)res->tr_TimeReq);
         DeleteIORequest((struct IORequest *)res->tr_TimeReq);
      }
      if (res->tr_TimePort) DeleteMsgPort(res->tr_TimePort);
      FreeVec(res);

      #ifndef __PPC__
      task->tc_UserData = NULL;
      #elif !defined(__POWERUP__)
      task->tp_Task.tc_UserData = NULL;
      #else
      PPCSetTaskAttr(PPCTASKTAG_EXTUSERDATA,(ULONG)NULL);
      #endif
   }
}

struct Device *OpenTimer(VOID)
{
   struct Device *timerbase = NULL;
   struct TimerResources *res;

   if ((res = (struct TimerResources *)AllocVec(sizeof(struct TimerResources),MEMF_CLEAR|MEMF_PUBLIC))) {
      #ifndef __PPC__
      (FindTask(NULL))->tc_UserData = (APTR)res;
      #elif !defined(__POWERUP__)
      (FindTaskPPC(NULL))->tp_Task.tc_UserData = (APTR)res;
      #else
      PPCSetTaskAttr(PPCTASKTAG_EXTUSERDATA,(ULONG)res);
      #endif

      if ((res->tr_TimePort = CreateMsgPort())) {
         if ((res->tr_TimeReq = (struct timerequest *)CreateIORequest(res->tr_TimePort,sizeof(struct timerequest)))) {
            if (!OpenDevice((unsigned char *)"timer.device",UNIT_VBLANK,(struct IORequest *)res->tr_TimeReq,0)) {
               res->tr_TimerDevOpen = TRUE;
               timerbase = res->tr_TimeReq->tr_node.io_Device;
	    }
         }
      }

      #if defined(__PPC__) && defined(__POWERUP__) && defined(USE_PPCDELAY)
      res->tr_DelaySig = (ULONG)-1;
      if (timerbase) {
         res->tr_DelaySig = PPCAllocSignal((ULONG)-1);
         if (res->tr_DelaySig == (ULONG)-1) timerbase = NULL;
      }
      #endif
   }

   if (!timerbase) CloseTimer();

   return(timerbase);
}

VOID GlobalTimerDeinit(VOID)
{
   CloseTimer();

   #if defined(__PPC__) && defined(__POWERUP__)
   #if defined(USE_PPCTIMER) || defined (USE_PPCSYSTIME)
   if (TimerObj) PPCDeleteTimerObject(TimerObj);
   #endif
   #endif
}

BOOL GlobalTimerInit(VOID)
{
   BOOL done;

   GMTOffset = GetGMTOffset();

   done = ((TimerBase = OpenTimer()) != NULL);

   #ifdef __PPC__
   #ifdef __POWERUP__
   /*
   ** PowerUp
   */
   #if defined(USE_PPCTIMER) || defined(USE_PPCSYSTIME)
   if (done) {
      done = FALSE;
      struct TagItem tags[2] = { {PPCTIMERTAG_CPU, TRUE}, {TAG_END,0} };
      if ((TimerObj = PPCCreateTimerObject(tags))) {
         #ifdef USE_PPCSYSTIME
         GetSysTime(&BaseTime);
         PPCSetTimerObject(TimerObj,PPCTIMERTAG_START,NULL);
         /* add the offset from UNIX to AmigaOS time system */
         BaseTime.tv_sec += 2922 * 24 * 3600 + 60 * GMTOffset;
         #endif
         done = TRUE;
      }
   }
   #endif

   #else
   /*
   ** WarpOS
   */
   #ifdef USE_PPCSYSTIME
   if (done) {
      struct timeval tv;
      GetSysTime(&BaseTime);
      GetSysTimePPC(&tv);
      /* add the offset from UNIX to AmigaOS time system */
      BaseTime.tv_sec += 2922 * 24 * 3600 + 60 * GMTOffset;
      SubTimePPC(&BaseTime,&tv);
   }
   #endif

   #endif
   #endif /* __PPC__ */

   if (!done) {
      GlobalTimerDeinit();
   }

   return(done);
}

void amigaSleep(unsigned int secs, unsigned int usecs)
{
#if !defined(__PPC__) || !defined(USE_PPCDELAY)
   /*
   ** 68K 
   */
   struct TimerResources *res;

   #ifndef __PPC__
   res = (struct TimerResources *)(FindTask(NULL))->tc_UserData;
   #elif !defined(__POWERUP__)
   res = (struct TimerResources *)(FindTaskPPC(NULL))->tp_Task.tc_UserData;
   #else
   res = (struct TimerResources *)PPCGetTaskAttr(PPCTASKTAG_EXTUSERDATA);
   #endif

   res->tr_TimeReq->tr_node.io_Command = TR_ADDREQUEST;
   struct timeval *tv = &res->tr_TimeReq->tr_time;
   tv->tv_secs = secs + (usecs / 1000000);
   tv->tv_micro = usecs % 1000000;
   DoIO((struct IORequest *)res->tr_TimeReq);
#else
#ifndef __POWERUP__
   /*
   ** WarpOS 
   */
   if (secs <= 4294) {
      WaitTime(0,secs*1000000 + usecs);
   } else {
      /* will almost definitely never happen! */
      Delay(secs * TICKS_PER_SECOND + (usecs * TICKS_PER_SECOND / 1000000));
   }
#else
   /*
   ** PowerUp 
   */
   unsigned int ticks = secs * TICKS_PER_SECOND + (usecs * TICKS_PER_SECOND / 1000000);

   if (ticks > 0) {  // PowerUp timer objects can't handle zero!
      struct TagItem tags[4];
      void *timer;
      struct TimerResources *res = (struct TimerResources *)PPCGetTaskAttr(PPCTASKTAG_EXTUSERDATA);

      tags[0].ti_Tag = PPCTIMERTAG_50HZ; tags[0].ti_Data = ticks;
      tags[1].ti_Tag = PPCTIMERTAG_SIGNALMASK; tags[1].ti_Data = 1L << res->tr_DelaySig;
      tags[2].ti_Tag = PPCTIMERTAG_AUTOREMOVE; tags[2].ti_Data = TRUE;
      tags[3].ti_Tag = TAG_END;

      if ((timer = PPCCreateTimerObject(tags))) {
         PPCWait(1L << res->tr_DelaySig);
         PPCDeleteTimerObject(timer);
      }
   }
   else {
      Delay(0);
   }
#endif
#endif
}

static int GetGMTOffset(void)
{
   struct Locale *locale;
   int gmtoffset = 0;

   if (!LocaleBase) LocaleBase = (struct LocaleBase *)OpenLibrary("locale.library",38L);

   if (LocaleBase) {
      if ((locale = OpenLocale(NULL))) {
         gmtoffset = locale->loc_GMTOffset;
         CloseLocale(locale);
      }
   }
   return(gmtoffset);
}

int gettimeofday(struct timeval *tp, struct timezone *tzp)
{
#if !defined(__PPC__) || !defined(USE_PPCSYSTIME)
   /*
   ** 68K 
   */
   if (tp) {
      /* This check required as C++ class init code calls this routine */
      if (!TimerBase) GlobalTimerInit();
      GetSysTime(tp);
      /* add the offset from UNIX to AmigaOS time system */
      tp->tv_sec += 2922 * 24 * 3600 + 60 * GMTOffset;
   }
#else
#ifndef __POWERUP__
   /*
   ** WarpOS
   */
   if (tp) {
      /* This check required as C++ class init code calls this routine */
      if (!TimerBase) GlobalTimerInit();

      GetSysTimePPC(tp);
      AddTimePPC(tp,&BaseTime);
   }
#else
   /*
   ** PowerUp
   */
   if (tp) {
      unsigned long long secs, usecs;

      /* This check required as C++ class init code calls this routine */
      if (!TimerBase) GlobalTimerInit();

      PPCSetTimerObject(TimerObj,PPCTIMERTAG_STOP,NULL);
      PPCGetTimerObject(TimerObj,PPCTIMERTAG_DIFFSECS,&secs);
      PPCGetTimerObject(TimerObj,PPCTIMERTAG_DIFFMICRO,&usecs);
      tp->tv_sec = BaseTime.tv_sec + secs;
      tp->tv_usec = BaseTime.tv_usec + (usecs - secs * 1000000);
      if (tp->tv_usec >= 1000000) {
         tp->tv_sec++;
         tp->tv_usec -= 1000000;
      }
   }
#endif
#endif

   if (tzp) {
      tzp->tz_minuteswest = -GMTOffset;
      tzp->tz_dsttime = DST_NONE;
   }

   return 0;
}

/*
** Since this routine is called frequently, especially during cracking, we
** need this routine to have as little overhead as possible.  On the PPC,
** this means using PPC native kernel calls (no context switches to 68k)
*/
int amigaGetMonoClock(struct timeval *tp)
{
//   printf("GetMonoClock()\n");
   if (tp) {
#if !defined(__PPC__) || !defined(USE_PPCTIMER)
      /*
      ** 68K
      */
      struct EClockVal ec;
      ULONG efreq = ReadEClock(&ec);
      unsigned long long *etime = (unsigned long long *)&ec;
      tp->tv_sec = *etime / efreq;
      tp->tv_usec = (*etime % efreq) * 1000000 / efreq;
#else
#ifndef __POWERUP__
      /*
      ** WarpOS
      */
      GetSysTimePPC(tp);  // contrary to the autodoc, this is not actually
                          // system time, but a linear monotonic clock
#else
      /*
      ** PowerUp
      */
      unsigned long long ppctime, tickspersec;
      if (!TimerBase) GlobalTimerInit();
      PPCGetTimerObject(TimerObj,PPCTIMERTAG_TICKSPERSEC,&tickspersec);
      PPCGetTimerObject(TimerObj,PPCTIMERTAG_CURRENTTICKS,&ppctime);
      tp->tv_sec = ppctime / tickspersec;
      tp->tv_usec = (ppctime % tickspersec) * 1000000 / tickspersec;   
#endif
#endif
   }

   return 0;
}

/*
** libnix mktime() has broken leap year handling, so use this instead
*/
time_t mktime(struct tm *tm)
{
   static int mon[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
   time_t  t;
   int notLeap,i;

   /*
    *  1976 was a leap year, take quad years from 1976, each
    *  366+365+365+365 days each.  Then adjust for the year
    */

   t = ((tm->tm_year - 76) / 4) * ((366 + 365 * 3) * 86400);

   /*
    *  compensate to make it work the same as unix time (unix time
    *  started 8 years earlier)
    */

   t += 2922 * 24 * 60 * 60;

   /*
    *  take care of the year within a four year set, add a day for
    *  the leap year if we are based at a year beyond it.
    */

   t += ((notLeap = (tm->tm_year - 76) % 4)) * (365 * 86400);

   if (notLeap) t += 86400;

   /*
    *  calculate days over months then days offset in the month
    */

   for (i = 0; i < tm->tm_mon; ++i) {
      t += mon[i] * 86400;
      if (i == 1 && notLeap == 0) t += 86400;
   }
   t += (tm->tm_mday - 1) * 86400;

   /*
    *  our time is from 1978, not 1976
    */

   t -= (365 + 366) * 86400;

   /*
    *  calculate hours, minutes, seconds
    */

   t += tm->tm_hour * 3600 + tm->tm_min * 60 + tm->tm_sec;

   return(t);
}
