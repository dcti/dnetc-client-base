/*
 * Copyright distributed.net 1997-2002 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: amGUI.c,v 1.2.4.1 2003/04/03 21:14:32 oliver Exp $
 *
 * Created by Oliver Roberts <oliver@futaura.co.uk>
 *
 * ----------------------------------------------------------------------
 * This file contains the API and support code for the GUI mode
 * ----------------------------------------------------------------------
*/

#ifndef NOGUI

#include "amiga.h"
#include "cputypes.h"
#include "triggers.h"
#include "modereq.h"
#include "problem.h"
#include "probfill.h"
#include "probman.h"
#include "clitime.h"
#include "clicdata.h"
#include "cliident.h"

#include "proto/dnetcgui.h"

struct Library *DnetcBase;

static ULONG GUISigMask;

#if 0
static struct ContestData {
   struct {
      unsigned long rate[120]; /* 120 seconds */
      int numcrunchers;
      struct 
      {
        long   threshold;
        int    thresh_in_swu;
        long   blk_count;
        long   swu_count;
        time_t till_completion;
      } buffers[2];
      u32 last_ratelo;
      /* sizeof avgrate array => tray refresh interval in secs */
      struct {u32 hi,lo;} avgrate[5]; 
      unsigned int avgrate_count;
   } cdata[CONTEST_COUNT];
   unsigned int cont_sel;
   int cont_sel_uncertain;
   int cont_sel_explicit;
} cd;

static struct ContestData *dd = &cd;

extern int LogGetContestLiveRate(unsigned int contest_i,
                                 u32 *ratehiP, u32 *rateloP,
                                 u32 *walltime_hiP, u32 *walltime_loP,
                                 u32 *coretime_hiP, u32 *coretime_loP);

static void __amigaDoInfoUpdates(void)
{
   int crunch_count_change = 0, buffers_changed = 0;
   int rate_cont_i = -1; u32 rate_ratehi = 0, rate_ratelo = 0;
   int cont_i;
   struct timeval tv;
   char buffer[256];
   Problem *selprob;

   CliGetMonotonicClock(&tv);

   for (cont_i = 0; cont_i < CONTEST_COUNT; cont_i++) {
      u32 ratehi,ratelo,wtimehi,wtimelo,ctimehi,ctimelo; 
      unsigned long efficiency = 0;
      int numcrunchers;
      int curpos;

      numcrunchers = LogGetContestLiveRate(cont_i, &ratehi, &ratelo,
                   &wtimehi, &wtimelo, &ctimehi, &ctimelo);

      if (numcrunchers < 1) {
         ratehi = ratelo = 0;
         numcrunchers = 0;
      }
      else {
         wtimelo = (wtimelo / 1000)+(wtimehi * 1000);
         ctimelo = ((ctimelo+499) / 1000)+(ctimehi * 1000);
         if (wtimelo) {
            efficiency = (((unsigned long)ctimelo) * 1000ul)/wtimelo;
            /*if (efficiency > dd->effmax)
               efficiency = dd->effmax;*/
	 }
      }
      if (efficiency == 0)
         efficiency = 1;

      if (cont_i == RC5) {
         int out_buffer_changed = 0;
         rate_cont_i = cont_i;
         rate_ratehi = ratehi;
         rate_ratelo = ratelo;

         if (!IsProblemLoadPermitted(-1, cont_i))
         {
           ; /* nothing */
         }
         else if (buffers_changed || crunch_count_change ||
                  (tv.tv_sec % 5) == 0)
         {
           int sel_buf;
           for (sel_buf = 0; sel_buf < 2; sel_buf++)
           {
             long threshold, blk_count, swu_count; 
             unsigned int till_completion;
             int thresh_in_swu; 
             if (ProbfillGetBufferCounts( cont_i, sel_buf,
                          &threshold, &thresh_in_swu,
                          &blk_count, &swu_count, &till_completion )>=0)
             {                               
               if (buffers_changed 
                   || dd->cdata[cont_i].buffers[sel_buf].threshold != threshold
                   || dd->cdata[cont_i].buffers[sel_buf].thresh_in_swu != thresh_in_swu
                   || dd->cdata[cont_i].buffers[sel_buf].blk_count != blk_count
                   || dd->cdata[cont_i].buffers[sel_buf].swu_count != swu_count
                   || dd->cdata[cont_i].buffers[sel_buf].till_completion != (int)till_completion)
               {
                  dd->cdata[cont_i].buffers[sel_buf].threshold = threshold;
                  dd->cdata[cont_i].buffers[sel_buf].thresh_in_swu = thresh_in_swu;
                  dd->cdata[cont_i].buffers[sel_buf].blk_count = blk_count;
                  dd->cdata[cont_i].buffers[sel_buf].swu_count = swu_count;
                  dd->cdata[cont_i].buffers[sel_buf].till_completion = till_completion;
                  buffers_changed = 1;
                  if (sel_buf == 1)
                    out_buffer_changed = 1;
               }
               sprintf(buffer,"buf %d: threshold %d, thresh_in_swu %d, blk_cnt = %d, swu_cnt %d, till_completion %d\n",
                       sel_buf,threshold,thresh_in_swu,blk_count,swu_count,till_completion);
               amigaConOut(buffer);
             }
           } /* for sel_buf ... */
         } /* need buffer level check */ 

         if (buffers_changed || ratelo > dd->cdata[cont_i].last_ratelo)
         {
              if (out_buffer_changed)
              {
                u32 iterhi, iterlo;
                unsigned int packets, swucount;
                struct timeval ttime;

                if (CliGetContestInfoSummaryData( cont_i, 
                     &packets, &iterhi, &iterlo,
                     &ttime, &swucount ) == 0)
                {
                  ProblemComputeRate( cont_i, ttime.tv_sec, ttime.tv_usec, 
                            iterhi, iterlo, 0, 0, buffer, sizeof(buffer) );
                  char *p = strchr(buffer,' ');
                  if (p) *p = '\0';
                  /*SetDlgItemText(dialog,IDC_SUM_RATE,buffer);
                  SetDlgItemInt(dialog,IDC_SUM_PKTS,(UINT)packets,FALSE);
                  sprintf(buffer, "%u.%02u", swucount/100, swucount%100);
                  SetDlgItemText(dialog,IDC_SUM_SWU,buffer);
                  ll = ttime.tv_sec;
                  sprintf( buffer,  "%d.%02d:%02d:%02d", (ll / 86400UL),
                           (int) ((ll % 86400L) / 3600UL), 
                           (int) ((ll % 3600UL)/60),
                           (int) (ll % 60) );
                  SetDlgItemText(dialog,IDC_SUM_TIME,buffer);*/
                }
              } /* if (out_buffer_changed) */
	 }
      }
   }

   selprob = GetProblemPointerFromIndex(0);
   if (selprob && ProblemIsInitialized(selprob)) {
      ProblemInfo info;
      if (ProblemGetInfo(selprob, &info, P_INFO_C_PERMIL | P_INFO_CCOUNT |
                                         P_INFO_SWUCOUNT) >= 0)
      {
         sprintf(buffer,"permille %d, ccounthi %d, ccountlo %d, swucount %d\n",
                 info.c_permille,info.ccounthi,info.ccountlo,info.swucount);
         amigaConOut(buffer);
      }
   }
}
#endif

#if !defined(__PPC__)
void amigaHandleGUI(struct timerequest *tr)
#elif !defined(__POWERUP__)
void amigaHandleGUI(struct timeval *tv)
#else
#undef SetSignal
#define SetSignal(newSignals, signalSet) \
	LP2(0x132, ULONG, SetSignal, unsigned long, newSignals, d0, unsigned long, signalSet, d1, \
	, EXEC_BASE_NAME, IF_CACHEFLUSHALL, NULL, 0, IF_CACHEFLUSHALL, NULL, 0)
void amigaHandleGUI(void *timer, ULONG timesig)
#endif
{
   BOOL done = FALSE;

   #ifndef __PPC__
   struct MsgPort *tport;
   ULONG waitsigs,timesig;

   if (tr) {
      tport = tr->tr_node.io_Message.mn_ReplyPort;
      timesig = 1L << tport->mp_SigBit;
   }
   else {
      timesig = 0;
   }

   waitsigs = timesig | GUISigMask;
   #elif defined(__POWERUP__)
   ULONG waitsigs = timesig;
   #else
   struct timeval tvend,tvnow;
   GetSysTimePPC(&tvend);
   AddTimePPC(&tvend,tv);
   #endif

   do {

      ULONG sigr;

      #ifndef __PPC__
      sigr = Wait(waitsigs);

      if (sigr & timesig) {
         done = TRUE;
         GetMsg(tport);
      }
      #elif defined(__POWERUP__)
      PPCWait(timesig);
      PPCDeleteTimerObject(timer);;
      done = TRUE;
      sigr = SetSignal(0,GUISigMask);
      #else
      sigr = WaitTime(GUISigMask,tv->tv_sec*1000000 + tv->tv_micro);
      #endif

      if (sigr & GUISigMask) {
         ULONG cmds = dnetcguiHandleMsgs(sigr);

         if ( cmds & DNETC_MSG_SHUTDOWN )
            RaiseExitRequestTrigger();
         if ( cmds & DNETC_MSG_RESTART )
            RaiseRestartRequestTrigger();
         if ( cmds & DNETC_MSG_PAUSE )
            RaisePauseRequestTrigger();
         if ( cmds & DNETC_MSG_UNPAUSE )
            ClearPauseRequestTrigger();
         if ( cmds & DNETC_MSG_FETCH )
            ModeReqSet(MODEREQ_FETCH);
         if ( cmds & DNETC_MSG_FLUSH )
            ModeReqSet(MODEREQ_FLUSH);
         if ( cmds & DNETC_MSG_BENCHMARK )
            ModeReqSet(MODEREQ_BENCHMARK);
         if ( cmds & DNETC_MSG_BENCHMARK_ALL )
            ModeReqSet(MODEREQ_BENCHMARK_ALLCORE);
         if ( cmds & DNETC_MSG_TEST )
            ModeReqSet(MODEREQ_TEST);
         if ( cmds & DNETC_MSG_CONFIG )
            ModeReqSet(MODEREQ_CONFIG | MODEREQ_CONFRESTART);

         #ifndef __PPC__
         if (cmds && !done && tr) {
            AbortIO((struct IORequest *)tr);
            WaitIO((struct IORequest *)tr);
            done = TRUE;
	 }
         #elif defined(__POWERUP__)
         if (cmds && !done && timer) {
            PPCDeleteTimerObject(timer);
            done = TRUE;
	 }
         #else
         if (cmds) done = TRUE;
         #endif
      }

      //if (!ModeReqIsSet(-1)) __amigaDoInfoUpdates();

      #if defined(__PPC__) && !defined(__POWERUP__)
      if (sigr && !done) {
         GetSysTimePPC(&tvnow);
         if (CmpTimePPC(&tvnow,&tvend) == 1) {
            tv->tv_sec = tvend.tv_sec;
            tv->tv_micro = tvend.tv_micro;
            SubTimePPC(tv,&tvnow);
	 }
         else {
            done = TRUE;
	 }
      }
      else {
         done = TRUE;
      }
      #endif

   } while (!done);
}

void amigaGUIOut(char *msg)
{
   char *end;
   int len, c;
   static BOOL prevnewline = TRUE, overwrite = FALSE;

   while (*msg) {
      len = strcspn(msg,"\r\n");
      end = msg + len;
      c = *end;
      if (c == '\r') {
         overwrite = !prevnewline;
         msg = end + 1;
         continue;
      }
      else if (c == '\n') {
         if (!prevnewline) {
            if (len == 0) {
               prevnewline = TRUE;
               msg++;
               continue;
	    }
	 }
         *end = '\0';
         prevnewline = TRUE;
      }
      else {
         prevnewline = FALSE;
      }
      dnetcguiConsoleOut(CLIENT_CPU,msg,overwrite);
      msg += len + (c ? 1 : 0);
      overwrite = FALSE;
   }
}

BOOL amigaGUIInit(char *programname, struct WBArg *iconname)
{
   BOOL guiopen = FALSE;

   if ((DnetcBase = OpenLibrary("dnetcgui.library",1))) {
      if ((GUISigMask = dnetcguiOpen(((CLIENT_CPU == CPU_POWERPC) ? DNETCGUI_PPC : DNETCGUI_68K),programname,iconname,CliGetFullVersionDescriptor()))) {
         guiopen = TRUE;
      }
      else {
         CloseLibrary(DnetcBase);
         DnetcBase = NULL;
      }
   }

   return(guiopen);
}

void amigaGUIDeinit(void)
{
   if (DnetcBase) {
      dnetcguiClose(NULL);
      CloseLibrary(DnetcBase);
      DnetcBase = NULL;
   }
}

#endif /* NOGUI */
