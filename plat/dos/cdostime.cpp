/* 
 * ------------------------------------------------------------------
 * POSIX gettimeofday() and clock_gettime() with 
 * real micro/nanosec granularity/resolution.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * NB: gettimeofday caveat: the timezone.tz_dsttime member is not accurate.
 * There is no way to translate the ANSI timezone variable to the
 * dsttime DST_* code.
 * ------------------------------------------------------------------
*/
const char *cdostime_cpp(void) {
return "@(#)$Id: cdostime.cpp,v 1.1.2.1 2001/01/21 15:10:23 cyp Exp $"; }

//#define DEBUG_MONOTONIC
//#define DEBUG_TIMEOFDAY

#include <time.h>
#include <sys/timeb.h>
#include "cdostime.h" /* keep prototypes in sync */

#undef timezone       /* undo our timezone define hackery */
extern long timezone; /* (namespace collision with 'struct timezone') */

#if defined(DEBUG_MONOTONIC) || defined(DEBUG_TIMEOFDAY)
#include <stdio.h> /* for debugging */
#include <math.h>
#define TRACE
#include "util.h"
#endif

/* --------------------------------------------------------------------- */

static void __get_timezone_info( int *minutes_west, int *dsttype, 
                                 long *abs_secs_west )
{
  /* abs_secs_west is equivalent of utctime-minus-localtime, ie the number */
  /* of secs to subtract from time(NULL) to get localtime, or inversely the */
  /* number of secs to add to localtime to get utctime */

  int minwest, dstflag;

  #if 0
  time_t timenow; struct tm * tmP;
  timenow = time(NULL);
  tmP = localtime( (const time_t *) &timenow);
  dstflag = minwest = 0;
  if (tmP)
  {
    struct tm loctime, utctime;
    memcpy( &loctime, tmP, sizeof( struct tm ));
    tmP = gmtime( (const time_t *) &timenow);
    if (tmP)
    { 
      memcpy( &utctime, tmP, sizeof( struct tm ));
      minwest =  ((utctime.tm_min  - loctime.tm_min) )
                +((utctime.tm_hour - loctime.tm_hour)*60 );
      /* last two are when the time is on a year boundary */
      if      (loctime.tm_yday == utctime.tm_yday)     { ;/* no change */ }
      else if (loctime.tm_yday == utctime.tm_yday + 1) { minwest -= 1440; }
      else if (loctime.tm_yday == utctime.tm_yday - 1) { minwest += 1440; }
      else if (loctime.tm_yday <  utctime.tm_yday)     { minwest -= 1440; }
      else                                             { minwest += 1440; }
      if (utctime.tm_isdst>0)  
        minwest-=60;
      if (minwest < -(12*60)) 
        minwest = -(12*60); 
      else if (minwest > +(12*60)) 
        minwest = +(12*60);
      dstflag = (utctime.tm_isdst>0);
    }
  }
  #else
  minwest = (((long)timezone)/60L); /* this is right for Watcom anyway */
  dstflag = (daylight != 0);
  #endif

  if (minutes_west)  *minutes_west = minwest;
  if (dsttype)       *dsttype  = (dstflag) ? (1) : (0); /* 1==DST_MET */
  if (abs_secs_west) *abs_secs_west= 60L*(long)(minwest-((dstflag)?(60):(0)));
  return;
}

/* --------------------------------------------------------------------- */

static void __convert_ticks_and_pit_to_timespec( unsigned long days,
                                                 unsigned long ticks,
                                                 unsigned int pit,
                                                 struct timespec *tsp )
{
   unsigned long secs, nsecs, rticks;

   #define TICKS_IN_90_MIN (0x1800B0ul >> 4) /* 0x1800B (98315) */
   #define SECS_IN_90_MIN  (86400ul >> 4)  /* 5400 (0x1518) */
   /* 
   ** the reason nothing else is defined is because the asm code will
   ** need adjustment even if one of the defines changes. The assumptions
   ** in effect are as follows:
   */
   /* 1 tick = 0.054,925,494,583,735,950,770,482,632,355,185 secs (exactly) */
   /*          (obtained from 86,400,000,000,000ns/day / 1800B0 ticks/day)  */
   /* 1 pit  = 0.000,000,838,096,536,006,713,116,004,678,838 secs (enough!) */
   /*          (obtained from [default rate] 1 tick / 0x10000)              */
   /*          if the rate is not the default, then scale it, eg:           */
   /*          if rate is twice the default then pass (pitcount * 2)        */
   /* Unlike ticks which we compute exactly, we only compute the pitcount   */
   /* to 0.000,000,838,096,536,006th/sec. The rest is worthless anyway.     */

   days += ticks / 0x1800B0ul;
   ticks %= 0x1800B0ul;
   ticks += pit / 0x10000ul;
   pit %= 0x10000ul;

   nsecs = 0;
   secs = (86400ul * days) + (SECS_IN_90_MIN * (ticks / TICKS_IN_90_MIN));
   rticks = ticks % TICKS_IN_90_MIN;

   /* +++++++++++++++++++++++++ */

   _asm mov  eax,pit       /* get pit count out of the way. Don't need ... */
   _asm xor  edx,edx       /* ... higher res than e-7 (think 'clock jitter')*/
   _asm mov  ecx,05C105C6h /* 0.000,000,838,[096,536,006],713,116,004,678,838*/
   _asm mul  ecx           /* 0x10000 => 6,326,583,689,216 (5C1:05C6000) */
   _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
   _asm div  ecx           /* yes, truncate, don't round */
   _asm push eax           /* 0x10000 => 6326 */
   _asm mov  eax,pit
   _asm xor  edx,edx
   _asm mov  ecx,346h      /* 0.[000,000,838],096,536,006,713,116,004,678,838*/
   _asm mul  ecx           /* 0x1000 => 54,919,168 (0:3460000) */
   _asm pop  ecx
   _asm add  eax,ecx       /* 54,919,168 + 6,326 => 54,925,494 */
   _asm adc  edx,0
   _asm add  nsecs,eax     /* 0x1000 => 0.054,925,494 secs */

   /* +++++++++++++++++++++++++ */

   _asm mov  eax,rticks    /* now do long multiplication with (ticks%0x1800B)*/
   _asm xor  edx,edx
   _asm mov  ecx,56b71h    /* 0.054,925,494,583,735,950,770,482,632,[355,185]*/
   _asm mul  ecx           /* 0x1800B => 34,920,013,275 (8:21651DDB) */
   _asm mov  ecx,0F4240h   /* 1,000,000 */
   _asm div  ecx           /* 0x1800B => 34920.013275 (33DB:8868) */
                           /* 0x1800B0 => 558720.2124 */
   _asm push eax
   _asm mov  eax,rticks
   _asm xor  edx,edx
   _asm mov  ecx,2deca1c8h /* 0.054,925,494,583,735,950,[770,482,632],355,185*/
   _asm mul  ecx           /* 0x1800B => 75,749,999,965,080 (44E4:EBD6F398) */
   _asm pop  ecx
   _asm add  eax,ecx       /* 75,749,999,965,080 + 34920 => 75750000000000 */
   _asm adc  edx,0
   _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
   _asm div  ecx           /* 0x1800B => 75750.000000000 (0:127E6) */
                           /* 0x1800B0 => 1212000.000000000 */
   _asm push eax
   _asm mov  eax,rticks
   _asm xor  edx,edx
   _asm mov  ecx,22CB1A8Eh /* 0.054,925,494,[583,735,950],770,482,632,355,185*/
   _asm mul  ecx           /* 0x1800B => 57,389,999,924,250 (3432:268F241A) */
   _asm pop  ecx
   _asm add  eax,ecx       /* 57,389,999,924,250 + 75750 => 57390000000000 */
   _asm adc  edx,0
   _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
   _asm div  ecx           /* 0x1800B => 57390.000000000 (0:E02E) */
                           /* 0x1800B0 => 918240.000000000 */
   _asm push eax
   _asm mov  eax,rticks    
   _asm xor  edx,edx
   _asm mov  ecx,34618b6h  /* 0.[054,925,494],583,735,950,770,482,632,355,185*/
   _asm mul  ecx           /* 0x1800B => 5,399,999,942,610 (4E9:49140FD2) */
   _asm pop  ecx
   _asm add  eax,ecx       /* 5,399,999,942,610 + 57390 => 5400000000000 */
   _asm adc  edx,0
   _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
   _asm div  ecx           /* 0x1800B => 5400.000000000 (0:1518) */
                           /* 0x1800B0 => 86399.[99908176+918240=100826416] */
   _asm add  secs,eax
   _asm add  nsecs,edx

   /* +++++++++++++++++++++++++ */

   /* adjust secs with nanosec overflow ala 'if (nsecs>=x){secs++;nsecs-=x};'*/
   _asm mov  ecx,3B9ACA00h /* 1,000,000,000 */
   _asm cmp  nsecs,ecx
   _asm cmc
   _asm sbb  eax,eax
   _asm mov  edx,eax
   _asm and  eax,1
   _asm and  edx,ecx
   _asm add  secs,eax
   _asm sub  nsecs,edx

   /* +++++++++++++++++++++++++ */

   tsp->tv_sec = secs;
   tsp->tv_nsec = nsecs;

   return;
}      

/* ---------------------------------------------------------------------- */

unsigned long __get_raw_tick_count(void);
unsigned long __get_raw_pit_count(void);
unsigned long __get_raw_pit_rate(void);
void __initialize_pit(void);

#pragma aux __get_raw_tick_count2 =         \
     "xor eax,eax"                          \
     "int 1ah"                              \
     "mov ax,cx"                            \
     "shl eax,10h"                          \
     "mov ax,dx"                            \
     __parm __nomemory [] __value [__eax]   \
     __modify __exact __nomemory [__eax __dx __cx];        
#pragma aux __get_raw_tick_count =          \
     "push es"                              \
     "mov  ax,40h"                          \
     "mov  es,ax"                           \
     "mov  eax,6Ch"                         \
     "mov  eax,es:[eax]"                    \
     "pop  es"                              \
     __parm __nomemory [] __value [__eax]   \
     __modify __exact __nomemory [__eax];
#pragma aux __get_raw_pit_count =           \
     "xor eax,eax"                          \
     "out 043h,al" /* outpb(0x43, 0x00); */ \
     "in al,040h"                           \
     "mov ah,al"    /* lsb = inpb(0x40); */ \
     "in al,040h"   /* msb = inpb(0x40); */ \
     "xchg al,ah"   /* pit=(msb<<8)+lsb; */ \
     __parm __nomemory [] __value [__eax] __modify __exact __nomemory [__eax];
#pragma aux __initialize_pit =              \
           /*bits6|7='cntr=0',4|5=rw mode*/ \
           /*  3|2|1=ctr type, 0=BCD/bin */ \
     "mov al,34h"  /* '3'=rd lsb then msb*/ \
     "out 043h,al" /* '4'=use rategen    */ \ 
     "xor al,al"   /* 0x10000 hi/lo      */ \
     "out 040h,al" /* (defrate = 65536)  */ \
     "out 040h,al" /* set rate=0x10000 */   \
     __parm __nomemory [] __modify __exact __nomemory [__eax];
#pragma aux __get_raw_pit_rate =            \
     "mov eax,10000h" /* hmm, how do we do this? */ \
     __parm __nomemory [] __value [__eax] __modify __exact __nomemory [__eax];

static unsigned long __get_tick_and_pit_count(unsigned long *pitcount)
{
  static int need_init = 1;
  unsigned long ticks = 0, pit = 0;

  if (need_init && pitcount)
  {
     __initialize_pit();
     //ticks = __get_raw_tick_count();
     //while (ticks == __get_raw_tick_count())
     //  ;
     need_init = 0;
  }
  do {
    ticks = __get_raw_tick_count();
    if (!pitcount)
      break;
    pit = (__get_raw_pit_rate() - __get_raw_pit_count()); /* its a countdown */
  } while (ticks != __get_raw_tick_count());

  if (pitcount)
    *pitcount = (pit & 0xffff);
  return ticks;
}
                                               
/* ---------------------------------------------------------------------- */

#ifdef DEBUG_MONOTONIC /* Icky stuff, for testing only */
static void __convert_ticks_and_pit_using_double( unsigned long days,
                                                  unsigned long ticks,
                                                  unsigned int pit,
                                                  struct timespec *tsp )
{
  double dbltime = (((double)ticks)*0.054925494583735950770482632355185) +
                   (((double)pit)*0.000000838096536006);
  tsp->tv_nsec = (long)(fmod(dbltime,1.0)*1000000000.0);
  tsp->tv_sec = (time_t)(dbltime + (86400ul * days));
  return;
}
#endif

/* ---------------------------------------------------------------------- */

int getnanotime(struct timespec *tsp)
{
  if (tsp)
  {
    static unsigned long lastticks = 0, lastclks = 0;
    static unsigned int day_wrap = 0;
    unsigned long ticks, clks;
    struct timespec ts;

    ticks = __get_tick_and_pit_count(&clks);

    if (ticks < lastticks)
      day_wrap++;
    else if (ticks == lastticks && clks < lastclks)
      clks = lastclks;
    lastclks  = clks;
    lastticks = ticks;

    __convert_ticks_and_pit_to_timespec( day_wrap, ticks, clks, &ts );

    #ifdef DEBUG_MONOTONIC
    {
      static time_t dosdelta = 0; /* the time_t at 0:00 today */
      static struct timespec lasttest = {0,0};
      time_t dossecs; struct timespec dbltest;

      dossecs = time(0);
      if (dosdelta == 0)
        dosdelta = dossecs-ts.tv_sec;
      dossecs -= dosdelta;

      __convert_ticks_and_pit_using_double(day_wrap, ticks, clks, &dbltest );
      if (dbltest.tv_sec != ts.tv_sec || ts.tv_nsec < (dbltest.tv_nsec-1) || 
          ts.tv_nsec > (dbltest.tv_nsec+1) || /* allow 1ns rounding bounds */
          ts.tv_sec < (dossecs-1) || /* dos time speeds up/slows down every */
          ts.tv_sec > (dossecs+1) || /* 5 secs as it does .2 adjustment */
          ts.tv_sec < lasttest.tv_sec ||
          (ts.tv_sec == lasttest.tv_sec && ts.tv_nsec < lasttest.tv_nsec))
      {
        printf("\r:%06X dos=%-5u dbl=%u.%6.06u (%u.%6.06u) last=%u.%6.06u\n",
                clks, dossecs, 
                dbltest.tv_sec,  dbltest.tv_nsec, 
                ts.tv_sec, ts.tv_nsec,
                lasttest.tv_sec, lasttest.tv_nsec );
        TRACE_OUT((0,"\r:%06X dos=%-5u dbl=%u.%6.06u (%u.%6.06u) last=%u.%6.06u\n",
                     clks, dossecs, dbltest.tv_sec,  dbltest.tv_nsec,
                     ts.tv_sec, ts.tv_nsec, lasttest.tv_sec, lasttest.tv_nsec ));
      }    
      lasttest.tv_sec = ts.tv_sec;
      lasttest.tv_nsec = ts.tv_nsec;
    }
    #endif /* DEBUG_MONOTONIC */

    tsp->tv_sec = ts.tv_sec;
    tsp->tv_nsec = ts.tv_nsec;
  }
  return 0;
}  

/* ---------------------------------------------------------------------- */

int getmicrotime(struct timeval *tv)
{
  if (tv)
  {
    struct timespec ts;
    getnanotime(&ts);
    tv->tv_sec = ts.tv_sec;
    tv->tv_usec = ts.tv_nsec / 1000;
  }
  return 0;
}

/* ---------------------------------------------------------------------- */

int clock_gettime( clockid_t clk_type, struct timespec *tspec )
{
  static struct timespec virt_base = {0,0};
  struct timespec ts;

  #ifdef EINVAL /* if we have errno.h */
  int new_errno = EINVAL;
  #endif

  if (virt_base.tv_nsec == 0)
  {
    getnanotime(&ts);
    if (ts.tv_nsec == 0)
      ts.tv_nsec++;
    virt_base.tv_sec = ts.tv_sec;
    virt_base.tv_nsec = ts.tv_nsec;
  }
  if (!tspec)
  {
    #ifdef EFAULT
    new_errno = EFAULT;
    #endif
  }
  else if (clk_type == CLOCK_REALTIME ||
           clk_type == CLOCK_VIRTUAL  ||
           clk_type == CLOCK_PROF     ||
           clk_type == CLOCK_MONOTONIC)
  {
    for (;;) /* goto without goto :) */
    {
      getnanotime(&ts);
      if (clk_type == CLOCK_VIRTUAL || clk_type == CLOCK_PROF)
      {
        if (ts.tv_nsec < virt_base.tv_nsec)
        {
          ts.tv_nsec += 1000000000ul;
          ts.tv_sec --;
        }
        ts.tv_nsec -= virt_base.tv_nsec;
        ts.tv_sec -= virt_base.tv_sec;
      }
      else if (clk_type == CLOCK_REALTIME)
      {
        static struct timespec real_delta = {0,0};
        struct timeb tmb; unsigned long now, exp;
        long abs_secs_west;

        /* abs_secs_west is equivalent of utctime-minus-localtime, ie the */
        /* number of secs to subtract from time() to get localtime, or */
        /* inversely, the number of secs to add to localtime to get utctime */

        //ftime(&tmb);
        tmb.time = time(0);
        __get_timezone_info( 0, 0, &abs_secs_west );
        /* don't trust ftime tz fields (broken on watcom) */

        now = tmb.time;                 /* time from dos as UTC */
        now -= abs_secs_west;           /* time from dos as local */        

        exp = ts.tv_sec + real_delta.tv_sec;
        if ((exp > (now+1)) || (exp < (now-1)))
        {                                /* time or timezone has changed */
          now /= 86400ul;                /* dos days */ 
          exp = (((unsigned long)ts.tv_sec) / 86400ul); /* uptime days */
          if (now < exp)         /* dos days is less than uptime days? */
          {                      
            #ifdef ERANGE
            new_errno = ERANGE;
            #endif
            break;
          }
          #ifdef DEBUG_MONOTONIC
          {
            printf("\rtime/timezone sync:\n"
                   "ticker.tv_sec = %ld, local.time = %ld\n"
                   "old delta=%lu, new=%lu\n", 
                   ts.tv_sec, (tmb.time-abs_secs_west), 
                   real_delta.tv_sec, ((now - exp) * 86400ul) );
            TRACE_OUT((0,"\rtime/timezone sync:\n"
                   "ticker.tv_sec = %ld, local.time = %ld\n"
                   "old delta=%lu, new=%lu\n",
                   ts.tv_sec, (tmb.time-abs_secs_west),
                   real_delta.tv_sec, ((now - exp) * 86400ul) ));        
          }
          #endif
          real_delta.tv_sec = ((now - exp) * 86400ul);
          real_delta.tv_nsec = 0;
        }
        ts.tv_sec += real_delta.tv_sec;
        ts.tv_sec += abs_secs_west; /* make it utc */
        ts.tv_nsec += real_delta.tv_nsec;
        if (((unsigned long)ts.tv_nsec) >= 1000000000ul)
        {
          ts.tv_sec ++;
          ts.tv_nsec -= 1000000000ul;
        }

        #ifdef DEBUG_TIMEOFDAY
        if (ts.tv_sec > (tmb.time+1) || (ts.tv_sec < (tmb.time-1)))
        {
          printf("\rts.tv_sec = %ld != tmb.time = %ld\n", ts.tv_sec, tmb.time);
          #ifdef TRACE
          TRACE_OUT((0,"\rts.tv_sec = %ld != tmb.time = %ld\n", ts.tv_sec, tmb.time));
          #endif
        }
        #endif
      }
      tspec->tv_sec = ts.tv_sec;
      tspec->tv_nsec = ts.tv_nsec;
      return 0;
    } /* for (;;) */
  }
  #ifdef EINVAL
  errno = new_errno;
  #endif
  return -1;
}

/* --------------------------------------------------------------------- */

int gettimeofday( struct timeval *tvp, struct __timezone *tzp )
{
  if (tvp || tzp)
  {
    if (tvp)
    {
      #if 0
      struct timeb tmb;
      ftime(&tmb);
      tvp->tv_sec = tmb.time;
      tvp->tv_usec = tmb.millitm * 1000;      
      #else
      struct timespec ts;
      if (clock_gettime(CLOCK_REALTIME, &ts) != 0)
        return -1;
      tvp->tv_sec = ts.tv_sec;
      tvp->tv_usec = ts.tv_nsec / 1000;
      #endif
    }  
    if (tzp)
    {
      int minuteswest, dsttime;
      __get_timezone_info( &minuteswest, &dsttime, 0 );
      tzp->tz_minuteswest = (((short)minuteswest) & 0xffff);
      tzp->tz_dsttime = (short)dsttime;
    }
    return 0; 
  }
  #ifdef EINVAL
  errno = EINVAL;
  #endif 
  return -1;
}

/* -------------------------------------------------------------------- */

#if 0
int crude_test( struct timeval *tvp )
{
  if (tvp)
  {
    #if 1
    struct timeb tb;
    ftime(&tb);
    tvp->tv_sec = tb.time;
    tvp->tv_usec = tb.millitm*1000;
    #else
    time_t t;
    unsigned long ticks,uticks;
    unsigned long secs,usecs;
    _asm 
    {
      push es
      mov  ax,40h
      mov  es,ax
      reread:
      mov  ecx,dword ptr es:[6Ch]
      xor  eax,eax
      out  043h,al
      in   al,040h
      mov  ah,al
      in   al,040h
      mov  edx,dword ptr es:[6Ch]
      cmp  edx,ecx
      jnz  reread
      pop  es
      xchg ah,al
      not  ax           /* timer counts down from 65535 so we flip it */
      mov  ticks,ecx
      mov  uticks,eax
    }
    t = time(NULL);

    ticks%= 65543ul;    /* make sure we only have ticks in the last hour */
                        /* max divisor is 78196 */
    ticks*= 54925;      /* 54935 == micro sec per BIOS clock tick. */

    secs   = t-(t % 3600ul); /* time - secs in last hour */
    usecs  = ( ticks + (( (uticks * 8381ul) + (ticks % 1000) ) / 1000) );
    secs  += usecs/1000000ul;
    usecs %= 1000000ul;

    tvp->tv_sec = secs;
    tvp->tv_usec = usecs;

    /*    
    printf("time() %u calctime() %u.%u %d %.3f\n", t, secs, usecs,
                  t-secs, (((double)t)-((double)secs))/54.925 );
    printf("%s", ctime(&t));
    printf("  %s ", ctime(&secs));
    printf("minwest: %d\n", (timezone/60) );
    */
    #endif
  }
}  
#endif
