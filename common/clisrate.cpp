/* Copyright distributed.net 1998-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * ----------------------------------------------------------------------
 * This file contains functions for formatting keyrate/time/summary data
 * statistics obtained from clirate.cpp into strings suitable for display.
 * ----------------------------------------------------------------------
*/ 
const char *clisrate_cpp(void) {
return "@(#)$Id: clisrate.cpp,v 1.45.2.18 2000/10/27 17:58:42 cyp Exp $"; }

//#define TRACE

#include "cputypes.h"  // u32
#include "problem.h"   // Problem class
#include "client.h"    // Fileentry struct
#include "baseincs.h"  // timeval, sprintf et al
#include "util.h"      // trace
#include "logstuff.h"  // Log()
#include "clitime.h"   // CliTimer(), CliTimerDiff(), CliGetTimeString()
#include "clicdata.h"  // CliGetContestInfo[Base|Summary]Data()
#include "clisrate.h"  // keep prototypes in sync

/* ----------------------------------------------------------------------- */

// returns double as string such that the length of the result is never
// greater than 19 (sizeof("9,999,999,999.99 X")) [will be less if 
// max_int_digits is less than 10, fractional portion and comma-separators 
// are optional). " X" will denote the 10**n magnitude if the integer 
// portion was 'squeezed' to honor the max-length or max_int_digits limits.
// The largest value thus representable is 9.NNe+27.
//
static char *__double_as_string( char *buffer, double rate, 
                                 int max_int_digits, /* < 0 means 'any' */
                                 int max_frac_digits, /* < 0 means 'if any' */
                                 int with_num_sep )
{
  TRACE_OUT((+1,"double_as_string(buffer,rate=%f, max_dig(%d.%d),numsep=%d)\n",
              rate, max_int_digits, max_frac_digits, with_num_sep ));

  if (rate<=((double)(0)))  // unfinished (-2) or error (-1) or impossible (0)
    strcpy( buffer, "---.-- " );
  else
  {
    /*kilo, Mega(10**6), Giga(10**9), Tera(10**12), Peta(10**15), Exa(10**18)*/
    const char *magna[]={"","k","M","G","T","P","E"};
    unsigned int idx; unsigned long limit;
    
    if (max_int_digits < 1 || max_int_digits > 10)
      max_int_digits = 10; /* 9,999,999,999 even on 64bit cpus */
    limit = 0;
    while (max_int_digits && limit < ((ULONG_MAX-9)/10))
    {
      limit *= 10;
      limit += 9;
      max_int_digits--;
    }  
    idx = 0;
    while (idx < (sizeof(magna)/sizeof(magna[0])) && rate > ((double)limit))
    {
      idx++;
      rate = ((double)(rate)) / ((double)(1000));
    }
    if (idx >= (sizeof(magna)/sizeof(magna[0])))
      strcpy( buffer, "***.** " ); //overflow
    else
    {
      unsigned long rateint = (unsigned long)rate;
      if (!with_num_sep || rateint <= 999ul)
        sprintf(buffer,"%lu",rateint);
      else  
      {
        char intbuf[60];
        unsigned int numdigits = 0, pos = sizeof(intbuf);
        unsigned long r = rateint;
        intbuf[--pos] = '\0';
        while (r)
        {
          if (numdigits && (numdigits % 3)==0)
            intbuf[--pos] = ',';
          intbuf[--pos] = (char)((r % 10)+'0');
          numdigits++;
          r /= 10;  
        }
        strcpy(buffer, &intbuf[pos] );
      } 
      if (idx != 0 || max_frac_digits > 0)
      { /* misnomer, we always do two digits (if any at all) */
        /* if max_frac_digits < 0 then only if we had to divide to fit */  
        rateint = ((unsigned long)
          ((((double)(rate-((double)(rateint)))))*((double)(100)))) ;
        sprintf( &buffer[strlen(buffer)], ".%02lu", rateint );
      }          
      strcat( strcat( buffer, " " ), magna[idx] );
    }
  }
  TRACE_OUT((-1,"double_as_string() => '%s'\n", buffer ));
  return buffer;
}

/* ----------------------------------------------------------------------- */

// "4 RC5 packets 12:34:56.78 - [234.56 Kkeys/s]" 
int CliPostSummaryStringForContest( int contestid )
{
  double totaliter, rate;
  unsigned int packets, units;
  struct timeval ttime;

  if (CliGetContestInfoSummaryData( contestid, &packets, &totaliter,
                                    &ttime, &units, &rate ) == 0)
  {
    char ratebuf[32], slicebuf[32];
    const char *name = CliGetContestNameFromID(contestid);
    const char *unitname = "nodes";

    __double_as_string( ratebuf, rate, 3, 2, !0);
    if (contestid == OGR)
    {
      __double_as_string( slicebuf, totaliter, 3, 2, !0);
    }
    else
    {
      sprintf( slicebuf, "%u*2^28 ", units );
      unitname = "keys";
    }

    Log("Summary: %u %s packet%s (%s%s)\n%s%c- [%s%s/s]\n",
        packets, name, ((packets==1)?(""):("s")), slicebuf, unitname,
        CliGetTimeString(&ttime,2), ((packets)?(' '):(0)), ratebuf, unitname);
    return 0;
  }
  return -1;
}

/* ----------------------------------------------------------------------- */

// returns rate as string "n,nnn.nn ['k'|'M'|'G'|'T']"
// return value is a pointer to buffer.
const char *CliGetKeyrateAsString( char *buffer, double rate )
{ return __double_as_string( buffer, rate, -1, 2, !0 ); }

/* ----------------------------------------------------------------------- */
