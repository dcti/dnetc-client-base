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
return "@(#)$Id: clisrate.cpp,v 1.45.2.16 2000/09/24 13:36:31 andreasb Exp $"; }

//#define TRACE

#include "cputypes.h"  // u32
#include "problem.h"   // Problem class
#include "client.h"    // Fileentry struct
#include "baseincs.h"  // timeval, sprintf et al
#include "util.h"      // trace
#include "logstuff.h"  // Log()
#include "clitime.h"   // CliTimer(), CliTimerDiff(), CliGetTimeString()
#include "clirate.h"   // CliGetKeyrateFor[Problem|Contest]()
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

static void __unbias_iter_count(unsigned int contestid, u32 *hiP, u32 *loP)
{
  /* for DES (and maybe others) where the number of keys we show is a 
     multiple of the number of iterations done. This has nothing to do
     with pipeline count etc, but is used when each iteration implies
     the check of another 'key', in DES' case, of the complement.
  */     
  unsigned int multiplier = 0;
  if (CliGetContestInfoBaseData( contestid, NULL, &multiplier )==0) //clicdata
  {
    if (multiplier == 2) /* DES for example */
    {
      *hiP <<= 1;
      *hiP |= (*loP >> 31);
      *loP <<= 1;
    }
    else if (multiplier > 1) /* and <= 0xffff is implied */
    {                   /* (the following will need to be changed if not) */
      u32 hi = *hiP, lo = *loP;
      *loP = (lo * multiplier);
      *hiP = (hi * multiplier) + 
          ((lo >> 16) * multiplier) + (((lo & 0xffff) * multiplier) >> 16);
    }
  }      
  return;
}  
      
/* ----------------------------------------------------------------------- */

static const char *__iter_as_string( char *buf, u32 iterhi, u32 iterlo, 
                                     int contestid )
{
  __unbias_iter_count(contestid, &iterhi, &iterlo);

#if 1 /* we always show iter as an n*2^28 count nowadays */  
  {
    unsigned int units = (((iterlo) >> 28) +
                          ((iterhi) <<  4));
    unsigned int twoxx = 28;
    if (!units) /* less than 2^28 packet (eg test) */
    {
      units = iterlo >> 10;
      twoxx = 10;
    }  
    sprintf(buf,"%u*2^%u ", units, twoxx );
  }
#else /* unused - we print n*2^28 count nowadays */
  #if (ULONG_MAX > 0xffffffff) /* 64 bit math */
  sprintf(buf,"%lu", ((unsigned long)iterhi)<<32)+iterlo );
  #else /* always do it the hard way, even if we have long long */
  {
    char str[sizeof("18,446,744,073,709,551,615  ")];
    u32 carry = 0;
    if (iterhi)
    {
      double d = (((double)iterhi)*4294967296.0)+iterlo;
      iterhi = (u32)(d / 1000000000.0);
      iterlo = (u32)(d - (((double)(iterhi))*1000000000.0));
      d = d / 1000000000.0;
      if (d > 0)
      {
        carry = (unsigned int)(d / 1000000000.0);
        iterhi = (unsigned int)(d - (((double)(i))*1000000000.0));
      }
    }  
    if (carry)        sprintf( buf, "%u%09u%09u ", (unsigned) i, (unsigned) iterhi, (unsigned) iterlo );
    else if (iterhi)  sprintf( buf, "%u%09u ", (unsigned) iterhi, (unsigned) iterlo );
    else              sprintf( buf, "%u ", (unsigned) iterlo );
  }
  #endif
#endif
  return buf;
}

/* ----------------------------------------------------------------------- */

#ifdef HAVE_OGR_CORES
static const char *__nodecount_as_string( char *buf, u32 nodeshi, u32 nodeslo )
{
  /* stats unit is Gnodes, so if nodes <= 9,999,999 (<0.01 Gnodes), */
  /* then "n,nnn,nnn\0", else "iii.ff X\0" */
  if (nodeshi == 0 && nodeslo <= 9999999)
  {
    char nodecount[32]; 
    int pos = sizeof(nodecount);
    unsigned int numdigits=0;
    nodecount[--pos] = '\0';
    nodecount[--pos] = ' ';
    do
    {
      if (numdigits && (numdigits%3)==0)
        nodecount[--pos]=',';
      nodecount[--pos]=(char)((nodeslo % 10)+'0');
      nodeslo/=10;
    } while (nodeslo);
    strcpy( buf, &nodecount[pos] );
  }  
  else 
  {
    __double_as_string( buf, ((((double)nodeshi)*4294967296.0)+
                             ((double)nodeslo)), 3, 2, !0 );
  }   
  return buf;
}  
#endif

/* ----------------------------------------------------------------------- */

// "4 RC5 packets 12:34:56.78 - [234.56 Kkeys/s]" 
int CliPostSummaryStringForContest( int contestid )
{
  char str[160];
  char ratestrbuf[32];
  double totaliter;
  const char *ratestrP, *name;
  unsigned int packets, units;
  struct timeval ttime;

  name = "???";
  units = packets = 0;
  ttime.tv_sec = 0;
  ttime.tv_usec = 0;
  ratestrP = "---.-- ";
  str[0] = '\0';

  if ( CliIsContestIDValid( contestid ) ) //clicdata.cpp
  {
    CliGetContestInfoBaseData( contestid, &name, NULL ); //clicdata.cpp
    CliGetContestInfoSummaryData( contestid, &packets, &totaliter, &ttime, &units ); //ditto
    ratestrP = __double_as_string(ratestrbuf,
          CliGetKeyrateForContest(contestid), 3, 2, !0);
  }
  switch (contestid)
  {
    case RC5:
    case DES:
    case CSC:
    {
      sprintf(str, "%d %s packet%s (%u*2^28 keys)\n"
                   "%s%c- [%skeys/s]", 
           packets, name, ((packets==1)?(""):("s")), units,
           CliGetTimeString( &ttime, 2 ), ((!packets)?(0):(' ')), ratestrP );
      break;
    }
    case OGR:
    {
      const char *nodestrP = ratestrP;
      char nodestrbuf[32];
      if (*nodestrP != '-') /* number is valid */
        nodestrP = __double_as_string( nodestrbuf, totaliter, 3, 2, !0);
      sprintf(str, "%d %s packet%s (%snodes)\n"
                   "%s%c- [%snodes/s]", 
           packets, name, ((packets==1)?(""):("s")), nodestrP,
           CliGetTimeString( &ttime, 2 ), ((!packets)?(0):(' ')), ratestrP );
      //printf("DEBUG: %.0f\n", totaliter);
      break;
    }
    default:
    {
      str[0] = '\0';
      break;
    }
  }
  if (str[0])
  {
    Log("Summary: %s\n", str );
    return 0;
  }
  return -1;
}

/* ----------------------------------------------------------------------- */

// adjust cumulative stats with or without "Completed" message
// Completed RC5 packet 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
// Completed OGR stub 22/1-3-5-7 (123456789 nodes)
//          123:45:67:89 - [987654321 nodes/s]
int CliRecordProblemCompleted( Problem *prob, int do_postmsg )
{
  int rc = -1;
  ContestWork work;
  unsigned int contestid = 0;
  int resultcode = prob->RetrieveState( &work, &contestid, 0 );

  if (resultcode == RESULT_NOTHING || resultcode == RESULT_FOUND)
  {
    const char *name;
    if (CliGetContestInfoBaseData(contestid,&name,0)==0) /*contestid is valid*/
    { 
      double rate; 
      //if (!dosave)
      //  rate = CliGetKeyrateForProblemNoSave( prob );
      //else
      rate = CliGetKeyrateForProblem( prob ); //add to totals
      rc = 0; /* success */
  
      if (do_postmsg)
      {
        struct timeval tv;
        char ratestrbuf[64];
        char countstrbuf[64]; 

        //tv.tv_sec = prob->runtime_sec;  //thread user time
        //tv.tv_usec = prob->runtime_usec;
        //tv.tv_sec  = prob->elapsed_time_sec;  //wall clock time
        //tv.tv_usec = prob->elapsed_time_usec;
        prob->GetElapsedTime(&tv);              // wall clock time
 
        switch (contestid) 
        {
          case RC5:
          case DES:
          case CSC:
          {
            //"Completed RC5 packet 00000000:00000000 (4*2^28 keys)\n"
            //"%s - [%skeys/sec]\n"
            __double_as_string( ratestrbuf, rate, -1, 2, !0 );
            __iter_as_string( countstrbuf, work.crypto.iterations.hi,
                                     work.crypto.iterations.lo, contestid );
            Log( "Completed %s packet %08lX:%08lX (%skeys)\n"
                 "%s - [%skeys/sec]\n",  
                        name, 
                        (unsigned long) ( work.crypto.key.hi ),
                        (unsigned long) ( work.crypto.key.lo ),
                        countstrbuf,
                        CliGetTimeString( &tv, 2 ),
                        ratestrbuf );
            break;
          }  
          #ifdef HAVE_OGR_CORES
          case OGR:
          {
            //Completed OGR stub 23/21-16-11-31 (17.63 Gnodes)
            //"%s - [%snodes/sec]\n"
            __double_as_string( ratestrbuf, rate, -1, 2, !0 );
            __nodecount_as_string( countstrbuf, work.ogr.nodes.hi, 
                                                work.ogr.nodes.lo );
            LogTo(LOGTO_SCREEN,
                     "Completed %s stub %s (%snodes)\n"
                     "%s - [%snodes/sec]\n",  
                     name, 
                     ogr_stubstr( &work.ogr.workstub.stub ),
                     countstrbuf,
                     CliGetTimeString( &tv, 2 ),
                     ratestrbuf );
            //Completed OGR stub 23/21-16-11-31 (17,633,305,532 nodes)
            //"%s - [%snodes/sec]\n"
            __double_as_string( countstrbuf, 
                              ((((double)work.ogr.nodes.hi)*4294967296.0)+
                              ((double)work.ogr.nodes.lo)), 3, 2, !0 );
            LogTo(LOGTO_FILE|LOGTO_MAIL,
                     "Completed %s stub %s (%snodes)\n"
                     "%s - [%snodes/sec]\n",  
                     name, 
                     ogr_stubstr( &work.ogr.workstub.stub ),
                     countstrbuf,
                     CliGetTimeString( &tv, 2 ),
                     ratestrbuf );             
            break;
          }  
          #endif /* OGR */
          default:
          {
            //rate = 0.0;                    
            break;
          }
        } /* switch (contestid) */
      } /* if (postmsg) */
    } /* name/contestid is valid */  
  } /* (RESULT_NOTHING || RESULT_FOUND) */
  return rc;
}

/* ----------------------------------------------------------------------- */

// returns rate as string "n,nnn.nn ['k'|'M'|'G'|'T']"
// return value is a pointer to buffer.
const char *CliGetKeyrateAsString( char *buffer, double rate )
{ return __double_as_string( buffer, rate, -1, 2, !0 ); }

/* ----------------------------------------------------------------------- */
