/* Copyright distributed.net 1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ----------------------------------------------------------------------
 * This file contains functions for formatting keyrate/time/summary data
 * statistics obtained from clirate.cpp into strings suitable for display.
 * ----------------------------------------------------------------------
*/ 
const char *clisrate_cpp(void) {
return "@(#)$Id: clisrate.cpp,v 1.43 1999/04/16 07:22:35 gregh Exp $"; }

#include "cputypes.h"  // u64
#include "problem.h"   // Problem class
#include "client.h"    // Fileentry struct
#include "baseincs.h"  // timeval, sprintf et al
#include "clitime.h"   // CliTimer(), CliTimerDiff(), CliGetTimeString()
#include "clirate.h"   // CliGetKeyrateFor[Problem|Contest]()
#include "clicdata.h"  // CliGetContestInfo[Base|Summary]Data()
#include "util.h"      // temporary home of ogr_stubstr()
#include "clisrate.h"  // keep prototypes in sync

/*
 *
 * Synopsis:
 *
 *  char *num_sep(char *number)
 *
 * Description:
 *
 *  This routine takes a string which actually represents a number
 *  for instance "1234567" and converts it to "1,234,567". Effectively
 *  it adds a comma for each 3 digits. The function checks whether
 *  a 'fractional part' indicated by a '.' is present and takes care
 *  of this by skipping the '.', the fraction, and whatever follows
 *  the fraction.
 *
 * Known Side Effects:
 *
 *  The function currently does not work correctly if 'number' points
 *  to a string with some other characters prefixed to the number in
 *  question. It also doesn't work, if the 'number' string is a pure
 *  integer without a fraction and is still postfixed with some non-number
 *  data.
 *
 * Return Value:
 *
 *  Good case:
 *     char * to buffer with number separators inserted.
 *
 *  Bad case:
 *     char *number (exactly what it got as input) The bad case happens
 *                  if the input points to a huge string which would
 *                  cause a buffer overflow during processing of the
 *                  input data.
 *
 */

static char *num_sep(char *number)
{

  #define STR_LEN 32
  static char num_string[STR_LEN + 1];

  char *rp, *wp, *xp;             /* read ptr, write ptr, aux ptr */
  register unsigned int digits;
  register unsigned int i, j;

  /*
   * just in case somebody passes a pointer to a long string which
   * then obviously holds much more than just a number. In this case
   * we simply return what we got and don't do anything.
   */

  digits = strlen(number);

  if (digits >= STR_LEN)
    return(number);

  strcpy(num_string, number);
  rp = num_string + digits - 1;
  wp = num_string + STR_LEN;
  *wp-- = '\0';
  if ((xp = strchr(num_string, '.')) != NULL) {
    while (rp >= xp) {
      *wp-- = *rp--;
      digits--;
    }
  }
  for (i = digits, j = 1; i; i--, j++) {
    *wp-- = *rp--;
    if (j && ((j % 3) == 0))
      *wp-- = ',';
  }
  if (wp[1] == ',')
    wp++;
  return(wp + 1);
}


// ---------------------------------------------------------------------------

// returns keyrate as string (len<=26) "nnnn.nn ['K'|'M'|'G'|'T']"
// return value is a pointer to buffer.
static char *__CliGetKeyrateAsString( char *buffer, double rate, double limit )
{
  if (rate<=((double)(0)))  // unfinished (-2) or error (-1) or impossible (0)
    strcpy( buffer, "---.-- " );
  else
  {
    unsigned int t1, t2 = 0;
    const char *t3[]={"","k","M","G","T"}; // "", "kilo", "mega", "giga", "tera"
    while (t2<=5 && (((double)(rate))>=((double)(limit))) )
    {
      t2++;
      rate = ((double)(rate)) / ((double)(1000));
    }
    if (t2 > 4)
      strcpy( buffer, "***.** " ); //overflow (rate>1.0 TKeys. Str>25 chars)
    else
    {
      t1 = (unsigned int)(rate);
      sprintf( buffer, "%u.%02u %s", t1,
         ((unsigned int)((((double)(rate-((double)(t1)))))*((double)(100)))),
         t3[t2] );
    }
  }
  return buffer;
}

// returns keyrate as string (len<=26) "nnnn.nn ['k'|'M'|'G'|'T']"
// return value is a pointer to buffer.
char *CliGetKeyrateAsString( char *buffer, double rate )
{ 
  return (num_sep(__CliGetKeyrateAsString( buffer, rate, _U32LimitDouble_ )));
}

// ---------------------------------------------------------------------------

// "4 RC5 packets 12:34:56.78 - [234.56 Kkeys/s]" 
const char *CliGetSummaryStringForContest( int contestid )
{
  static char str[70];
  char keyrate[32];
  const char *keyrateP, *name;
  unsigned int packets;
  struct timeval ttime;

  if ( CliIsContestIDValid( contestid ) ) //clicdata.cpp
  {
    CliGetContestInfoBaseData( contestid, &name, NULL ); //clicdata.cpp
    CliGetContestInfoSummaryData( contestid, &packets, NULL, &ttime ); //ditto
    keyrateP=__CliGetKeyrateAsString(keyrate,
          CliGetKeyrateForContest(contestid),((double)(1000)));
  }
  else
  {
    name = "???";
    packets = 0;
    ttime.tv_sec = 0;
    ttime.tv_usec = 0;
    keyrateP = "---.-- ";
  }

  sprintf(str, "%d %s packet%s %s%c- [%s%s/s]", 
       packets, name, ((packets==1)?(""):("s")),
       CliGetTimeString( &ttime, 2 ), ((!packets)?(0):(' ')), keyrateP,
       contestid == 2 /* OGR */ ? "nodes" : "keys" );
  return str;
}

// ---------------------------------------------------------------------------

// return iter/keysdone/whatever as string. 
// set contestID = -1 to have the ID ignored
const char *CliGetU64AsString( u64 *u, int /*inNetOrder*/, int contestid )
{
  static char str[32];
  unsigned int i;
  u64 norm;
  double d;

  norm.hi = u->hi;
  norm.lo = u->lo;

  d = U64TODOUBLE(norm.hi, norm.lo);
  if (CliGetContestInfoBaseData( contestid, NULL, &i )==0 && i>1) //clicdata
    d = d * ((double)(i));

  i = 0;
  norm.hi = (unsigned int)(d / 1000000000.0);
  norm.lo = (unsigned int)(d - (((double)(norm.hi))*1000000000.0));
  d = d / 1000000000.0;
  if (d > 0)
  {
    i = (unsigned int)(d / 1000000000.0);
    norm.hi = (unsigned int)(d - (((double)(i))*1000000000.0));
  }

  if (i)            sprintf( str, "%u%09u%09u", (unsigned) i, (unsigned) norm.hi, (unsigned) norm.lo );
  else if (norm.hi) sprintf( str, "%u%09u", (unsigned) norm.hi, (unsigned) norm.lo );
  else              sprintf( str, "%u", (unsigned) norm.lo );

  return str;
}

// ---------------------------------------------------------------------------

// internal - with or without adjusting cumulative stats
// Completed RC5 packet 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
// Completed OGR stub 22/1-3-5-7 (123456789 nodes)
//          123:45:67:89 - [987654321 nodes/s]
static const char *__CliGetMessageForProblemCompleted( Problem *prob, int doSave )
{
  static char str[160];
  ContestWork work;
  struct timeval tv = {0,0};
  char keyrate[32];
  unsigned int /* size=1, count=32, */ itermul;
  unsigned int mulfactor, contestid = 0;
  const char *keyrateP = "---.-- ", *name = "???";
  int resultcode = prob->RetrieveState( &work, &contestid, 0 );

  if (resultcode != RESULT_NOTHING && resultcode != RESULT_FOUND)
    memset((void *)&work,0,sizeof(work));
  else
  {
    if (CliGetContestInfoBaseData( contestid, &name, &mulfactor )==0) //clicdata
    {
      keyrateP = CliGetKeyrateAsString( keyrate, 
          ((doSave) ? ( CliGetKeyrateForProblem( prob ) ) :
                      ( CliGetKeyrateForProblemNoSave( prob ) ))  );
    }
    /*  
    tv.tv_sec = prob->timehi;
    tv.tv_usec = prob->timelo;
    CliTimerDiff( &tv, &tv, NULL );
    */
    tv.tv_sec = prob->runtime_sec;
    tv.tv_usec = prob->runtime_usec;
  }
  
  switch (contestid) 
  {
    case 0: // RC5
    case 1: // DES
    case 3: // CSC
//"Completed one RC5 packet 00000000:00000000 (4*2^28 keys)\n"
//"%s - [%skeys/sec]\n"
      itermul = (((work.crypto.iterations.lo) >> 28) +
                 ((work.crypto.iterations.hi) <<  4) );
      sprintf( str, "Completed one %s packet %08lX:%08lX (%u*2^28 keys)\n"
                    "%s - [%skeys/sec]\n",  
                    name, 
                    (unsigned long) ( work.crypto.key.hi ) ,
                    (unsigned long) ( work.crypto.key.lo ),
                    (unsigned int)(itermul),
                    CliGetTimeString( &tv, 2 ),
                    keyrateP );
      break;
    case 2: // OGR
//"Completed one OGR stub 22/1-3-5-7 (123456789 nodes)\n"
//"%s - [%snodes/sec]\n"
      sprintf( str, "Completed one %s stub %s (%s nodes)\n"
                    "%s - [%snodes/sec]\n",  
                    name, 
                    ogr_stubstr( &work.ogr.stub ),
                    CliGetU64AsString(&work.ogr.nodes, 0, -1),
                    CliGetTimeString( &tv, 2 ),
                    keyrateP );
      break;
  }
  return str;
}

// Completed RC5 packet 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
const char *CliGetMessageForProblemCompleted( Problem *prob )
{ return __CliGetMessageForProblemCompleted( prob, 1 ); }

const char *CliGetMessageForProblemCompletedNoSave( Problem *prob )
{ return __CliGetMessageForProblemCompleted( prob, 0 ); }

// ---------------------------------------------------------------------------

