// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for formatting keyrate/time/summary data
// statistics obtained from clirate.cpp into strings suitable for display.
//
// $Log: clisrate.cpp,v $
// Revision 1.14  1998/06/14 08:12:41  friedbait
// 'Log' keywords added to maintain automatic change history
//
//


/* module history:
   01 May 1998 - created - Cyrus Patel <cyp@fb14.uni-mainz.de>

   ?? May 1998 - ?? - changed phrase "kps" to "keys/s"

   30 May 1998 - Cyrus Patel <cyp@fb14.uni-mainz.de>
                 changed phrase "keys/s" *BACK* to "k/s" in
                 CliGetSummaryStringForContest. That line has
                 exactly 3 characters to spare, and >9 blocks (or
                 >9 days, or >999Kk/s) will cause line to wrap.
*/

#include "clisrate.h" //includes client.h, clitime.h, clirate.h, clicdata.h

// ---------------------------------------------------------------------------

// returns keyrate as string (len<=26) "nnnn.nn ['K'|'M'|'G'|'T']"
// return value is a pointer to buffer.
char *CliGetKeyrateAsString( char *buffer, double rate )
{
  if (rate<=((double)(0)))  // unfinished (-2) or error (-1) or impossible (0)
    strcpy( buffer, "---.-- " );
  else
  {
    unsigned int t1, t2 = 0;
    const char *t3[]={"","k","M","G","T"}; // "", "kilo", "mega", "giga", "tera"
    while (t2<=5 && (((double)(rate))>((double)(_U32LimitDouble_))) )
    {
      t2++;
      rate=((double)(rate))/((double)(1000));
    }
    if (t2>4)
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

// ---------------------------------------------------------------------------

// "4 RC5 blocks 12:34:56.78 - [123456789 k/s]" !!!DONT MAKE IT keys/s!!!
const char *CliGetSummaryStringForContest( int contestid )
{
  static char str[70];
  char keyrate[32];
  const char *keyrateP, *name;
  unsigned blocks;
  struct timeval ttime;

  if ( CliIsContestIDValid( contestid ) ) //clicdata.cpp
  {
    CliGetContestInfoBaseData( contestid, &name, NULL ); //clicdata.cpp
    keyrateP=CliGetKeyrateAsString(keyrate,CliGetKeyrateForContest(contestid));
    CliGetContestInfoSummaryData( contestid, &blocks, NULL, &ttime ); //ditto
  }
  else
  {
    name = "???";
    blocks = ttime.tv_sec = ttime.tv_usec = 0;
    keyrateP = "---.-- ";
  }

  sprintf(str, "%d %s block%s %s%c- [%sk/s]", //!!!DON'T MAKE IT longer!
       blocks, name, ((blocks==1)?(""):("s")),
       CliGetTimeString( &ttime, 2 ), ((!blocks)?(0):(' ')), keyrateP );
  return str;
}

// ---------------------------------------------------------------------------

// return iter/keysdone/whatever as string. set inNetOrder if 'u'
// needs ntohl()ing first, set contestID = -1 to have the ID ignored
const char *CliGetU64AsString( u64 *u, int inNetOrder, int contestid )
{
  static char str[32];
  unsigned int i;
  u64 norm;
  double d;

  norm.hi = u->hi;
  norm.lo = u->lo;

  if (inNetOrder)
  {
    norm.hi = ntohl( norm.hi );
    norm.lo = ntohl( norm.lo );
  }

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

// Queued 1 RC5 1*2^30 block 68E0D85A:A0000000 (10.25% done)
const char *CliGetMessageForFileentryLoaded( FileEntry *fileentry )
{
  static char str[84];
  const char *name;
  unsigned int size=1, count=32, iter = ntohl(fileentry->iterations.lo);
  unsigned int startpercent = (unsigned int) ( (double) 10000.0 *
           ( (double) (ntohl(fileentry->keysdone.lo)) / (double)(iter) ) );

  if (iter)
  {
    count = 1;
    size = 0;
    while ((iter & count)==0)
      { size++; count <<= 1; }
    count = iter / (1<<size);
  }
  if (CliGetContestInfoBaseData( fileentry->contest, &name, NULL )!=0) //clicdata
    name = "???";

  sprintf( str, "%s %s %d*2^%d block %08lX:%08lX%c(%u.%02u%% done)",
  #ifdef MULTITHREAD
           "Loaded",
  #else
           "Started",
  #endif
           name, (int) count, (int)size,
           (unsigned long) ntohl( fileentry->key.hi ),
           (unsigned long) ntohl( fileentry->key.lo ),
           ((startpercent)?(' '):(0)),
           (unsigned)(startpercent/100), (unsigned)(startpercent%100) );
  return str;
}

// ---------------------------------------------------------------------------

// Completed RC5 block 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
const char *CliGetMessageForProblemCompleted( Problem *prob )
{
  static char str[160];
  RC5Result rc5result;
  struct timeval tv;
  char keyrate[32];
  const char *keyrateP, *name;
  int contestid = prob->GetResult( &rc5result );

  if (CliGetContestInfoBaseData( contestid, &name, NULL )==0) //clicdata
    keyrateP = CliGetKeyrateAsString(keyrate, CliGetKeyrateForProblem( prob ));
  else
  {
    keyrateP = "---.-- ";
    name = "???";
  }

  tv.tv_sec = prob->timehi;
  tv.tv_usec = prob->timelo;
  CliTimerDiff( &tv, &tv, NULL );

  sprintf( str, "Completed %s block %08lX:%08lX (%s keys)\n"
                " %s  %s - [%skeys/sec]\n",
                name,
                (unsigned long) ntohl( rc5result.key.hi ) ,
                (unsigned long) ntohl( rc5result.key.lo ),
                CliGetU64AsString( &(rc5result.iterations), 1, contestid ),
                CliGetTimeString( NULL, 0 ),
                CliGetTimeString( &tv, 2 ),
                keyrateP );
  return str;
}

// ---------------------------------------------------------------------------

const char *CliReformatMessage( const char *header, const char *message )
{
  static char strspace[160];
  unsigned int prelen, linelen, doquote = (header!=NULL);
  char buffer[84], *bptr, *sptr;
  const char *mptr = message;

  strspace[0]=0;
  if (mptr && *mptr)
  {
    while (*mptr == ' ' || *mptr == '\n')
      mptr++;

    sprintf( strspace, "[%s] ", CliGetTimeString(NULL,1) );
    prelen = strlen( strspace );
    if (header && *header) strcat( strspace, header );
    if (doquote) strcat( strspace, "\"" );

    //first line

    linelen = strlen( strspace );
    bptr = buffer;

    while (linelen < 78)
    {
      if (!*mptr)
        break;
      if (*mptr=='\n')
      {
        mptr++;
        break;
      }
      *bptr++ = *mptr++;
      linelen++;
    }
    if (linelen >= 78)
    {
      *bptr = 0;
      if ((sptr = strrchr( buffer, ' '))!=NULL)
      {
        mptr -= (bptr-sptr);
        bptr = sptr;
      }
    }
    while (*mptr==' ' || *mptr=='\n')
      mptr++;
    if (!*mptr && doquote)
      *bptr++ = '\"';
    *bptr++='\n';
    *bptr=0;

    strcat(strspace, buffer );

    //second line

    if (*mptr)
    {
      bptr = strspace+strlen( strspace );
      for (linelen=0;linelen<prelen;linelen++)
        *bptr++ = ' ';
      *bptr = 0;

      linelen = prelen;
      bptr = buffer;

      while (linelen < 78)
      {
        if (!*mptr)
          break;
        if (*mptr=='\n')
        {
          mptr++;
          break;
        }
        *bptr++ = *mptr++;
        linelen++;
      }
      if (linelen >= 78)
      {
        *bptr = 0;
        if ((sptr = strrchr( buffer, ' '))!=NULL)
        {
          mptr -= (bptr-sptr);
          bptr = sptr;
        }
      }
      if (!*mptr && doquote)
        *bptr++ = '\"';
      *bptr++='\n';
      *bptr=0;

      strcat(strspace, buffer );
    }

    //end of second line

    bptr = strspace;                 //convert non-breaking space
    while (*bptr)
    {
      if (*bptr == (char) 0xFF)
        *bptr = ' ';
      bptr++;
    }
  }
  return (const char *)(strspace);
}

// ---------------------------------------------------------------------------

