// Copyright distributed.net 1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// This file contains functions for formatting keyrate/time/summary data
// statistics obtained from clirate.cpp into strings suitable for display.
//
// $Log: clisrate.cpp,v $
// Revision 1.32  1998/10/04 11:35:28  remi
// Id tags fun.
//
// Revision 1.31  1998/08/07 19:57:33  cyruspatel
// By popular demand: GetMessageForProblemCompleted() displays the normalized
// keycount (x*2^28) instead of the numeric value.
//
// Revision 1.30  1998/08/07 17:49:15  cyruspatel
// Modified CliGetMessageForFileEntryLoaded() to handle blocksize x*2^28
// normalization (eg 4*2^28 rather than 1*2^30)
//
// Revision 1.29  1998/07/13 18:01:23  friedbait
// fixed some typos in 'num_sep()'
//
// Revision 1.28  1998/07/13 03:29:54  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// ambiguities. ("declaration/type or an expression")
//
// Revision 1.27  1998/07/12 11:20:11  friedbait
// added side effect description to 'num_sep' function.
//
// Revision 1.26  1998/07/11 02:10:15  cyruspatel
// Umm, fixed call to __CliGetKeyrateAsString() - third argument is now a
// double not an unsigned int (caused a compiler warning)
//
// Revision 1.25  1998/07/11 01:53:19  silby
// Change in logging statements - all have full timestamps now so they
// look correct in the win32gui.
//
// Revision 1.24  1998/07/10 20:56:59  cyruspatel
// The summary line creation code now generates the keyrate as a factor of
// 1000, and uses the saved space to write "kkeys/sec" instead of "k/s".
// This also works around a potential line-wrap problem.
//
// Revision 1.23  1998/07/09 21:09:45  friedbait
// added some simple logic to display keyrates like: [657,661.45 k/s]
// which means I have added a number separater ',' to separate groups
// of 3 digits from the right.
//
// Revision 1.22  1998/07/08 08:11:07  cyruspatel
// Added '#include "network.h"' for ntohl()/htonl() prototypes.
//
// Revision 1.21  1998/07/07 21:55:25  cyruspatel
// Serious house cleaning - client.h has been split into client.h (Client
// class, FileEntry struct etc - but nothing that depends on anything) and
// baseincs.h (inclusion of generic, also platform-specific, header files).
// The catchall '#include "client.h"' has been removed where appropriate and
// replaced with correct dependancies. cvs Ids have been encapsulated in
// functions which are later called from cliident.cpp. Corrected other
// compile-time warnings where I caught them. Removed obsolete timer and
// display code previously def'd out with #if NEW_STATS_AND_LOGMSG_STUFF.
// Made MailMessage in the client class a static object (in client.cpp) in
// anticipation of global log functions.
//
// Revision 1.20  1998/07/05 21:49:30  silby
// Modified logging so that manual wrapping is not done on win32gui,
// as it looks terrible in a non-fixed spaced font.
//
// Revision 1.19  1998/06/29 08:44:04  jlawson
// More OS_WIN32S/OS_WIN16 differences and long constants added.
//
// Revision 1.18  1998/06/29 06:57:51  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.17  1998/06/24 21:53:45  cyruspatel
// Created CliGetMessageForProblemCompletedNoSave() in clisrate.cpp. It
// is similar to its non-nosave pendant but doesn't affect cumulative
// statistics.  Modified Client::Benchmark() to use the function.
//
// Revision 1.16  1998/06/15 12:03:53  kbracey
// Lots of consts.
//
// Revision 1.15  1998/06/14 08:26:43  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.14  1998/06/14 08:12:41  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 1.13  1998/06/12 09:12:44  kbracey
// Tidied up progress display some more: capitalisation of Block/block
// standardised to block, pluralisation sorted out, ensured that
// "Retrieved/sent n xxx blocks from/to server" lines fully blank out
// "n% transferred" lines.
//
// Revision 1.12  1998/06/09 12:35:06  kbracey
// Changed to print (3.29% done) instead of (03.29% done).
//
// Revision 1.11  1998/06/09 10:17:05  kbracey
// Cleared up Bovine's commit of Cyrus' stuff. Mostly putting all the consts
// back. Also updated RISC OS GUI.
//
// Revision 1.10  1998/06/09 08:54:27  jlawson
// Committed Cyrus' changes: phrase "keys/s" is changed back to "k/s" in
// CliGetSummaryStringForContest. That line has exactly 3 characters to 
// spare, and >9 blocks (or >9 days, or >999Kk/s) will cause line to wrap.
//
// Revision 1.9  1998/06/08 15:47:06  kbracey
// Added lots of "const"s and "static"s to reduce compiler warnings, and
// hopefully improve output code, too.
//
// Revision 1.8  1998/06/08 14:10:33  kbracey
// Changed "Kkeys to kkeys"
//
// Revision 1.7  1998/05/29 08:01:07  bovine
// copyright update, indents
//
// Revision 1.6  1998/05/27 18:21:27  bovine
// SGI Irix warnings and configure fixes
//
// Revision 1.5  1998/05/26 15:18:51  bovine
// fixed compile warnings for g++/linux
//
// Revision 1.4  1998/05/25 07:16:45  bovine
// fixed warnings on g++/solaris
//
// Revision 1.3  1998/05/25 05:58:32  bovine
// fixed warnings for Borland C++
//
// Revision 1.2  1998/05/25 02:54:18  bovine
// fixed indents
//
// Revision 1.1  1998/05/24 14:25:49  daa
// Import 5/23/98 client tree
//
// Revision 0.0  1998/05/01 05:01:08  cyruspatel
// Created
//
// ==============================================================

#if (!defined(lint) && defined(__showids__))
const char *clisrate_cpp(void) {
return "@(#)$Id: clisrate.cpp,v 1.32 1998/10/04 11:35:28 remi Exp $"; }
#endif

#include "cputypes.h"  // for u64
#include "problem.h"   // for Problem class
#include "client.h"    // for Fileentry struct
#include "baseincs.h"  // for timeval, sprintf et al
#include "clitime.h"   // for CliTimer(), CliTimerDiff(), CliGetTimeString()
#include "clirate.h"   // for CliGetKeyrateFor[Problem|Contest]()
#include "clicdata.h"  // for CliGetContestInfo[Base|Summary]Data()
#include "network.h"   // for ntohl()/htonl()
#include "clisrate.h"  // included just to keep prototypes accurate

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

    if (strlen(number) >= STR_LEN)
        return(number);

    strcpy(num_string, number);
    digits = strlen(num_string);
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

// returns keyrate as string (len<=26) "nnnn.nn ['k'|'M'|'G'|'T']"
// return value is a pointer to buffer.
char *CliGetKeyrateAsString( char *buffer, double rate )
{ 
  return (num_sep(__CliGetKeyrateAsString( buffer, rate, _U32LimitDouble_ )));
}

// ---------------------------------------------------------------------------

// "4 RC5 blocks 12:34:56.78 - [234.56 Kkeys/s]" 
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
    CliGetContestInfoSummaryData( contestid, &blocks, NULL, &ttime ); //ditto
    keyrateP=__CliGetKeyrateAsString(keyrate,
          CliGetKeyrateForContest(contestid),((double)(1000)));
  }
  else
  {
    name = "???";
    blocks = 0;
    ttime.tv_sec = 0;
    ttime.tv_usec = 0;
    keyrateP = "---.-- ";
  }

  sprintf(str, "%d %s block%s %s%c- [%skeys/s]", 
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

// Loaded 1 RC5 1*2^30 block 68E0D85A:A0000000 (10.25% done)
const char *CliGetMessageForFileentryLoaded( FileEntry *fileentry )
{
  static char str[84];
  const char *name;
  unsigned int size, count;
  u32 iter = ntohl(fileentry->iterations.lo);
  unsigned int startpercent = (unsigned int) ( (double) 10000.0 *
           ( (double) (ntohl(fileentry->keysdone.lo)) / (double)(iter) ) );


  /*                       old format: eg 1*2^30
  size = 1;
  count = 32;
  if (iter)
  {
    count = 1;
    size = 0;
    while ((iter & count)==0)
      { size++; count <<= 1; }
    count = iter / (1<<size);
  }
  */
  
  //iter = 268435456L;  //1<<28  //2^28 
  //iter = 536870912L;  //1<<29  //2^29
  //iter =1073741824L;  //1<<30  //2^30
  //iter =2147483648L;  //1<<31  //2^31
  
  if (!iter)                  /* new format 4*2^28 */
    {
    count = 16;
    size  = 28;
    }
  else
    {
    size =  0;
    while (iter>1 && size<28)
      { size++; iter>>=1; }
    count = iter;
    }

  if (CliGetContestInfoBaseData( fileentry->contest, &name, NULL )!=0) //clicdata
    name = "???";

  sprintf( str, "%s %s %d*2^%d block %08lX:%08lX%c(%u.%02u%% done)",
           "Loaded",
           name, (int) count, (int)size,  
           (unsigned long) ntohl( fileentry->key.hi ),
           (unsigned long) ntohl( fileentry->key.lo ),
           ((startpercent)?(' '):(0)),
           (unsigned)(startpercent/100), (unsigned)(startpercent%100) );
  return str;
}

// ---------------------------------------------------------------------------

// internal - with or without adjusting cumulative stats
// Completed RC5 block 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
static const char *__CliGetMessageForProblemCompleted( Problem *prob, int doSave )
{
  static char str[160];
  RC5Result rc5result;
  struct timeval tv;
  char keyrate[32];
  unsigned int mulfactor;
  const char *keyrateP, *name;
  int contestid = prob->GetResult( &rc5result );

  if (CliGetContestInfoBaseData( contestid, &name, &mulfactor )==0) //clicdata
    {
    keyrateP = CliGetKeyrateAsString( keyrate, 
        ((doSave) ? ( CliGetKeyrateForProblem( prob ) ) :
                    ( CliGetKeyrateForProblemNoSave( prob ) ))  );
    }                    
  else
    {
    keyrateP = "---.-- ";
    name = "???";
    }

  tv.tv_sec = prob->timehi;
  tv.tv_usec = prob->timelo;
  CliTimerDiff( &tv, &tv, NULL );

/*                               Old method with numeric keycount display
  sprintf( str, "Completed %s block %08lX:%08lX (%s keys)\n"
                "[%s] %s - [%skeys/sec]\n",
                name,
                (unsigned long) ntohl( rc5result.key.hi ) ,
                (unsigned long) ntohl( rc5result.key.lo ),
                CliGetU64AsString( &(rc5result.iterations), 1, contestid ),
                CliGetTimeString( NULL, 1 ),
                CliGetTimeString( &tv, 2 ),
                keyrateP );
*/

  unsigned int /* size=1, count=32, */ itermul=16;
  if (!rc5result.iterations.hi)
    {
    u32 iter = ntohl(rc5result.iterations.lo);
    /*
    count = 1;
    size = 0;
    while ((iter & count)==0)
      { size++; count <<= 1; }
    count = iter / (1<<size);
    iter = ntohl(rc5result.iterations.lo);
    */
    itermul = 0;
    while (iter > 1 && itermul < 28)
      { iter>>=1; itermul++; }
    itermul = (unsigned int)iter;
    }

//[Aug 01 10:00:00 GMT] Completed one RC5 block 00000000:00000000 (4*2^28 keys)\n"
  sprintf( str, "Completed one %s " /* "%d*2^%d " */ "block %08lX:%08lX (%u*2^28 keys)\n"
                "[%s] %s - [%skeys/sec]\n",  
                name, /* (int)size, (int)count, */
                (unsigned long) ntohl( rc5result.key.hi ) ,
                (unsigned long) ntohl( rc5result.key.lo ),
                (unsigned int)(itermul),
                CliGetTimeString( NULL, 1 ),
                CliGetTimeString( &tv, 2 ),
                keyrateP );

  return str;
}

// Completed RC5 block 68E0D85A:A0000000 (123456789 keys)
//          123:45:67:89 - [987654321 keys/s]
const char *CliGetMessageForProblemCompleted( Problem *prob )
{ return __CliGetMessageForProblemCompleted( prob, 1 ); }

const char *CliGetMessageForProblemCompletedNoSave( Problem *prob )
{ return __CliGetMessageForProblemCompleted( prob, 0 ); }

// ---------------------------------------------------------------------------

const char *CliReformatMessage( const char *header, const char *message )
{
  static char strspace[160];

  unsigned int prelen, linelen, doquote = (header!=NULL);
  char buffer[84];
  char *bptr, *sptr;
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

