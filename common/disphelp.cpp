// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: disphelp.cpp,v $
// Revision 1.45  1998/10/07 18:41:51  silby
// Removed the extra "v" that was being prepended to the version string in the help display.
//
// Revision 1.44  1998/10/05 02:43:28  cyp
// Removed "-nommx" from the option list. -nommx is/will be undocumented and
// for internal/developer use only.
//
// Revision 1.43  1998/10/04 19:43:00  remi
// Added help lines for -benchmark*
//
// Revision 1.42  1998/10/04 11:35:35  remi
// Id tags fun.
//
// Revision 1.41  1998/10/03 22:57:56  remi
// Added a line for the "-nommx" option.
//
// Revision 1.40  1998/10/03 05:43:33  cyp
// Genericified to use ConClear() and ConGetKey(). Why on earth do we
// "avoid_bad_interaction_with_external_pagers"? We should be avoiding the
// pagers themselves if they interact badly.
//
// Revision 1.39  1998/09/05 20:12:18  silby
// Change so that disphelp is a valid function for the win32gui, it just 
// doesn't do anything (perhaps a windows help file will open in the future?)
//
// Revision 1.38  1998/08/10 20:08:00  cyruspatel
// Removed reference to NO!NETWORK
//
// Revision 1.37  1998/08/02 16:17:58  cyruspatel
// Completed support for logging.
//
// Revision 1.36  1998/07/20 00:28:36  silby
// Change to combine NT Service and 95 CLI.
//
// Revision 1.35  1998/07/13 12:40:30  kbracey
// RISC OS update. Added -noquiet option.
//
// Revision 1.34  1998/07/13 03:29:59  cyruspatel
// Added 'const's or 'register's where the compiler was complaining about
// ambiguities. ("declaration/type or an expression")
//
// Revision 1.33  1998/07/08 05:19:30  jlawson
// updates to get Borland C++ to compile under Win32.
//
// Revision 1.32  1998/07/07 21:55:39  cyruspatel
// client.h has been split into client.h and baseincs.h 
//
// Revision 1.31  1998/07/06 01:41:19  cyruspatel
// Added support for /DNOTERMIOS and /DNOCURSES compiler switches to allow
// the linux client to be built on systems without termios/curses libraries
// or include files.
//
// Revision 1.30  1998/07/01 10:50:30  ziggyb
// -lurk/-lurkonly shows up on the -help in OS/2
//
// Revision 1.29  1998/06/29 04:22:23  jlawson
// Updates for 16-bit Win16 support
//
// Revision 1.28  1998/06/27 21:23:34  jlawson
// fixed tabs
//
// Revision 1.27  1998/06/24 06:45:15  remi
// The terminfo database can be located everywhere. Try some known locations
// before falling back to a 25 lines screen.
//
// Revision 1.26  1998/06/23 19:48:50  remi
// - Fixed pager crashing when terminal has more lines than help text
// - Fixed off by one bug in pageup/pagedown logic
//   (shows up when we can scroll by only one line)
// - Don't need to test ^C in Win32, signal handler catch it for us
//   when we usleep()
// - Fixed stupid #ifdef bug (it's "defined(TERMIOS)" and not "!defined()")
//
// Revision 1.25  1998/06/23 18:41:36  cyruspatel
// Removed fprintf(stderr,"**Break**") and SHELL_INSERT_NL_AT_END for NetWare
// and DOS. IMO, this is getting ridiculous!
//
// Revision 1.24  1998/06/23 13:53:37  kbracey
// Restored line ending type.
// Fixed paging for RISC OS.
//
// Revision 1.23  1998/06/23 09:23:39  remi
// - Added gettermheight support for Linux (and possibly for other *nixes)
//   (Add your OS to the list and add -lcurses to configure if it fit your
//   needs)
// - Turn off ECHO in readkeypress for Linux/NetBSD/BeOS
//   (so we can clear "--More--" when the user hit CR)
// - Resolved bad interaction between termios pager and external pager
// - Don't screw up the common "rc5des --help | more"
//   (Don't use our own pager if the user ask for help and stdout is
//   redirected, use it if the user gives a bad option)
// - The termios pager doesn't need an extra line at the bottom
// - Catch ^C under Win32 etc ... so the user can abort in the "--More--"
//   pager (why it doesn't get caught by the signal handler ?)
// - ^Break exit immediately under Win32 (not sure the kbhit()/usleep()
//   hack is the right thing to do under DOS/Netware/Win16)
//
// Revision 1.22  1998/06/22 17:29:09  remi
// Added gettermheight() so the pager is allowed to use more than 25 lines
// if there is more. Win32 only for the moment.
//
// Revision 1.21  1998/06/22 00:41:37  cyruspatel
// What started out as an intension to add two fflush()es turned into a four
// hour journey through hell :). Redirection is now handled properly and AFAIK
// thoroughly (not that anyone cares). Added --More-- type listing for people
// who don't like less (or when stdout is not a tty). If you don't like that
// either, then define NOMORE. And (I almost forgot) added two fflush()es.
//
// Revision 1.20  1998/06/21 02:41:53  silby
// This is a much improved way of disabling the pager. It looks decent. :)
//
// Revision 1.19  1998/06/21 02:36:26  silby
// Made changes so that the help display would detect if it was piped and
// not wait for user input.  It's ugly and could use work, but it prevents
// the former problem of the help screen appearing to lock up.
//
// Revision 1.18  1998/06/15 12:03:58  kbracey
// Lots of consts.
//
// Revision 1.17  1998/06/15 06:18:35  dicamillo
// Updates for BeOS
//
// Revision 1.16  1998/06/14 08:26:47  friedbait
// 'Id' tags added in order to support 'ident' command to display a bill of
// material of the binary executable
//
// Revision 1.15  1998/06/14 08:12:51  friedbait
// 'Log' keywords added to maintain automatic change history
//
// Revision 0.00  1998/05/28 28:05:07  cyruspatel
// Created

// Created 28. May 98 by Cyrus Patel <cyp@fb14.uni-mainz.de>
//
// call DisplayHelp() from main with the 'unrecognized option' argv[x]
// or NULL or "-help" or "help" (or whatever)
//

// -----------------------------------------------------------------------

#if (!defined(lint) && defined(__showids__))
const char *disphelp_cpp(void) {
return "@(#)$Id: disphelp.cpp,v 1.45 1998/10/07 18:41:51 silby Exp $"; }
#endif

#include "cputypes.h"
#include "version.h"  //CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "client.h"   //client class
#include "baseincs.h"
#include "cmpidefs.h" //strcmpi()
#include "sleepdef.h"
#include "triggers.h" //[Check|Raise]ExitRequestTrigger()
#include "logstuff.h" //Log()/LogScreen()
#include "console.h"  //ConClear(), ConInkey()

// --------------------------------------------------------------------------

//#define NOLESS // (aka NOPAGER) define if you don't like less (+/- paging)
//#define NOMORE // define this if you don't like --More--

#ifndef NOTERMIOS
#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_BEOS)
#include <termios.h>
#include "sleepdef.h"
#define TERMIOSPAGER
#endif
#endif

#ifndef NOCURSES
// Other *nixes may want to use this (add "-lcurses" to configure / Makefile)
#if (CLIENT_OS == OS_LINUX)
#include <curses.h>
#include <term.h>
#define TERMINFOLINES
#endif
#endif

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
#define SHELL_INSERT_NL_AT_END
#endif

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
// read a single keypress, without waiting for an Enter if possible
static int readkeypress()
{
#if defined(TERMIOSPAGER)
  usleep (200*1000);
  /* Wait a bit to avoid bad interaction with external pagers 
   *
   * Without this delay (just a guess) :
   * 1) We start outputing some text, change the termios settings
   *    and wait for a key.
   * 2) The external pager comes into action, gets the current termios
   *    settings and change them to something suitable for it.
   * 3) We get the key, and restore the original termios settings
   *    (the ones suitable for shell interaction, but not the ones suitable
   *    for the external pager, strange things occurs)
   * 4) The client terminates
   * 5) User exit from the external pager
   * 6) The external pager restore termios settings to the values
   *    we set in this routine. Everythhing's broken, even more with
   *    ECHO turned off :-(
   *
   * {1,3,4} and {2,5,6} occurs asynchonously.
   *
   * With this delay the external pager can fetch the shell termios settings
   * and change them to its own flavour before we start cooking them up.
   *
   * This is a hack, it will not work if the external pager takes more than
   * 2/10s to start up...
   */
#endif  /* the rest is generic */

  return ConInKey(-1); /* -1 == wait forever. returns zero if break. */
}
#endif

// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
// How many visible lines there is in this terminal ?
static int gettermheight()
{

#if (CLIENT_OS == OS_WIN32)

  HANDLE hStdout;
  CONSOLE_SCREEN_BUFFER_INFO csbiInfo;

  hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hStdout == INVALID_HANDLE_VALUE) return -1;
  if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo)) return -1;
  return csbiInfo.srWindow.Bottom - csbiInfo.srWindow.Top + 1;

#elif (CLIENT_OS == OS_RISCOS)

  // nlines = TWBRow - TWTRow + 1
  static const int var[3] = { 133, 135, -1 };
  int value[3];

  if (riscos_in_taskwindow)
    return -1;

  if (_swix(OS_ReadVduVariables, _INR(0,1), var, value))
    return -1;

  return value[0] - value[1] + 1;


#elif defined(TERMINFOLINES)

  // grrr... terminfo database location is installation dependent
  // search some standard (?) locations
  char *terminfo_locations[] = {
      "/usr/share/terminfo",       // ncurses 1.9.9g defaults
      "/usr/local/share/terminfo", //
      "/usr/lib/terminfo",         // Debian 1.3x use this one
      "/usr/local/lib/terminfo",   // variation
      "/etc/terminfo",             // found something here on my machine, doesn't hurt
      "~/.terminfo",               // last resort
      NULL                         // stop tag
  };
  for (;;) {
    char **location = &terminfo_locations[0];
    int termerr;
    if (setupterm( NULL, 1, &termerr ) == ERR) {
      if ((termerr == 0 || termerr == -1) && *location != NULL)
        setenv( "TERMINFO", *(location++), 1);
      else
        return -1;
    } else
      break;
  }

  int nlines = tigetnum( "lines" );
  // check for insane values
  if (nlines <= 0 || nlines >= 300)
    return -1;
  else
    return nlines;

#else
  // check for common $LINES / $COLUMNS environment variables
  char *p = getenv( "LINES" );
  if (!p) return -1;
  int nlines = atoi( p );
  // check for insane values
  if (nlines <= 0 || nlines >= 300)
    return -1;
  else
    return nlines;
#endif
}
#endif

// --------------------------------------------------------------------------

// provide a full-screen, interactive help for an invalid option (argv[x])

void Client::DisplayHelp( const char * unrecognized_option )
{
#if !defined(NOCONFIG)
  static const char *valid_help_requests[] =
  { "-help", "--help", "help", "-h", "/h", "/?", "-?", "?", "/help" };

  static const char *helpbody[] =
  {
  "Special Options: (the client will execute the option and then exit)",
  "-config            start the configuration menu",
  "-test              tests for client errors",
  "-flush             flush all output buffers",
  "-fetch             fill all input buffers",
  "-forceflush        ignore most errors & retry",
  "-forcefetch        ignore most errors & retry",
  "-update            fetch + flush",
  "-forceunlock <fn>  unlock buffer file <fn>",
  "-benchmark         tests the client speed",
  "                   (add rc5 or des to test only one core)",
  "-benchmark2        quick (but slightly inaccurate) client speed test",
  "                   (add rc5 or des to test only one core)",
  "-help              display these help screens",
  "",
//----the runoffline/runbuffers lines are the longest a description may get-----#
  "Other Options:",
  "-runoffline        don't attempt any flush/fetches (generate random if needed)",
  "-runbuffers        like -runoffline, but exit when the current buffer is empty",
  "-run               normal run (override ini offlinemode/runbuffer settings)",
  "-a <address>       proxy server name or IP address",
  "-p <port>          proxy server port number",
  "-e <address>       the email id by which you are known to distributed.net",
  "-nice <[0-2]>      niceness",
  "-c <cputype>       cpu type (run -config for a list of valid cputype numbers)",
  "-numcpu <n>        run <n> threads/run on <n> cpus",
  "-h <hours>         time limit in hours",
  "-n <count>         blocks to complete",
  "-until <HHMM>      quit at HHMM (eg 0700)",
  "-u <uuehttp>       use UUE/HTTP mode",
  "-ha <address>      http proxy name or IP address",
  "-hp <port>         http proxy port",
  "-ini <filename>    override default name of INI file",
  "-nodisk            don't use disk buffer files",
  "-b <blocks>        maximum number of blocks in an RC5 buffer file",
  "-b2 <blocks>       maximum number of blocks in an DES buffer file",
  "-in <filename>     override name of RC5 input buffer file",
  "-out <filename>    override name of RC5 output buffer file",
  "-in2 <filename>    override name of DES input buffer file",
  "-out2 <filename>   override name of DES output buffer file",
  "-ckpoint <fname>   set the name of the RC5 checkpoint file",
  "-ckpoint2 <fn>     set the name of the RC5 checkpoint file",
  "-cktime <min>      set the number of seconds between saving checkpoints",
  "-noexitfilecheck   don't check for a 'exitrc5.now' command file",
  "-exitfilechecktime <t> number of seconds that must elapse between checks",
  "-pausefile <fn>    name of file that causes the client to pause",
  "-processdes <x>    determines if the client will compete in DES contests",
  "-l <filename>      name of the log file",
  "-nofallback        don't fallback to a distributed.net proxy",
  "-smtplen <len>     max size (in bytes) of a mail message (0 means no mail)",
  "-smtpsrvr <nm>     name or IP address of mail (SMTP) server",
  "-smtpport <port>   mail (SMTP) server port number",
  "-smtpfrom <id>     who the client should say is sending the message",
  "-smtpdest <id>     who the client should send mail to",
  "-nettimeout <x>    set the network timeout to x",
  "-frequent          attempt updates often",
  "-blsize <n>        set a preferred blocksize (2^n)",
#if (CLIENT_OS == OS_WIN32)
  "-install           install the client as a service",
  "-uninstall         uninstall the client if running as a service",
  "-hide              run hidden",
#endif
#if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32)
  "-lurk              automatically detect modem connections",
  "-lurkonly          perform buffer updates only when a connection is detected",
#endif
#if (CLIENT_OS == OS_OS2)
  "-install           install the client in the startup folder",
  "-uninstall         remove the client from the startup folder",
  "-hide              run detached (hidden)",
#endif
  "-percentoff        don't display block completion as a running percentage",
  "-quiet             suppress screen output",
  "-noquiet           don't suppress screen output (override ini quiet setting)"
  };

  static const char *helpheader[] =
  {
  "RC5DES " CLIENT_VERSIONSTRING " client - a project of distributed.net",
  "Visit http://www.distributed.net/FAQ/ for in-depth command line help",
  "-------------------------------------------------------------------------"
  };

  int headerlines, bodylines, footerlines;
  int startline, maxscreenlines, maxpagesize;
  int foundhelprequest = 0;

  int nostdin, forcenopagemode = 0; //forcenopagemode is "--More--" mode
  FILE *outstream, *teestream;

  #if defined(NOLESS) || defined(NOPAGER) // fyi: noless was previously
    forcenopagemode = 1;                  // called nopager
  #endif

  nostdin = (!isatty(fileno(stdin))); //may not work for </dev/nul
  outstream = stdout;
  if (isatty(fileno(outstream))) //normal mode
    {
    teestream = NULL;
    }
  else if (isatty(fileno(stderr))) // could dup() but thats not supported
    {                              // everywhere
    teestream = outstream; //stdout
    outstream = stderr;
    forcenopagemode = 1;  //paging works, but turn it off
    }                     //for aesthetic reasons. :)
  else //neither is a tty, so leave outstream==stdout and turn off paging
    {
    nostdin = 1;
    teestream = NULL;
    }

  if (unrecognized_option && *unrecognized_option)
    {
    for (int i = 0; ((!foundhelprequest) && (i < (int)
         (sizeof(valid_help_requests)/sizeof(char *)))); i++)
      foundhelprequest = (strcmpi(unrecognized_option,valid_help_requests[i]) == 0);
    if (!foundhelprequest)
      {
      fprintf( outstream, "\nUnrecognized option '%s'\n", unrecognized_option);
      if (teestream)
        fprintf( teestream, "\nUnrecognized option '%s'\n", unrecognized_option);
      const char *msg = "\n\nThe following list may be obtained at any time by "
                        "running\nthe client with the '-help' option.\n\n";
      fprintf( outstream, ((nostdin || forcenopagemode)?(msg):
          ("Press enter/space to display a list of valid command line\n"
          "options or press any other key to quit... ")) );
      if (teestream)
        fprintf( teestream, msg );
      fflush( outstream );
      if (!nostdin && !forcenopagemode)
        {
        int i = readkeypress();
        if (CheckExitRequestTriggerNoIO())
          return;
        fprintf( outstream, "\n" );
        if (i != '\n' && i != '\r' && i != ' ')
          return;
        }
      }
    }

  if ((maxscreenlines = gettermheight()) == -1) maxscreenlines = 25;
#if defined(SHELL_INSERT_NL_AT_END) || !defined(TERMIOSPAGER)
  maxscreenlines--;
#endif
  headerlines = (sizeof(helpheader) / sizeof(char *));
  bodylines = (sizeof(helpbody) / sizeof(char *));
  footerlines = 2;
  startline = 0;
  maxpagesize = maxscreenlines - (headerlines+footerlines);

  if (teestream) //we do this first, so we're not paging there
    {
    int i;
    for (i = 0; i < headerlines; i++)
      fprintf( teestream, "%s\n", helpheader[i] );
    for (i = 0; i < bodylines; i++)
      fprintf( teestream, "%s\n", helpbody[i] );
    }
  if (nostdin || forcenopagemode) //stdin is redirected or NOLESS
    {
    if (!foundhelprequest || !teestream)
      {
      int i, l, n=maxpagesize-5; // -5 to see the 'invalid option' message
      for (i = 0; i < headerlines; i++)
        fprintf( outstream, "%s\n", helpheader[i] );
      for (l = 0; l < bodylines; )
        {
          for (i = 0; (l < bodylines) && (i < n); i++ )
            fprintf( outstream, "%s\n", helpbody[l++] );
          n = maxscreenlines-2; //use a two line overlap
          if (l<bodylines && !nostdin && !foundhelprequest)
            {  //NOLESS mode: stdin is ok
              #ifndef NOMORE // very obstinate people :)
              fprintf( outstream, "--More--" );
              fflush( outstream );
              readkeypress();
              if (CheckExitRequestTriggerNoIO())
                break;
              fprintf( outstream, "\r" ); //overwrite the --More--
              #endif
            }
        }
      }
    }
  else if (maxpagesize >= bodylines) { // enough lines in a single screen ?
    int i;
    for (i = 0; i < headerlines; i++)
      fprintf( outstream, "%s\n", helpheader[i] );
    for (i = startline; i < bodylines; i++)
      fprintf( outstream, "%s\n", helpbody[i] );
  }
  else  //stdout may or may not be redirected
    {
    int i = 1;
    do{
      if (i > 0) //do we need a screen refresh?
        {
        if (teestream) //can't use clearscreen if not stdout
          {
          for (i=0;i<10;i++) // 10*20 should be more than enough :)
            fprintf( outstream, "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n");
          }
        else
          ConClear(); //this applies to stdout only

        for (i = 0; i < headerlines; i++)
          fprintf( outstream, "%s\n", helpheader[i] );
        for (i = startline; i < (startline+maxpagesize); i++)
          fprintf( outstream, "%s\n", helpbody[i] );

        if (startline == 0)
          fprintf( outstream, "\nPress '+' for the next page... ");
        else if (startline >= ((bodylines-maxpagesize)-1))
          fprintf( outstream, "\nPress '-' for the previous page... ");
        else
          fprintf( outstream, "\nPress '+' or '-' for the next/previous page,"
                  " or any other key to quit... ");
        }

      fflush( outstream );
      i = readkeypress();
      if (CheckExitRequestTriggerNoIO())
        break;
      fprintf( outstream, "\r");

      if (i == '+' || i == '=' || i == ' ' ||
        i == 'f' || i == '\r' || i == '\n')
        {
        i = 0; //assume no refresh required
        if (startline <= ((bodylines-maxpagesize) - 1))
          {
          startline += maxpagesize;
          if ( startline >= (bodylines-maxpagesize))
            startline = (bodylines-maxpagesize);
          i = 1; //signal refresh required
          }
        //else if (teestream) //stdout is redirected, so clean up after getch()
        //  i = 1;
        }
      else if (i == '-' || i == 'b')
        {
        i = 0; //assume no refresh required
        if (startline > 0)
          {
          startline -= maxpagesize;
          if ( startline < 0 )
            startline = 0;
          i = 1; //signal refresh required
          }
        //else if (teestream) //stdout is redirected, so clean up after getch()
        //  i = 1;
        }
      else
        {
        i = -1; //unknown keystroke, so quit
        }
      } while (i >= 0);
#if !defined(SHELL_INSERT_NL_AT_END) && (defined(TERMIOSPAGER) || (CLIENT_OS == OS_RISCOS))
      // clear end of line (pager line)
      // put spaces in case ANSI is not supported
      #if (CLIENT_OS == OS_RISCOS)
      if (!teestream && !nostdin)
        {
        static char clear[] = { 13, 23, 8, 4, 6, 0, 0, 0, 0, 0, 0 };
        _swix(OS_WriteN, _INR(0,1), clear, sizeof clear);
        }
      #else
      if (!teestream && !nostdin) fprintf( outstream, "\r\x1B[K\r    \r" );
      #endif
#endif
    } //stdin is a tty

  return;
#endif
}

// --------------------------------------------------------------------------

