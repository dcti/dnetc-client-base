// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: disphelp.cpp,v $
// Revision 1.54  1999/01/30 12:11:06  snake
//
// fixed a small typo affecting BSD/OS
//
// Revision 1.53  1998/12/05 22:19:57  cyp
// Added -kill (aka -shutdown) and -hup (aka -restart) to the list of switches.
//
// Revision 1.52  1998/11/22 14:54:36  cyp
// Adjusted to reflect changed -runonline, -runoffline, -n behaviour
//
// Revision 1.51  1998/11/19 20:54:00  cyp
// Updated command line options to reflect changed -until format.
//
// Revision 1.50  1998/11/09 20:05:22  cyp
// Did away with client.cktime altogether. Time-to-Checkpoint is calculated
// dynamically based on problem completion state and is now the greater of 1
// minute and time_to_complete_1_percent (an average change of 1% that is).
//
// Revision 1.49  1998/11/08 19:01:04  cyp
// Removed lots and lots of junk; DisplayHelp() is no longer a client method;
// unix-ish clients no longer use the internal pager.
//
// Revision 1.48  1998/11/08 14:30:44  remi
// char **location should be moved out of the loop in gettermheight()
//
// Revision 1.47  1998/10/26 03:15:21  cyp
// More tag fun.
//
// Revision 1.46  1998/10/19 13:42:53  cyp
// completed implementation of 'priority'.
//
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
return "@(#)$Id: disphelp.cpp,v 1.54 1999/01/30 12:11:06 snake Exp $"; }
#endif

#include "cputypes.h"
#include "version.h"  //CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h" //generic include
#include "cmpidefs.h" //strcmpi()
#include "triggers.h" //CheckExitRequestTriggerNoIO()
#include "logstuff.h" //LogScreenRaw()
#include "console.h"  //ConClear(), ConInkey()
#include "lurk.h"     //define LURK if 'lurk' is supported

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD)  || \
    (CLIENT_OS == OS_BEOS)  || (CLIENT_OS == OS_SOLARIS) || \
    (CLIENT_OS == OS_IRIX)  || (CLIENT_OS == OS_FREEBSD) || \
    (CLIENT_OS == OS_BSDI) || (CLIENT_OS == OS_AIX)     || \
    (CLIENT_OS == OS_OS390) || (CLIENT_OS == OS_NEXT)    || \
    (CLIENT_OS == OS_DYNIX) || (CLIENT_OS == OS_MACH)    || \
    (CLIENT_OS == OS_SCO)   || (CLIENT_OS == OS_OPENBSD) || \
    (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_HPUX)    || \
    (CLIENT_OS == OS_DGUX)  || (CLIENT_OS == OS_ULTRIX)
  #define NO_INTERNAL_PAGING  //internal paging is very un-unix-ish
#endif  

// --------------------------------------------------------------------------

// provide a full-screen, interactive help for an invalid option (argv[x])
// 'unrecognized_option' may be NULL or a null string

void DisplayHelp( const char * unrecognized_option )
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
  "-benchmark2        quick (but slightly inaccurate) client speed test",
  "-hup, -restart     restart all active clients",
  "-kill, -shutdown   gracefully shut down all active clients",
  "-help              display these help screens",
  "",
//----the runoffline/runbuffers lines are the longest a description may get-----#
  "Other Options:",
  "-runoffline        disable network access",
  "-runonline         enable network access",
  "-runbuffers        set -n == -1 (exit when buffers are empty)",
  /*
  "-run               normal run (override ini offlinemode/runbuffer settings)",
  */
  "-a <address>       proxy server name or IP address",
  "-p <port>          proxy server port number",
  "-e <address>       the email id by which you are known to distributed.net",
  #ifdef OLDNICENESS
  "-nice <[0-2]>      niceness",
  #else
  "-priority <[0-9]>  scheduling priority from 0 (lowest/idle) to 9 (normal/user)",
  #endif
  "-c <cputype>       cpu type (run -config for a list of valid cputype numbers)",
  "-numcpu <n>        run <n> threads/run on <n> cpus. 0 forces single-threading.",
  "-h <hours>         time limit in hours",
  "-n <count>         blocks to complete. -1 forces exit when buffer is empty.",
  "-until <HH:MM>     quit at HH:MM (eg 07:30)",
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
  "-noexitfilecheck   don't check for a 'exitrc5.now' command file",
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
#endif
#if (CLIENT_OS == OS_OS2)
  "-install           install the client in the startup folder",
  "-uninstall         remove the client from the startup folder",
#endif
   #ifdef LURK
  "-lurk              automatically detect modem connections",
  "-lurkonly          perform buffer updates only when a connection is detected",
  #endif
  "-percentoff        don't display block completion as a running percentage",
  "-quiet or -hide    suppress screen output (== detach for some clients)",
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
  int i, key, nopaging = (!ConIsScreen());
  char linebuffer[128];

  if (ConGetSize(NULL,&maxscreenlines) == -1)
    maxscreenlines = 25;
  headerlines = (sizeof(helpheader) / sizeof(char *));
  bodylines = (sizeof(helpbody) / sizeof(char *));
  footerlines = 2;
  startline = 0;
  maxpagesize = maxscreenlines - (headerlines+footerlines);

  #if defined(NO_INTERNAL_PAGING)
  nopaging = 1;
  #endif

  /* -------------------------------------------------- */

  if (unrecognized_option && *unrecognized_option)
    {
    int goodopt = 0;

    for (i = 0; ((goodopt == 0) && (i < (int)
         (sizeof(valid_help_requests)/sizeof(char *)))); i++)
      goodopt = (strcmpi(unrecognized_option,valid_help_requests[i])==0);

    if (goodopt == 0)
      {
      LogScreenRaw( "\nUnrecognized option '%s'\n\n", unrecognized_option);
      LogScreenRaw( "The following list of command line switches may be obtained\n"
             "at any time by running the client with the '-help' option.\n\n");
      if (!nopaging)
        {
        LogScreenRaw("Press enter to continue... ");
        key = ConInKey(-1); /* -1 == wait forever. returns zero if break. */
        LogScreenRaw( "\r                          \r" );
        if (CheckExitRequestTriggerNoIO())
          return;
        if (key != '\n' && key != '\r' && key != ' ')
          return;
        }
      }
    }

  /* -------------------------------------------------- */

  if (nopaging || (maxscreenlines > (headerlines+bodylines)))
    {
    for (i = 0; i < headerlines; i++)
      LogScreenRaw( "%s\n", helpheader[i] );
    for (i = 0; i < bodylines; i++)
      LogScreenRaw( "%s\n", helpbody[i] );
    return;
    }

  /* -------------------------------------------------- */

  key = 0;
  do{
    if (key == 0) /* refresh required */
      {
      ConClear();
      for (i = 0; i < headerlines; i++)
        LogScreenRaw( "%s\n", helpheader[i] );
      for (i = startline; i < (startline+maxpagesize); i++)
        LogScreenRaw( "%s\n", helpbody[i] );
      LogScreenRaw("\n");
      }

    if (startline == 0)
      strcpy( linebuffer, "Press '+' for the next page... ");
    else if (startline >= ((bodylines-maxpagesize)-1))
      strcpy( linebuffer, "Press '-' for the previous page... ");
    else
      strcpy( linebuffer, "Press '+' or '-' for the next/previous page,"
                          " or any other key to quit... ");
    LogScreenRaw( linebuffer );

    key = ConInKey(-1);
    
    linebuffer[i=strlen(linebuffer)]='\r';
    linebuffer[i+1]=0;
    for (--i;i>0;i--) 
      linebuffer[i]=' ';
    linebuffer[0]='\r';
    LogScreenRaw( linebuffer );
    
    if (CheckExitRequestTriggerNoIO())
      break;

    if (key == '+' || key == '=' || key == ' ' ||
      key == 'f' || key == '\r' || key == '\n')
      {
      if (startline <= ((bodylines-maxpagesize) - 1))
        {
        startline += maxpagesize;
        if ( startline >= (bodylines-maxpagesize))
          startline = (bodylines-maxpagesize);
        key = 0; //refresh required
        }
      }
    else if (key == '-' || key == 'b')
      {
      if (startline > 0)
        {
        startline -= maxpagesize;
        if ( startline < 0 )
          startline = 0;
        key = 0; //refresh required
        }
      }
    else
      {
      key = -1; //unknown keystroke, so quit
      }
    } while (key >= 0);
      
  return;
#endif
}

// --------------------------------------------------------------------------
