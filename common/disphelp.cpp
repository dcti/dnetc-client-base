// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// Created 28. May 98 by Cyrus Patel <cyp@fb14.uni-mainz.de>
//
// call DisplayHelp() from main with the 'unrecognized option' argv[x]
// or NULL or "-help" or "help" (or whatever)

#include "client.h"


#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD)
#include <termios.h>
#endif


// --------------------------------------------------------------------------

#if !defined(NOCONFIG)
// read a single keypress, without waiting for an Enter if possible
static int readkeypress()
{
  int ch;

#if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOS) || \
  (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_NETWARE)
  ch = getch();
  if (!ch) ch = (getch() << 8);
#elif (CLIENT_OS == OS_RISCOS)
  ch = _swi(OS_ReadC, _RETURN(0));
#elif (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD)
  struct termios stored;
  struct termios newios;

  /* Get the original termios configuration */
  tcgetattr(0,&stored);

  /* Disable canonical mode, and set buffer size to 1 byte */
  memcpy(&newios,&stored,sizeof(struct termios));
  newios.c_lflag &= (~ICANON);
  newios.c_cc[VTIME] = 0;
  newios.c_cc[VMIN] = 1;

  /* Activate the new settings */
  tcsetattr(0,TCSANOW,&newios);

  /* Read the single character */
  ch = getchar();

  /* Restore the original settings */
  tcsetattr(0,TCSANOW,&stored);
#else
  ch = getchar();
#endif

  return ch;
}
#endif

// --------------------------------------------------------------------------

// provide a full-screen, interactive help for an invalid option (argv[x])

#if !defined(NOCONFIG)
void Client::DisplayHelp( const char * unrecognized_option )
{
  static const char *valid_help_requests[] =
  { "-help", "--help", "help", "-h", "/h", "/?", "-?", "?", "/help" };

  static const char *helpbody[] =
  {
  "Special Options: (the client will execute the option and then exit)",
  ""
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
#if defined(MULTITHREAD)
  "-numcpu <n>        run simultaneously on <n> CPUs",
#endif
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
  #if defined(WINNTSERVICE)
  "-install           install the client as an NT service",
  "-uninstall         uninstall the client if running as an NT service",
  #else
  "-hide              hide the client from the desktop",
  #endif
  "-lurk              automatically detect modem connections",
  "-lurkonly          perform buffer updates only when a connection is detected",
#elif (CLIENT_OS == OS_OS2)
  "-install           install the client in the startup folder",
  "-uninstall         remove the client from the startup folder",
  "-hide              run detached (hidden)",
#endif
  "-percentoff        don't display block completion as a running percentage",
  "-quiet             suppress screen output"
  };

  static const char *helpheader[] =
  {
  NULL, // "RC5DES v2.%d.%d client - a project of distributed.net" goes here
  #if (CLIENT_OS == OS_VMS)
    #if defined(MULTINET)
      "Compiled for OpenVMS with Multinet support",
    #elif defined(__VMS_UCX__)
      "Compiled for OpenVMS with UCX support",
    #elif defined(NONETWORK)
      "Compiled for OpenVMS with no network support",
    #endif
  #endif
  "Visit http://www.distributed.net/FAQ/ for in-depth command line help",
  "-------------------------------------------------------------------------"
  };

  int headerlines, bodylines, footerlines;
  int startline, maxscreenlines, maxpagesize;
  char whoami[64];

  if (unrecognized_option && *unrecognized_option)
  {
    bool done = false;
    for (int i = 0; i < (int) (sizeof(valid_help_requests)/sizeof(char *)); i++)
    {
      if (strcmpi(unrecognized_option,valid_help_requests[i]) == 0)
      {
        done = true;
        break;
      }
    }
    if (!done)
    {
      printf( "\nUnrecognized option '%s'\n"
         "Press enter/space to display a list of valid command line\n"
         "options or press any other key to quit... ", unrecognized_option );
      int i = readkeypress();
      printf("\n");
      if (i != '\n' && i != '\r' && i != ' ') return;
    }
  }

  maxscreenlines = 24;    /* you can decrease, but don't increase this */
  headerlines = (sizeof(helpheader) / sizeof(char *));
  bodylines = (sizeof(helpbody) / sizeof(char *));
  footerlines = 2;
  startline = 0;
  maxpagesize = maxscreenlines - (headerlines+footerlines);

  sprintf(whoami, "RC5DES v2.%d.%d client - a project of distributed.net",
                  CLIENT_CONTEST*100 + CLIENT_BUILD, CLIENT_BUILD_FRAC );
  helpheader[0] = whoami;

  while (true)
  {
    int i;
    clearscreen();

    for (i = 0; i < headerlines; i++)
      printf("%s\n", helpheader[i] );
    for (i = startline; i < (startline+maxpagesize); i++)
      printf("%s\n", helpbody[i] );

    if (startline == 0)
      printf("\nPress '+' for the next page... ");
    else if (startline >= ((bodylines-maxpagesize)-1))
      printf("\nPress '-' for the previous page... ");
    else
      printf("\nPress '+' or '-' for the next/previous page, or any other key to quit... ");

    i = readkeypress();
    if (i == '+' || i == '=' || i == ' ' || 
        i == 'f' || i == '\r' || i == '\n')
    {
      startline += maxpagesize;
      if ( startline >= (bodylines-maxpagesize))
        startline = (bodylines-maxpagesize) - 1;
    }
    else if (i == '-' || i == 'b')
    {
      startline -= maxpagesize;
      if ( startline < 0 )
        startline = 0;
    }
    else
    {
      break;
    }
  }
  printf("\n\n");
  return;
}
#endif

// --------------------------------------------------------------------------

