/*
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *disphelp_cpp(void) {
return "@(#)$Id: disphelp.cpp,v 1.67 1999/07/23 03:16:54 fordbr Exp $"; }

/* ----------------------------------------------------------------------- */

#include "cputypes.h"
#include "version.h"  //CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h" //generic include
#include "triggers.h" //CheckExitRequestTriggerNoIO()
#include "logstuff.h" //LogScreenRaw()
#include "console.h"  //ConClear(), ConInkey()

#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_NETBSD)  || \
    (CLIENT_OS == OS_BEOS)  || (CLIENT_OS == OS_SOLARIS) || \
    (CLIENT_OS == OS_IRIX)  || (CLIENT_OS == OS_FREEBSD) || \
    (CLIENT_OS == OS_BSDI)  || (CLIENT_OS == OS_AIX)     || \
    (CLIENT_OS == OS_OS390) || (CLIENT_OS == OS_NEXT)    || \
    (CLIENT_OS == OS_DYNIX) || (CLIENT_OS == OS_MACH)    || \
    (CLIENT_OS == OS_SCO)   || (CLIENT_OS == OS_OPENBSD) || \
    (CLIENT_OS == OS_SUNOS) || (CLIENT_OS == OS_HPUX)    || \
    (CLIENT_OS == OS_DGUX)  || (CLIENT_OS == OS_ULTRIX)
  #define NO_INTERNAL_PAGING  //internal paging is very un-unix-ish
#endif  

/* ------------------------------------------------------------------------ */

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
    "-test              tests for core errors",
    "-flush             flush all output buffers",
    "-fetch             fill all input buffers",
    "-update            fetch + flush",
    "-forceunlock <fn>  unlock buffer file <fn>",
    "-benchmark         tests the client speed",
    "-benchmark2        quick (but slightly inaccurate) client speed test",
    "-restart, -hup     restart all active clients",
    "-shutdown, -kill   gracefully shut down all active clients",
    "-pause             pause all (-unpause is equivalent to -restart)",
  #if (CLIENT_OS == OS_WIN32)
    "-install           install the client as a service",
    "-uninstall         uninstall the client if running as a service",
  #endif
  #if (CLIENT_OS == OS_OS2)
    "-install           install the client in the startup folder",
    "-uninstall         remove the client from the startup folder",
  #endif
//  "-import <fn> [cnt] import [cnt] packets from file <fn> into client buffers",
    "-import <fn>       import packets from file <fn> into client buffers",
    "-help              display this text",
    "",
/*  "------------------------------------ max width == 77 ------------------------" */
    "Project and buffer related options:",
    "",
    "-ini <filename>    override default name of INI file",
    "-e <address>       the email id by which you are known to distributed.net",
    "-nodisk            don't use disk buffer files",
    "-n <count>         packets to complete. -1 forces exit when buffer is empty.",
    "-runbuffers        set -n == -1 (exit when buffers are empty)",
    "-frequent          frequently check for empty buffers",
    "-blsize <n>        set a preferred packet size (2^n keys/packet)",
    "-b <n>             set in-buffer threshold to <n> packets",
    "-b2 <n>            set out-buffer threshold to <n> packets",
    "-inbase <filename> input buffer basename (ie without 'extension'/suffix)",
    "-outbase <filename> output buffer basename (ie without 'extension'/suffix)",
    "-ckpoint <fname>   set the name of the checkpoint file",
    "",
    "Network update related options:",
    "",
    "-runoffline        disable network access",
    "-runonline         enable network access",
    "-nettimeout <secs> set the network timeout. Use -1 to force blocking mode",
    "-a <address>       keyserver name or IP address",
    "-p <port>          keyserver port number",
    "-nofallback        don't fallback to a distributed.net keyserver",
    "-u <method>        use this UUE/HTTP encoding method (see -config)",
    "-ha <address>      http/socks proxy name or IP address",
    "-hp <port>         http/socks proxy port",
  #ifdef LURK
    "-lurk              automatically detect modem connections",
    "-lurkonly          perform buffer updates only when a connection is detected",
    "-interfaces <list> limit the interfaces to monitor for online/offline status",
  #endif
    "",
    "Performance related options:",
    "",
    "-c <cputype>       cpu type (run -config for a list of valid cputype numbers)",
    "-numcpu <n>        run <n> threads/run on <n> cpus. 0 forces single-threading.",
    "-priority <[0-9]>  scheduling priority from 0 (lowest/idle) to 9 (normal/user)",
#ifdef CSC_TEST
    "-csccore <[0-3]>   run CSC with various types of core",
#endif
    "",
    "Logging options:",
    "",
    "-l <filename>      name of the log file",
    "-smtplen <len>     max size (in bytes) of a mail message (0 means no mail)",
    "-smtpsrvr <host>   name or IP address of mail (SMTP) server",
    "-smtpport <port>   mail (SMTP) server port number",
    "-smtpfrom <id>     who the client should say is sending the message",
    "-smtpdest <id>     who the client should send mail to",
    "",
    "Miscellaneous runtime options:",
    "",
    "-h <hours>         time limit in hours",
    "-until <HH:MM>     quit at HH:MM (eg 07:30)",
    "-noexitfilecheck   don't check for a 'exitrc5.now' command file",
    "-pausefile <fn>    name of file that causes the client to pause",
    "-percentoff        don't display work completion as a running percentage",
    "-quiet or -hide    suppress screen output (== detach for some clients)",
    "-noquiet           don't suppress screen output (override ini quiet setting)"
  };
  
  static const char *helpheader[] =
  {
    "RC5DES v" CLIENT_VERSIONSTRING " client - a project of distributed.net",
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
  maxpagesize = maxscreenlines - (headerlines + footerlines);

  #if defined(NO_INTERNAL_PAGING)
  nopaging = 1;
  #endif

  /* -------------------------------------------------- */

  if (unrecognized_option && *unrecognized_option)
  {
    int goodopt = 0;

    for (i = 0; ((goodopt == 0) && (i < (int)
         (sizeof(valid_help_requests)/sizeof(char *)))); i++)
    {
      int n=0;
      for (;((valid_help_requests[i][n])!=0 && unrecognized_option[n]!=0);n++)
      {
        if (tolower(valid_help_requests[i][n])!=tolower(unrecognized_option[n]))
          break;
      }
      goodopt = ((valid_help_requests[i][n])==0 && unrecognized_option[n]==0);
    }

    if (!goodopt)
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
  do
  {
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
    for (--i; i > 0; i--) 
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

