// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

// Created 28. May 98 by Cyrus Patel <cyp@fb14.uni-mainz.de>
// 
// call DisplayHelp() from main with the 'unrecognized option' argv[x]
// or NULL or "-help" or "help" (or whatever)

#include "client.h"

// --------------------------------------------------------------------------
// call from main() with the invalid option (argv[x]) that triggered help
// --------------------------------------------------------------------------

void Client::DisplayHelp( char * unrecognized_option )
{
  static char *valid_help_requests[] = 
  { "-help", "help", "-h", "/h", "/?", "-?", "?", "/help" };

  static char *helpbody[] =
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
  "-prefer <x>        set the preferred contest to RC5 or DES (1=RC5, 2=DES)",
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

  static char *helpheader[] = 
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

  unsigned int headerlines, bodylines, footerlines;
  unsigned int maxscreenlines, maxpagesize;
  int i, done, startline;
  char whoami[64];

  if (unrecognized_option && *unrecognized_option)
    {
    done = 0;
    for (i=0;i<(sizeof(valid_help_requests)/sizeof(char *));i++)
      {
      if (strcmpi(unrecognized_option,valid_help_requests[i])==0)
        {
        done = 1;
        break;
        }
      }
    if (!done)
      {
      printf( "Unrecognized option '%s'\n"
           "Press enter to display a list of valid command line options\n"
            "or press any other key to quit... ", unrecognized_option );
      i=0;
      while (!i)
        {        
        if ((i=getch())==0) //non-blocking or DOS-style getch()
          {
          #if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOS) || \
              (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) || \
              (CLIENT_OS == OS_NETWARE)
            getch(); //dos-style getch(), so this is an extended keystroke
          #else
            usleep(250000);
          #endif
          }
        }
      if (i!='\n' && i!='\r')
        return;
      } // if (!found)
    } //if (unrecognized_option && *unrecognized_option)

  maxscreenlines = 24; /* you can decrease, but don't increase this */
  headerlines = (sizeof(helpheader)/sizeof(char *));
  bodylines = (sizeof(helpbody)/sizeof(char *));
  footerlines = 2;
  done = startline = 0;
  maxpagesize = maxscreenlines - (headerlines+footerlines);

  sprintf(whoami, "RC5DES v2.%d.%d client - a project of distributed.net",
                  CLIENT_CONTEST*100 + CLIENT_BUILD, CLIENT_BUILD_FRAC );
  helpheader[0] = whoami;
 
  while (!done)
    {
    clearscreen();

    for (i=0;i<headerlines;i++)
      printf("%s\n", helpheader[i] );
    for (i=startline;i<(startline+maxpagesize);i++)
      printf("%s\n", helpbody[i] );
      
    if (startline == 0)
      printf("\nPress '+' for the next page... ");
    else if (startline >= ((bodylines-maxpagesize)-1))
      printf("\nPress '-' for the previous page... ");
    else 
      printf("\nPress '+' or '-' for the next/previous page, or any other key to quit... ");
  
    i = 0;
    while (!i)
      {
      if ((i=getch())==0) //non-blocking or DOS-style getch()
        {
        #if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOS) || \
            (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32) || \
            (CLIENT_OS == OS_NETWARE)
          getch(); //dos-style getch(), so this is an extended keystroke
        #else
          usleep(250000);
        #endif
        }
      else if (i=='+' || i=='\r' || i=='\n') 
        {
        startline += maxpagesize;
        if ( startline >= (bodylines-maxpagesize))
          startline = (bodylines-maxpagesize)-1;
        }
      else if (i=='-') 
        {
        startline -= maxpagesize;
        if ( startline < 0 )
          startline = 0;
        }
      else
        {
        done = 1;
        }
      }
    } // while !done
  
  clearscreen();
  return;
}

