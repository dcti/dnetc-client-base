/*
 * Copyright distributed.net 1997 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *disphelp_cpp(void) {
return "@(#)$Id: disphelp.cpp,v 1.64.2.11 2000/04/14 18:11:51 cyp Exp $"; }

/* ----------------------------------------------------------------------- */

#include "cputypes.h"
#include "version.h"  //CLIENT_CONTEST, CLIENT_BUILD, CLIENT_BUILD_FRAC
#include "baseincs.h" //generic include
#include "triggers.h" //CheckExitRequestTriggerNoIO()
#include "logstuff.h" //LogScreenRaw()
#include "util.h"     //UtilGetAppName()
#include "console.h"  //ConClear(), ConInkey()
#include "client.h"   //various #defines

#if defined(__unix__) && !defined(__EMX__)
  #define NO_INTERNAL_PAGING  //internal paging is very un-unix-ish
#endif

/* ------------------------------------------------------------------------ */

static const char *helpbody[] =
{
/*"------------------------------------ max width == 77 ------------------------" */
  "Special Options: (the client will execute the option and then exit)",
  "-config            start the configuration menu",
  "-flush             flush all output buffers",
  "-fetch             fill all input buffers",
  "-update            fetch + flush",
  "-benchmark [pn]    16-20 sec speed check [optional: only project pn]",
  "-benchmark2 [pn]   half (8-10 sec) and slightly inaccurate -benchmark",
  "-bench [pn]        -benchmark all cores [optional: only project pn]",
  "-test [pn]         tests for core errors [optional: only project pn]",
  "-restart           restart all active clients (equivalent to -hup)",
  "-shutdown          gracefully shut down all active clients",
  "-pause             pause all active clients",
  "-unpause           unpause all active clients",
#if (CLIENT_OS == OS_WIN32)
  "-install           install the client as a service",
  "-uninstall         uninstall the client previously -installed",
  "-svcstart          start a previously -installed client-as-service",
  "                   equivalent to NT's 'net start ...'",
#endif
#if (CLIENT_OS == OS_OS2)
  "-install           install the client in the startup folder",
  "-uninstall         remove the client from the startup folder",
#endif
//"-import <fn> [cnt] import [cnt] packets from file <fn> into client buffers",
  "-import <fn>       import packets from file <fn> into client buffers",
  "-forceunlock <fn>  unlock buffer file <fn>",
  "-help              display this text",
  "",
/*"------------------------------------ max width == 77 ------------------------" */
  "Project and buffer related options:",
  "",
  "-ini <filename>    override default name of INI file",
  "-e <address>       the email id by which you are known to distributed.net",
  "-nodisk            don't use disk buffer files",
  "-n <count>         packets to complete. -1 forces exit when buffer is empty.",
  "-runbuffers        set -n == -1 (exit when buffers are empty)",
  "-frequent          frequently check if buffers need topping-up",
  "-inbase <fname>    input buffer basename (ie without 'extension'/suffix)",
  "-outbase <fname>   output buffer basename (ie without 'extension'/suffix)",
  "-ckpoint <fname>   set the name of the checkpoint file",
  "-blsize <pn> <n>   set preferred packet size (2^n keys/packet)",
  "-bin <pn> <n>      set fetch buffer threshold to <n> work units",
  #if !defined(NO_OUTBUFFER_THRESHOLDS)
  "-bout <pn> <n>     set flush buffer threshold to <n> work units",
  "-b <pn> <n>        set both buffer thresholds to <n> work units",
  #endif
  "-btime <pn> <n>    set fetch time threshold to <n> hours",
  "                   If not specified, project name <pn> defaults to RC5",
  "",
/*"------------------------------------ max width == 77 ------------------------" */
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
/*"------------------------------------ max width == 77 ------------------------" */
  "Performance related options:",
  "",
  "-c <pn> <n>        core number (run -config for a list of valid core numbers)",
  "                   project name \"pn\" defaults to RC5",
  "-numcpu <n>        run <n> threads/run on <n> cpus. 0 forces single-threading.",
  "-priority <0-9>    scheduling priority from 0 (lowest/idle) to 9 (normal/user)",
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
/*"------------------------------------ max width == 77 ------------------------" */
  "Miscellaneous runtime options:",
  "",
  "-h <hours>         time limit in hours",
  "-until <HH:MM>     quit at HH:MM (eg 07:30)",
  "-noexitfilecheck   don't check for an 'exitrc5.now' file",
  "-pausefile <fn>    name of file that causes the client to pause",
  "-percentoff        don't display work completion as a running percentage",
  "-quiet/-hide       suppress screen output (== detach for some clients)",
  "-noquiet           don't suppress screen output (override ini quiet setting)"
};

/* ------------------------------------------------------------------------ */

#define _istrofspecial(_c) (!(!strchr("\\-\"",_c)))

void GenerateManPage( void )
{
  #if defined(__unix__) || defined(__GNUC__)
  char buffer[80];
  const char *appname = utilGetAppName();
  FILE *manp;

  strncpy(buffer,appname,sizeof(buffer));
  buffer[sizeof(buffer)-1] = '\0';
  strcpy( buffer, buffer );
  strcat( buffer, ".1" );

  manp = fopen(buffer,"w");
  if (!manp)
    fprintf(stderr,"Unable to create %s", buffer );
  else
  {
    unsigned int linelen, pos;
    char *p; const char *cp;
    time_t t = time(NULL);
    struct tm *gmt = gmtime(&t);

    fprintf(manp, ".\\\" Copyright (c) 1996-%d\n", gmt->tm_year+1900 );
    fprintf(manp, ".\\\"         distributed.net. All rights reserved.\n" );
    fprintf(manp, ".\\\"\n");
    fprintf(manp, ".\\\" %s\n",disphelp_cpp());
    fprintf(manp, ".\\\"\n");
    fprintf(manp, ".Dd %s", ctime(&t));
    strncpy(buffer, appname,sizeof(buffer));
    buffer[sizeof(buffer)-1] = '\0';
    for (pos=0;buffer[pos];pos++)
      buffer[pos]=(char)toupper(buffer[pos]);
    fprintf(manp, ".Dt %s 1\n", buffer );
    //fprintf(manp, ".Os "CLIENT_OS_NAME"\n");
    fprintf(manp, ".Sh NAME\n");
    fprintf(manp, ".Nm %s\n", appname);
    fprintf(manp, ".Nd distributed.net distributed computing client for "
                    CLIENT_OS_NAME"\n" );

    fprintf(manp,"\n");
    fprintf(manp, ".Sh SYNOPSIS\n");
    fprintf(manp, ".Nm %s\n", appname);
    for (pos=0;pos<(sizeof(helpbody)/sizeof(helpbody[0]));pos++)
    {
      cp = helpbody[pos];
      if (*cp == '-')
      {
        strncpy(buffer,helpbody[pos]+1,sizeof(buffer));
        buffer[sizeof(buffer)-1]='\0';
        p = &buffer[0];
        while (*p && *p!=' ')
          p++;
        while (*p==' ' && (p[1]=='<' || p[1]=='['))
        {
          while (*p && *p!='>' && *p!=']')
            p++;
          while (*p && *p!=' ')
            p++;
        }
        *p='\0';
        fprintf(manp,".Op \"\\-");
        for (linelen=0;buffer[linelen];linelen++)
        {
          if (buffer[linelen]=='\"')
            buffer[linelen] = '\'';
          else if (_istrofspecial(buffer[linelen]))
            fputc('\\', manp);
          fputc(buffer[linelen],manp);
        }
        fprintf(manp,"\"\n");
      }
    }

    fprintf(manp,"\n");
    fprintf(manp, ".Sh DESCRIPTION\n");
    fprintf(manp,
      ".Ar %s\nis a distributed computing client that coordinates with servers\n"
      "operated by\n.Ar distributed.net\nto cooperate with other network-connected\n"
      "computers to work on a common task.  It communicates over public networks\n"
      "and processes work assigned by the\n.Ar distributed.net\nkeyservers.\n"
      "It is designed to run in idle time so as to not impact the normal operation\n"
      "of the computer.\n", appname);

    fprintf(manp,"\n");
    fprintf(manp, ".Sh INSTALLATION\n");
    fprintf(manp,
      "Since you are already reading this, I assume you know how to\n"
      "unpack an archive (don't laugh!) into a directory of your\n"
      "choice.\n"
      ".sp 1\n"
      "Now, simply fire up the client...\n"
      ".sp 1\n"
      "If you have never run the client before, it will initiate the\n"
      "menu-driven configuration. Save and quit when done, the configuration\n"
      "file will be saved \\fBin the same directory as the client.\\fP \n"
      "Now, simply restart the client. From that point on it will use the\n"
      "saved configuration.\n"
      ".sp 1\n"
      "The configuration options are fairly self-explanatory and can be run\n"
      "at any time by starting the client with the '-config' option.\n"
      "A list of command line options is listed below.\n"
      );

    fprintf(manp,"\n");
    fprintf(manp, ".Sh COMMAND LINE OPTIONS\n");

    for (pos=0;pos<(sizeof(helpbody)/sizeof(helpbody[0]));pos++)
    {
      cp = helpbody[pos];
      if (*cp=='-')
      {
        fprintf(manp,".It Fl ");
        cp++;
        while (*cp && *cp != ' ')
        {
          if (*cp == '\"')
          {
            cp++;
            fputc('\'', manp);
          }
          else
          {
            if (_istrofspecial(*cp))
              fputc('\\', manp);
            fputc(*cp++,manp);
          }
        }
        while (*cp == ' ')
          cp++;
        while (*cp == '<' || *cp == '[')
        {
          const char closure = ((*cp == '<')?('>'):(']'));
          fprintf(manp, (*cp == '<')?(" Ar "):(" Op "));
          cp++;
          while (*cp && *cp!=closure)
          {
            if (*cp == '\"')
            {
              cp++;
              fputc('\'', manp);
            }
            else
            {
              if (_istrofspecial(*cp))
                fputc('\\', manp);
              fputc(*cp++,manp);
            }
          }
          if (*cp == closure)
            cp++;
          while (*cp == ' ')
            cp++;
        }
        fprintf(manp,"\n");
        if (*cp)
        {
          while (*cp)
          {
            if (*cp == '\"')
            {
              cp++;
              fputc('\'', manp);
            }
            else
            {
              if (_istrofspecial(*cp))
                fputc('\\', manp);
              fputc(*cp++,manp);
            }
          }
          fprintf(manp,"\n");
        }
      }
      else if (*cp == ' ') /* continuation */
      {
        while (*cp && *cp==' ')
          cp++;
        if (*cp)
        {
          while (*cp)
          {
            if (*cp == '\"')
            {
              cp++;
              fputc('\'', manp);
            }
            else
            {
              if (_istrofspecial(*cp))
                fputc('\\', manp);
              fputc(*cp++,manp);
            }
          }
          fprintf(manp,"\n");
        }
      }
      else if (*cp) /* new section */
      {
        if (pos)
          fprintf(manp,".El\n");
        fprintf(manp, ".sp 2\n");
        fprintf(manp,".Ss \"");
        while (*cp)
        {
          if (*cp == '\"')
          {
            cp++;
            fputc('\'', manp);
          }
          else
          {
            if (_istrofspecial(*cp))
              fputc('\\', manp);
            fputc(*cp++,manp);
          }
        }
        fprintf(manp,"\"\n");
        fprintf(manp,".Bl -tag -width Fl\n");
      }
    }

    #if 0
    fprintf(manp,"\n");
    fprintf(manp,".Sh ENVIRONMENT\n"
                 ".Pp\N"
                 ".Ip \\\"RC5INI\\\"\n"
                 "Full path to alternate .ini file\n");
    #endif
    fprintf(manp,"\n");
    fprintf(manp,".Sh SEE ALSO\n"
                 ".Pp\n"
                 "Client documentation: %s.txt and http://www.distributed.net/FAQ/\n",
                 appname);
    fprintf(manp,"\n");
    fprintf(manp,".Sh AUTHOR\n"
                 "distributed.net\n"
                 "http://www.distributed.net/\n");

    fclose(manp);
  }
  #endif /* __unix__ */
  return;
}

/* ------------------------------------------------------------------------ */

// provide a full-screen, interactive help for an invalid option (argv[x])
// 'unrecognized_option' may be NULL or a null string

void DisplayHelp( const char * unrecognized_option )
{
  static const char *valid_help_requests[] =
  { "-help", "--help", "help", "-h", "/h", "/?", "-?", "?", "/help" };

  static const char *helpheader[] =
  {
    "distributed.net v" CLIENT_VERSIONSTRING " client for " CLIENT_OS_NAME,
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
}

