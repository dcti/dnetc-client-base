/* Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * -------------------------------------------------------------------
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * *All* option handling is performed by ParseCommandLine(), including
 * options loaded from an external .ini.
 *
 * Note to porters: your port can be expected to break frequently if your
 * implementation does not call this or does start the client via the
 * Client::Main() in client.cpp
 * -------------------------------------------------------------------
*/
const char *cmdline_cpp(void) {
return "@(#)$Id: cmdline.cpp,v 1.133.2.53 2000/04/24 18:20:31 ctate Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "pathwork.h"  // InitWorkingDirectoryFromSamplePaths();
#include "lurk.h"      // dialup object
#include "util.h"      // trace, utilGetAppName()
#include "sleepdef.h"  // usleep()
#include "modereq.h"   // get/set/clear mode request bits
#include "console.h"   // ConOutErr()
#include "clitime.h"   // CliTimer() for -until setting
#include "confrwv.h"   // ValidateConfig()
#include "clicdata.h"  // CliGetContestNameFromID()
#include "cmdline.h"   // ourselves
#include "triggers.h"  // TRIGGER_PAUSE_SIGNAL
#include "confopt.h"   // conf_options[] for defaults/ranges
#include "selcore.h"   // selcoreValidateCoreIndex()

#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
    (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_OPENBSD)
#include <dirent.h> /* for direct read of /proc/ */
#endif
#ifdef __unix__
# include <fcntl.h>
#endif /* __unix__ */

/* -------------------------------------- */

static int __arg2cname(const char *arg,int def_on_fail)
{
  if (arg)
  {
    char buf[32];
    unsigned int i=0;
    while (i<(sizeof(buf)-1) && *arg)
      buf[i++] = (char)toupper((char)(*arg++));
    if (i)
    {
      buf[i]='\0';
      for (i=0;i<CONTEST_COUNT;i++)
      {
        arg = CliGetContestNameFromID(i);
        if (arg)
        {
          if (strcmp(arg,buf)==0)
            return (int)i;
        }
      }
    }
  }
  return def_on_fail;
}

/* -------------------------------------- */

int ParseCommandline( Client *client,
                      int run_level, int argc, const char *argv[],
                      int *retcodeP, int logging_is_initialized )
{
  int inimissing = 0;
  int terminate_app = 0, havemode = 0;
  int pos, skip_next;
  const char *thisarg, *argvalue;

  TRACE_OUT((+1,"ParseCommandline(%d,%d)\n",run_level,argc));

  //-----------------------------------
  // In the first loop we (a) get the ini filename and
  // (b) get switches that won't be overriden by the ini
  //-----------------------------------

  TRACE_OUT((+1,"ParseCommandline(P1)\n"));

  if (!terminate_app && run_level == 0)
  {
    client->inifilename[0] = 0; //so we know when it changes
    ModeReqClear(-1);   // clear all mode request bits
    int loop0_quiet = 0;

    skip_next = 0;
    for (pos = 1; !terminate_app && pos < argc; pos += (1+skip_next))
    {
      int not_supported = 0;
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      argvalue = ((pos < (argc-1))?(argv[pos+1]):((char *)NULL));
      skip_next = 0;

      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
      else if ( strcmp( thisarg, "-genman" ) == 0)
      {
        extern void GenerateManPage( void );
        GenerateManPage();
        terminate_app = 1;
      }
      else if ( strcmp( thisarg, "-hide" ) == 0 ||
                strcmp( thisarg, "-quiet" ) == 0 )
        loop0_quiet = 1; //used for stuff in this loop
      else if ( strcmp( thisarg, "-noquiet" ) == 0 )
        loop0_quiet = 0; //used for stuff in this loop
      else if ( strcmp(thisarg, "-ini" ) == 0)
      {
        if (argvalue)
        {
          skip_next = 1;
          strcpy( client->inifilename, argvalue );
        }
        else
          terminate_app = 1;
      }
      else if ( ( strcmp( thisarg, "-restart" ) == 0) ||
                ( strcmp( thisarg, "-hup" ) == 0 ) ||
                ( strcmp( thisarg, "-kill" ) == 0 ) ||
                ( strcmp( thisarg, "-shutdown" ) == 0 ) ||
                ( strcmp( thisarg, "-pause" ) == 0 ) ||
                ( strcmp( thisarg, "-unpause" ) == 0 ) )
      {
        #if (CLIENT_OS == OS_NETWARE)
        {
          if (!loop0_quiet)
          {
            const char *appname = nwCliGetNLMBaseName();
            ConsolePrintf("%s: %s cannot be used as a load time option.\r\n"
            "\tPlease use '%s [-quiet] %s'\r\n"
            "\tinstead of 'LOAD %s [-quiet] %s'\r\n"
            "\t(note: this is only available when a client is running)\r\n",
                     appname, thisarg, appname, thisarg, appname, thisarg );
          }
          terminate_app = 1;
        }
        #elif defined(__unix__) && !defined(__EMX__) && (CLIENT_OS != OS_NEXTSTEP)
        {
          char buffer[1024];
          int sig = SIGHUP; char *dowhat_descrip = "-HUP'ed";
          unsigned int bin_index, kill_ok = 0, kill_failed = 0;
          int last_errno = 0, kill_found = 0;
          const char *binnames[3];
          char rc5des[8]; rc5des[0]='r';rc5des[1]='c';rc5des[2]='5';
          rc5des[3]='d';rc5des[4]='e';rc5des[5]='s';rc5des[6]='\0';
          binnames[0] = (const char *)strrchr( argv[0], '/' );
          binnames[0] = ((!binnames[0])?(argv[0]):(binnames[0]+1));
          binnames[1] = utilGetAppName();
          binnames[2] = rc5des;

          if ( strcmp( thisarg, "-kill" ) == 0 ||
               strcmp( thisarg, "-shutdown") == 0 )
          { sig = SIGTERM; dowhat_descrip = "shutdown"; }
          else if (strcmp( thisarg, "-pause" ) == 0)
          { sig = TRIGGER_PAUSE_SIGNAL; dowhat_descrip = "paused";  }
          else if (strcmp( thisarg, "-unpause" ) == 0)
          { sig = TRIGGER_UNPAUSE_SIGNAL; dowhat_descrip = "unpaused"; }

          pid_t already_sigd[128]; unsigned int sigd_count = 0;
          #if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
              (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD)
          DIR *dirp = opendir("/proc");
          if (!dirp)
            kill_found = -1;
          else
          {
            struct dirent *dp;
            pid_t ourpid = getpid();
            char realbinname[64];
            size_t len; FILE *file = fopen("/proc/curproc/cmdline","r");
            if (file)
            {
              /* useless for OSs that set argv[0] in client.cpp */
              len = fread( buffer, 1, sizeof(buffer), file );
              fclose( file );
              if (len!=0)
              {
                char *p, *q=&buffer[0];
                if (len == sizeof(buffer))
                  len--;
                buffer[len] = '\0';
                while (*q && isspace(*q))
                  q++;
                p = q;
                while (*q && !isspace(*q))
                {
                  if (*q=='/')
                    p = q+1;
                  q++;
                }
                *q = '\0';
                strncpy(realbinname,p,sizeof(realbinname));
                realbinname[sizeof(realbinname)-1]='\0';
                binnames[0] = (const char *)&realbinname[0];
              }
            }
            while ((dp = readdir(dirp)) != ((struct dirent *)0))
            {
              pid_t thatpid = (pid_t)atoi(dp->d_name);
              if (thatpid == 0 /* .,..,curproc,etc */ || thatpid == ourpid)
                continue;
              sprintf( buffer, "/proc/%s/cmdline", dp->d_name );
              if (( file = fopen( buffer, "r" ) ) == ((FILE *)0))
                continue; /* already died */
              len = fread( buffer, 1, sizeof(buffer), file );
              fclose( file );
              if (len != 0)
              {
                char *q, *procname = &buffer[0];
                if (len == sizeof(buffer))
                  len--;
                buffer[len] = '\0';
                //printf("%s: %60s\n", dp->d_name, buffer );
                if (memcmp(buffer,"Name:",5)==0) /* linux status*/
                  procname+=5;
                while (*procname && isspace(*procname))
                  procname++;
                q = procname;
                while (*q && !isspace(*q))
                {
                  if (*q =='/')
                    procname = q+1;
                  q++;
                }
                *q = '\0';
                //printf("%s: %s (binname0:%s,binname1:%s)\n",dp->d_name,procname,binname[0],binname[1]);
                for (bin_index=0;
                     bin_index<(sizeof(binnames)/sizeof(binnames[0]));
                     bin_index++)
                {
                  if (strcmp(procname,binnames[bin_index])==0)
                  {
                    kill_found++;
                    if ( kill( thatpid, sig ) == 0)
                    {
                      if (sigd_count < (sizeof(already_sigd)/sizeof(pid_t)-1))
                        already_sigd[sigd_count++] = thatpid;
                      kill_ok++;
                    }
                    else if ((errno != ESRCH) && (errno != ENOENT))
                    {
                      kill_failed++;
                      last_errno = errno;
                    }
                    break;
                  }
                }
              }
            }
            closedir(dirp);
          }
          #elif (CLIENT_OS == OS_HPUX)
          {
            pid_t ourpid = getpid();
            struct pst_status pst[10];
            int count, idx = 0; /* index within the context */
            kill_found = -1; /* assume all failed */

            /* loop until count == 0, will occur all have been returned */
            while ((count = pstat_getproc(pst, sizeof(pst[0]),
                          (sizeof(pst)/sizeof(pst[0])), idx)) > 0)
            {
              int pspos;
              if (kill_found < 0)
                kill_found = 0;
              for (pspos=0; pspos < count; pspos++)
              {
                //printf("pid: %d, cmd: %s\n",pst[pspos].pst_pid,pst[pspos].pst_ucomm);
                char *procname = (char *)pst[pspos].pst_ucomm;
                pid_t thatpid = (pid_t)pst[pspos].pst_pid;
                if (thatpid != ourpid)
                {
                  for (bin_index=0;
                       bin_index<(sizeof(binnames)/sizeof(binnames[0]));
                       bin_index++)
                  {
                    if (strcmp(procname,binnames[bin_index])==0)
                    {
                      kill_found++;
                      if ( kill( thatpid, sig ) == 0)
                        kill_ok++;
                      else if ((errno != ESRCH) && (errno != EINVAL))
                      {
                        kill_failed++;
                        last_errno = errno;
                      }
                      break;
                    }
                  }
                }
              }
              idx = pst[count-1].pst_idx + 1;
            }
          }
          #endif
          #if (CLIENT_OS != OS_LINUX) && (CLIENT_OS != OS_HPUX)
          // this part is only needed for OSs that do not read /proc OR
          // do not have a reliable method to set the name as read from /proc
          // (as opposed to reading it from ps output)
          const char *pscmd = NULL;
          #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
              (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_LINUX) || \
              (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_MACOSX)
          pscmd = "ps ax|awk '{print$1\" \"$5}' 2>/dev/null"; /* bsd, no -o */
          //fbsd: "ps ax -o pid -o command 2>/dev/null";  /* bsd + -o ext */
          //lnux: "ps ax --format pid,comm 2>/dev/null";  /* bsd + gnu -o */
          #elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || \
                (CLIENT_OS == OS_DEC_UNIX) || (CLIENT_OS == OS_AIX)
          pscmd = "/usr/bin/ps -ef -o pid -o comm 2>/dev/null"; /*svr4/posix*/
          #elif (CLIENT_OS == OS_IRIX) || (CLIENT_OS == OS_HPUX)
          pscmd = "/usr/bin/ps -e |awk '{print$1\" \"$4\" \"$5\" \"$6\" \"$7\" \"$8\" \"$9}' 2>/dev/null";
          #elif (CLIENT_OS == OS_BEOS)
          pscmd = "/bin/ps | /bin/egrep zzz | /bin/egrep -v crunch 2>/dev/null";  /* get the (sleeping) main thread ID, not the team ID */
          #elif (CLIENT_OS == OS_QNX)
          pscmd = "ps -A -F"%p %c" 2>/dev/null";
          #elif (CLIENT_OS == OS_NTO2)
          pscmd = "ps -A -o pid,comm 2>/dev/null";
          #else
          #error fixme: select an appropriate ps syntax
          #endif
          FILE *file = (pscmd ? popen( pscmd, "r" ) : ((FILE *)NULL));
          if (file == ((FILE *)NULL))
          {
            if (kill_found == 0) /* /proc read also failed/wasn't done? */
              kill_found = -1; /* spawn failed */
          }
          else
          {
            pid_t ourpid = getpid();
            unsigned int linelen = 0;
            int got_output = 0, eof_count = 0;
            //ConOutModal(pscmd);
            while (file) /* dummy while */
            {
              int ch;
              if (( ch = fgetc( file ) ) == EOF )
              {
                if (ferror(file))
                  break;
                if (linelen == 0)
                {
                  if ((++eof_count) > 2)
                    break;
                }
                usleep(250000);
              }
              else if (ch == '\n')
              {
                eof_count = 0;
                if (linelen == 0)
                  continue;
                if (linelen < sizeof(buffer)-1) /* otherwise, line too long */
                {
                  pid_t thatpid;
                  char *q, *procname = &buffer[0];
                  buffer[linelen]='\0';
                  while (*procname && isspace(*procname))
                    procname++;
                  thatpid = (pid_t)atol(procname);
                  if (thatpid == ourpid)  /* ignore it */
                  {
                    got_output = 1;
                    //printf("'%s' ** THIS IS US ** \n",buffer,thatpid);
                    thatpid = 0;
                  }
                  else if (thatpid != 0)
                  {
                    got_output = 1;
                    if (sigd_count != 0)
                    {
                      unsigned int pid_pos;
                      for (pid_pos=0;pid_pos<sigd_count;pid_pos++)
                      {
                        if (already_sigd[pid_pos] == thatpid)
                        {
                          thatpid = 0;
                          break;
                        }
                      }
                    }
                  }
                  if (thatpid != 0)
                  {
                    while (*procname && (isdigit(*procname) || isspace(*procname)))
                      procname++;
                    q = procname;
                    while (*q && !isspace(*q))
                    {
                      if (*q == '/')
                        procname = q+1;
                      q++;
                    }
                    *q = '\0';
                    //printf("pid='%d' procname='%s'\n",thatpid,procname);

                    for (bin_index=0;
                         bin_index<(sizeof(binnames)/sizeof(binnames[0]));
                         bin_index++)
                    {
                      if (strcmp(procname,binnames[bin_index])==0)
                      {
                        kill_found++;
                        if ( kill( thatpid, sig ) == 0)
                          kill_ok++;
                        else if ((errno != ESRCH) && (errno != ENOENT))
                        {
                          kill_failed++;
                          last_errno = errno;
                        }
                        break;
                      }
                    }
                  } /* not ourselves and not already done */
                } /* if (linelen < sizeof(buffer)-1) */
                linelen = 0; /* prepare for next line */
              } /* if (ch == '\n') */
              else
              {
                eof_count = 0;
                if (linelen < sizeof(buffer)-1)
                  buffer[linelen++] = ch;
              }
            } /* while (file) */
            if (!got_output && kill_found == 0)
              kill_found = -1;
            pclose(file);
          }
          #endif /* either read /proc/ or spawn ps */

          if (!loop0_quiet && kill_found >= -1)
          {
            if (kill_found == -1)
              sprintf( buffer, "%s failed. Unable to get pid list", thisarg );
            else if (kill_found == 0)
              sprintf(buffer,"No distributed.net clients were found. "
                             "None %s.", dowhat_descrip );
            else
              sprintf(buffer,"%u distributed.net client%s %s. %u failure%s%s%s%s.",
                       kill_ok,
                       ((kill_ok==1)?(" was"):("s were")),
                       dowhat_descrip,
                       kill_failed, (kill_failed==1)?(""):("s"),
                       ((kill_failed==0)?(""):(" (")),
                       ((kill_failed==0)?(""):(strerror(last_errno))),
                       ((kill_failed==0)?(""):(")")) );
            ConOutErr(buffer);
          }
          terminate_app = 1;
        }
        #elif (CLIENT_OS == OS_WIN16 || CLIENT_OS == OS_WIN32)
        {
          int rc, cmd = DNETC_WCMD_RESTART;
          const char *dowhat_descrip = "restarted";

          if ( strcmp( thisarg, "-kill" ) == 0 ||
               strcmp( thisarg, "-shutdown") == 0 )
          {
            cmd = DNETC_WCMD_SHUTDOWN;
            thisarg = "-shutdown";
            dowhat_descrip = "shutdown";
          }
          else if (strcmp( thisarg, "-pause" ) == 0)
          {
            cmd = DNETC_WCMD_PAUSE;
            dowhat_descrip = "paused";
          }
          else if (strcmp( thisarg, "-unpause" ) == 0)
          {
            cmd = DNETC_WCMD_UNPAUSE;
            dowhat_descrip = "unpaused";
          }

          rc = w32PostRemoteWCMD(cmd); /*<0=notfound,0=found+ok,>0=found+err*/
          terminate_app = 1;
          if (!loop0_quiet)
          {
            char scratch[128];
            if (rc < 0)
              sprintf(scratch,"No distributed.net clients are currently running. "
                              "None were %s.", dowhat_descrip);
            else if (rc > 0)
              sprintf(scratch,"One or more distributed.net clients were found "
                              "but one or more could not be %s.\n", dowhat_descrip);
            else
              sprintf(scratch,"One or more distributed.net clients were found "
                              "and have been requested to %s.", thisarg+1 );
            ConOutModal(scratch);
          }
        }
        #else
          not_supported = 1;
        #endif
      }
      else if ( strcmp(thisarg, "-install" ) == 0)
      {
        #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16)
        win32CliInstallService(loop0_quiet); /*w32svc.cpp*/
        terminate_app = 1;
        #elif (CLIENT_OS == OS_OS2)
        extern int os2CliInstallClient(int quiet, const char *exename);
        os2CliInstallClient(loop0_quiet, argv[0]); /* os2inst.cpp */
        terminate_app = 1;
        #else
        not_supported = 1;
        #endif
      }
      else if (strcmp(thisarg,"-svcstart") == 0)
      {
        #if (CLIENT_OS == OS_WIN32)
        terminate_app = 1;
        if (!loop0_quiet)
        {
          int isinst = win32CliIsServiceInstalled();/*<0=err,0=no,>0=yes */
          if (isinst < 0)
            ConOutErr("Service manager error. Service could not be started.\n");
          else if (isinst == 0)
            ConOutErr("Cannot start a service that is not -installed.\n");
          else
            terminate_app = 0;
        }
        if (!terminate_app) /* no error */
        {
          char *xargv[2]; xargv[0] = (char *)argv[0]; xargv[1]=NULL;
          win32CliStartService( 1, &xargv[0] ); /* *installed* client */
          terminate_app = 1;
        }
        #else
        not_supported = 1;
        #endif
      }
      else if (!strcmp(thisarg,"-uninstall") || !strcmp(thisarg, "-deinstall"))
      {
        #if (CLIENT_OS == OS_OS2)
        extern int os2CliUninstallClient(int /*do it without feedback*/);
        os2CliUninstallClient(loop0_quiet); /* os2inst.cpp */
        terminate_app = 1;
        #elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16)
        win32CliUninstallService(loop0_quiet); /*w32svc.cpp*/
        terminate_app = 1;
        #else
        not_supported = 1;
        #endif
      }
      if (not_supported)
      {
        char scratch[80];
        sprintf(scratch,"%s is not supported for this platform.\n",thisarg);
        ConOutErr(scratch);
        terminate_app = 1;
      }
    }
  }

  TRACE_OUT((-1,"ParseCommandline(P1)\n"));

  //-----------------------------------
  // In the next section we get inifilename defaults
  // and load the config from file
  //-----------------------------------

  TRACE_OUT((+1,"ParseCommandline(P2)\n"));

  if (!terminate_app && run_level == 0)
  {
    if (client->inifilename[0]==0) // determine the filename of the ini file
    {
      char * inienvp = getenv( "RC5INI" );
      if ((inienvp != NULL) && (strlen( inienvp ) < sizeof(client->inifilename)))
        strcpy( client->inifilename, inienvp );
      else
      {
        #if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_DOS) || \
            (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_OS2) || \
            (CLIENT_OS == OS_WIN32)
        //not really needed for netware (appname in argv[0] won't be anything
        //except what I tell it to be at link time.)
        client->inifilename[0] = 0;
        if (argv[0]!=NULL && ((strlen(argv[0])+5) < sizeof(client->inifilename)))
        {
          strcpy( client->inifilename, argv[0] );
          char *slash = strrchr( client->inifilename, '/' );
          char *slash2 = strrchr( client->inifilename, '\\');
          if (slash2 > slash ) slash = slash2;
          slash2 = strrchr( client->inifilename, ':' );
          if (slash2 > slash ) slash = slash2;
          if ( slash == NULL ) slash = client->inifilename;
          if ( ( slash2 = strrchr( slash, '.' ) ) != NULL ) // ie > slash
            strcpy( slash2, ".ini" );
          else if ( strlen( slash ) > 0 )
           strcat( slash, ".ini" );
        }
        if ( client->inifilename[0] == 0 )
          strcat( strcpy( client->inifilename, utilGetAppName() ), ".ini" );
        #elif (CLIENT_OS == OS_VMS)
          strcat( strcpy( client->inifilename, utilGetAppName() ), EXTN_SEP "ini" );
        #else
        strcpy( client->inifilename, argv[0] );
        strcat( client->inifilename, EXTN_SEP "ini" );
        #endif
      }
    } // if (inifilename[0]==0)

    TRACE_OUT((+1,"InitWorkingDirectoryFromSamplePaths()\n"));
    InitWorkingDirectoryFromSamplePaths( client->inifilename, argv[0] );
    TRACE_OUT((-1,"InitWorkingDirectoryFromSamplePaths()\n"));

    if ( (pos = ReadConfig(client)) != 0)
    {
      if (pos < 0) /* fatal */
        terminate_app = 1;
      else
      {
        //client->stopiniio = 1; /* client class */
        //ModeReqSet( MODEREQ_CONFIG );
        inimissing = 1;
      }
    }
  }

  TRACE_OUT((-1,"ParseCommandline(P2,%s)\n", client->inifilename));

  //-----------------------------------
  // In the next loop we parse the other options
  //-----------------------------------

  TRACE_OUT((+1,"ParseCommandline(P3)\n"));

  if (!terminate_app && ((run_level == 0) || (logging_is_initialized)))
  {
    for (pos = 1; pos < argc; pos += (1+skip_next))
    {
      int missing_value = 0;
      int invalid_value = 0;
      skip_next = 0;
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      argvalue = ((pos < (argc-1))?(argv[pos+1]):((char *)NULL));

      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
      else if ( strcmp( thisarg, "-c" ) == 0 ||
                strcmp( thisarg, "-blsize" ) == 0 ||
                strcmp( thisarg, "-b" ) == 0 ||
                strcmp( thisarg, "-b2" ) == 0 ||
                strcmp( thisarg, "-bin" ) == 0 ||
                //#if !defined(NO_OUTBUFFER_THRESHOLDS)
                strcmp( thisarg, "-bout" ) == 0 || /* no effect */
                strcmp( thisarg, "-bout2") == 0 || /* no effect */
                //#endif
                strcmp( thisarg, "-bin2")==0 ||
                strcmp( thisarg, "-btime") == 0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          // isthresh: 1 = in, 2 = out, 4 = time_in
          int n, maxval = 0, minval = 0, defval = 0, confoption = -1, isthresh = 0, isblsize = 0;
          int contest_defaulted = 0;
          unsigned int contest;
          const char *op;

          if (strcmp(thisarg,"-blsize")==0)
            isblsize = 1;
          else if (strcmp(thisarg,"-bin")==0 || strcmp( thisarg, "-bin2")==0)
            isthresh = 1;
          else if (strcmp(thisarg,"-bout")==0 || strcmp(thisarg,"-bout2")==0)
            isthresh = 2;
          else if (strcmp( thisarg, "-b" )==0 || strcmp( thisarg, "-b2" )==0)
            isthresh = 1+2;
          else if (strcmp( thisarg, "-btime" ) == 0)
            isthresh = 4;

          skip_next = 1;
          op = argvalue;
          contest = (unsigned int)__arg2cname(argvalue,CONTEST_COUNT);
          if (contest < CONTEST_COUNT)
          {
            skip_next = 2;
            op = ((pos < (argc-2))?(argv[pos+2]):((char *)NULL));
          }
          else
          {
            contest = RC5;
            if (strcmp( thisarg, "-bin2")==0 ||
                strcmp( thisarg, "-bout2")==0 ||
                strcmp( thisarg, "-b2")==0)
              contest = DES;
            //else if (isblsize)       //-blsize without contest means both
            //  contest_defaulted = 1; //RC5 and DES
          }

          n = -123;
          if (op != NULL)
          {
            n = atoi(op);
            if (n == 0 && !isdigit(*op))
              n = -123;
          }

          // get default values and ranges from conf_options[]
          if (isblsize)
          {
            confoption = CONF_PREFERREDBLOCKSIZE;
            if (contest == OGR) /* invalid for ogr */
              n = -123;
          }
          else if (isthresh & (1+2))
          {
            confoption = CONF_THRESHOLDI;
          }
          else if (isthresh & 4)
          {
            confoption = CONF_THRESHOLDT;
            if (contest == OGR) /* time threshold invalid for ogr */
              n = -123;
          }
          else if (isthresh)
            missing_value = 1; // uups ?
          else /* coretype */
          {
            confoption = -1;
            if ((n != -1) && (n != selcoreValidateCoreIndex(contest, n)))
              invalid_value = 1;
          }

          if (confoption >= 0)
          {
            minval = conf_options[confoption].choicemin;
            maxval = conf_options[confoption].choicemax;
            defval = atoi(conf_options[confoption].defaultsetting);
          }
          else
            minval = maxval = defval = 0;

          if ((n != defval) && (minval || maxval) && (n < minval || n > maxval))
            invalid_value = 1;
          else if (run_level == 0)
          {
            inimissing = 0; // Don't complain if the inifile is missing
            if (isblsize)
            {
              if (n == -1)
                n = 0; // default
              client->preferred_blocksize[contest] = n;
              if (contest_defaulted)
                client->preferred_blocksize[DES] = n;
            }
            else if (isthresh)
            {
              if ((isthresh & 1)!=0)
              {
                client->inthreshold[contest] = n;
                /* {-b,-bin} <pn> <n> overrides time threshold,
                   user may add -btime <pn> <n> if needed */
                client->timethreshold[contest] = 0;
              }
              #if !defined(NO_OUTBUFFER_THRESHOLDS)
              if ((isthresh & 2)!=0)
                client->outthreshold[contest] = n;
              #endif  
              if ((isthresh & 4)!=0)
                client->timethreshold[contest] = n;
            }
            else /* coretype */
            {
              client->coretypes[contest] = n;
            }
          }
          else if (logging_is_initialized)
          {
            if (isblsize)
            {
              LogScreenRaw("%s preferred packet size set to 2^%d\n",
                  CliGetContestNameFromID(contest),
                  client->preferred_blocksize[contest] );
              if (contest_defaulted)
                LogScreenRaw("DES preferred packet size set to 2^%d\n",
                  client->preferred_blocksize[DES] );
            }
            else if (isthresh)
            {
              if ((isthresh & 1)!=0)
              {
                LogScreenRaw("%s fetch threshold set to %d work unit%s\n",
                  CliGetContestNameFromID(contest),
                  client->inthreshold[contest], (client->inthreshold[contest]==1)?"":"s" );
                if (contest != OGR)
                  LogScreenRaw("%s fetch time threshold cleared\n",
                    CliGetContestNameFromID(contest) );
              }
              #if !defined(NO_OUTBUFFER_THRESHOLDS)
              if ((isthresh & 2)!=0)
                LogScreenRaw("%s flush threshold set to %d work unit%s\n",
                  CliGetContestNameFromID(contest),
                  client->outthreshold[contest], (client->outthreshold[contest]==1)?"":"s" );
              #endif                  
              if ((isthresh & 4)!=0)
                LogScreenRaw("%s fetch time threshold set to %d hour%s\n",
                  CliGetContestNameFromID(contest),
                  client->timethreshold[contest], (client->timethreshold[contest]==1)?"":"s" );
            }
            else /* coretype */
            {
              LogScreenRaw("Default core for %s set to #%d\n",
                  CliGetContestNameFromID(contest),
                  client->coretypes[contest] );
            }
          }
        }
      }
      else if ( strcmp( thisarg, "-ini" ) == 0)
      {
        //we already did this so skip it
        if (argvalue)
          skip_next = 1;
        else
          missing_value = 1;
      }
      else if ( strcmp( thisarg, "-guiriscos" ) == 0)
      {
        #if (CLIENT_OS == OS_RISCOS)
        if (run_level == 0)
          guiriscos = 1;
        #endif
      }
      else if ( strcmp( thisarg, "-guirestart" ) == 0)
      {          // See if are restarting (hence less banners wanted)
        #if (CLIENT_OS == OS_RISCOS)
        if (run_level == 0)
          guirestart = 1;
        #endif
      }
      else if ( strcmp( thisarg, "-multiok" ) == 0 ) /* keep undocumented! */
      {
        /* allow multiple instances - keep this undocumented */
        #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || \
            (CLIENT_OS == OS_OS2)
        putenv("dnetc_multiok=1");
        #endif
      }
      else if ( strcmp( thisarg, "-hide" ) == 0 ||
                strcmp( thisarg, "-quiet" ) == 0 )
      {
        if (run_level == 0)
          client->quietmode = 1;
      }
      else if ( strcmp( thisarg, "-noquiet" ) == 0 )
      {
        if (run_level == 0)
          client->quietmode = 0;
      }
      else if ( strcmp(thisarg, "-percentoff" ) == 0)
      {
        if (run_level == 0)
          client->percentprintingoff = 1;
      }
      else if ( strcmp( thisarg, "-nofallback" ) == 0 )
      {
        if (run_level == 0)
          client->nofallback = 1;
      }
      else if ( strcmp( thisarg, "-lurk" ) == 0 )
      {                           // Detect modem connections
        #if defined(LURK)
        if (run_level == 0)
          client->lurk_conf.lurkmode=CONNECT_LURK;
        #endif
      }
      else if ( strcmp( thisarg, "-lurkonly" ) == 0 )
      {                           // Only connect when modem connects
        #if defined(LURK)
        if (run_level == 0)
          client->lurk_conf.lurkmode=CONNECT_LURKONLY;
        #endif
      }
      else if ( strcmp( thisarg, "-interfaces" ) == 0 )
      {
        if (!argvalue)
        {
          missing_value = 1;
        }
        else
        {
          skip_next = 1;
          #if defined(LURK)
          if (run_level!=0)
          {
            if (logging_is_initialized)
              LogScreenRaw ("Limited interface watch list to %s\n",
                             client->lurk_conf.connifacemask );
          }
          else
          {
            strncpy(client->lurk_conf.connifacemask, argvalue,
                       sizeof(client->lurk_conf.connifacemask) );
            client->lurk_conf.connifacemask[sizeof(client->lurk_conf.connifacemask)-1] = 0;
          }
          #endif
        }
      }
      else if ( strcmp( thisarg, "-noexitfilecheck" ) == 0 )
      {
        if (run_level == 0)
          client->exitflagfile[0]='\0';
      }
      else if ( strcmp( thisarg, "-runoffline" ) == 0 ||
                strcmp( thisarg, "-runonline" ) == 0)
      {
        if (run_level != 0)
        {
          if (logging_is_initialized)
            LogScreenRaw("Client will run with%s network access.\n",
                       ((client->offlinemode)?("out"):("")) );
        }
        else
          client->offlinemode = ((strcmp( thisarg, "-runoffline" ) == 0)?(1):(0));
      }
      else if (strcmp(thisarg,"-runbuffers")==0 || strcmp(thisarg,"-run")==0)
      {
        if (run_level != 0)
        {
          if (logging_is_initialized)
          {
            LogScreenRaw("Warning: %s is obsolete.\n"
                         "         Active settings: -runo%sline and -n %d%s.\n",
              thisarg, ((client->offlinemode)?("ff"):("n")),
              ((client->blockcount<0)?(-1):((int)client->blockcount)),
              ((client->blockcount<0)?(" (exit on empty buffers)"):("")) );
          }
        }
        else
        {
          if (strcmp(thisarg,"-run")==0)
          {
            client->offlinemode = 0;
            if (client->blockcount < 0)
              client->blockcount = 0;
          }
          else /* -runbuffers */
          {
            client->offlinemode = 1;
            client->blockcount = -1;
          }
        }
      }
      else if ( strcmp( thisarg, "-nodisk" ) == 0 )
      {
        if (run_level == 0)
          client->nodiskbuffers=1;              // No disk buff-*.rc5 files.
        inimissing = 0; // Don't complain if the inifile is missing
      }
      else if ( strcmp(thisarg, "-frequent" ) == 0)
      {
        if (run_level!=0)
        {
          if (logging_is_initialized && client->connectoften)
            LogScreenRaw("Buffer thresholds will be checked frequently.\n");
        }
        else
          client->connectoften = 1;
      }
      else if ( strcmp( thisarg, "-inbase" ) == 0 || strcmp( thisarg, "-outbase")==0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          int out = ( strcmp( thisarg, "-outbase" ) == 0 );
          char *p = (out ? client->out_buffer_basename : client->in_buffer_basename);
          skip_next = 1;
          if ( run_level == 0 )
          {
            strncpy( p, argvalue, sizeof(client->in_buffer_basename) );
            p[sizeof(client->in_buffer_basename)-1]=0;
            inimissing = 0; // Don't complain if the inifile is missing
          }
          else if (logging_is_initialized)
          {
            LogScreenRaw("Setting %s-buffer base name to %s\n",
              (out ? "out" : "in"), p );
          }
        }
      }
      else if ( strcmp( thisarg, "-u" ) == 0 ) // UUE/HTTP Mode
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting uue/http mode to %u\n",
              (unsigned int)client->uuehttpmode);
          }
          else
          {
            client->uuehttpmode = atoi( argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-a" ) == 0 ) // Override the keyserver name
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting keyserver to %s\n", client->keyproxy );
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            client->autofindkeyserver = 0;
            strcpy( client->keyproxy, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-p" ) == 0 ) // UUE/HTTP Mode
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level!=0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting keyserver port to %u\n",
              (unsigned int)client->keyport);
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            client->keyport = atoi( argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-ha" ) == 0 ) // Override the http proxy name
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting SOCKS/HTTP proxy to %s\n",
              client->httpproxy);
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            strcpy( client->httpproxy, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-hp" ) == 0 ) // Override the socks/http proxy port
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting SOCKS/HTTP proxy port to %u\n",
              (unsigned int)client->httpport);
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            client->httpport = atoi( argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-l" ) == 0 ) // Override the log file name
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting log file to %s\n", client->logname );
          }
          else
          {
            strcpy( client->logname, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-smtplen" ) == 0 ) // Override the mail message length
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting Mail message length to %u\n",
              (unsigned int)client->messagelen );
          }
          else
          {
            client->messagelen = atoi(argvalue);
          }
        }
      }
      else if ( strcmp( thisarg, "-smtpport" ) == 0 ) // Override the smtp port for mailing
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting smtp port to %u\n",
              (unsigned int)client->smtpport);
          }
          else
          {
            client->smtpport = atoi(argvalue);
          }
        }
      }
      else if ( strcmp( thisarg, "-smtpsrvr" ) == 0 ) // Override the smtp server name
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting SMTP relay host to %s\n", client->smtpsrvr);
          }
          else
          {
            strcpy( client->smtpsrvr, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-smtpfrom" ) == 0 ) // Override the smtp source id
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting mail 'from' address to %s\n",
              client->smtpfrom );
          }
          else
            strcpy( client->smtpfrom, argvalue );
        }
      }
      else if ( strcmp( thisarg, "-smtpdest" ) == 0 ) // Override the smtp source id
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting mail 'to' address to %s\n", client->smtpdest );
          }
          else
          {
            strcpy( client->smtpdest, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-e" ) == 0 )     // Override the email id
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting distributed.net ID to %s\n", client->id );
          }
          else
          {
            strcpy( client->id, argvalue );
            inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      }
      else if ( strcmp( thisarg, "-nettimeout" ) == 0 ) // Change network timeout
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting network timeout to %u\n",
                                     (unsigned int)(client->nettimeout));
          }
          else
          {
            client->nettimeout = atoi(argvalue);
          }
        }
      }
      else if ( strcmp( thisarg, "-exitfilechecktime" ) == 0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          /* obsolete */
        }
      }
      else if ( strcmp( thisarg, "-nice" ) == 0
             || strcmp( thisarg, "-priority" ) == 0 ) // Nice level
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting priority to %u\n", client->priority );
          }
          else
          {
            client->priority = atoi( argvalue );
            if ( strcmp( thisarg, "-nice" ) == 0 )
              client->priority = ((client->priority==2)?(8):
                                 ((client->priority==1)?(4):(0)));
            inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      }
      else if ( strcmp( thisarg, "-h" ) == 0 || strcmp( thisarg, "-until" ) == 0)
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          int isuntil = (strcmp( thisarg, "-until" ) == 0);
          int h=0, m=0, pos, isok = 0, dotpos=0;
          if (isdigit(*argvalue))
          {
            isok = 1;
            for (pos = 0; argvalue[pos] != 0; pos++)
            {
              if (!isdigit(argvalue[pos]))
              {
                if (dotpos != 0 || (argvalue[pos] != ':' && argvalue[pos] != '.'))
                {
                  isok = 0;
                  break;
                }
                dotpos = pos;
              }
            }
            if (isok)
            {
              if ((h = atoi( argvalue )) < 0)
                isok = 0;
              else if (isuntil && h > 23)
                isok = 0;
              else if (dotpos == 0 && isuntil)
                isok = 0;
              else if (dotpos != 0 && strlen(argvalue+dotpos+1) != 2)
                isok = 0;
              else if (dotpos != 0 && ((m = atoi(argvalue+dotpos+1)) > 59))
                isok = 0;
            }
          }
          if (run_level != 0)
          {
            if (logging_is_initialized)
            {
              if (!isok)
                LogScreenRaw("%s option is invalid. Was it in hh:mm format?\n",thisarg);
              else if (client->minutes == 0)
                LogScreenRaw("Setting time limit to zero (no limit).\n");
              else
              {
                struct timeval tv; CliTimer(&tv);
                tv.tv_sec+=(time_t)(client->minutes*60);
                LogScreenRaw("Setting time limit to %u:%02u hours (stops at %s)\n",
                             client->minutes/60, client->minutes%60, CliGetTimeString(&tv,1) );
              }
            }
          }
          else if (isok)
          {
            client->minutes = ((h*60)+m);
            if (isuntil)
            {
              time_t timenow = CliTimer(NULL)->tv_sec;
              struct tm *ltm = localtime( &timenow );
              if (ltm->tm_hour > h || (ltm->tm_hour == h && ltm->tm_min >= m))
                client->minutes+=(24*60);
              client->minutes -= (((ltm->tm_hour)*60)+(ltm->tm_min));
            }
          }
        }
      }
      else if ( strcmp( thisarg, "-n" ) == 0 ) // Blocks to complete in a run
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
            {
              if (client->blockcount < 0)
                LogScreenRaw("Client will exit when buffers are empty.\n");
              else
                LogScreenRaw("Setting block completion limit to %u%s\n",
                    (unsigned int)client->blockcount,
                    ((client->blockcount==0)?(" (no limit)"):("")));
            }
          }
          else if ( (client->blockcount = atoi( argvalue )) < 0)
            client->blockcount = -1;
        }
      }
      else if ( strcmp( thisarg, "-numcpu" ) == 0 ) // Override the number of cpus
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
            ;
          else
          {
            client->numcpu = atoi(argvalue);
            inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      }
      else if ( strcmp( thisarg, "-ckpoint" ) == 0 || strcmp( thisarg, "-ckpoint2" ) == 0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting checkpoint file to %s\n",
                                                 client->checkpoint_file );
          }
          else
          {
            strcpy(client->checkpoint_file, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-cktime" ) == 0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          /* obsolete */
        }
      }
      else if ( strcmp( thisarg, "-pausefile" ) == 0)
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting pause file to %s\n",client->pausefile);
          }
          else
          {
            strcpy(client->pausefile, argvalue );
          }
        }
      }
      else if (( strcmp( thisarg, "-fetch"  ) == 0 ) ||
          ( strcmp( thisarg, "-forcefetch"  ) == 0 ) ||
          ( strcmp( thisarg, "-flush"       ) == 0 ) ||
          ( strcmp( thisarg, "-forceflush"  ) == 0 ) ||
          ( strcmp( thisarg, "-update"      ) == 0 ) ||
          ( strcmp( thisarg, "-ident"       ) == 0 ) ||
          ( strcmp( thisarg, "-cpuinfo"     ) == 0 ) ||
          ( strcmp( thisarg, "-config"      ) == 0 ) )
      {
        havemode = 1; //nothing - handled in next loop
      }
      else if ( strcmp( thisarg, "-benchmark"   ) == 0  ||
                strcmp( thisarg, "-benchmark2"  ) == 0 ||
                strcmp( thisarg, "-bench"  ) == 0 ||
                strcmp( thisarg, "-test" ) == 0 )
      {
        havemode = 1;
        if (argvalue)
        {
          if (__arg2cname(argvalue,CONTEST_COUNT) < CONTEST_COUNT)
            skip_next = 1;
        }
      }
      else if (( strcmp( thisarg, "-forceunlock" ) == 0 ) ||
               ( strcmp( thisarg, "-import" ) == 0 ))
      {
        if (!argvalue)
        {
          havemode = 0;
          missing_value = 1;
          if (run_level!=0)
            terminate_app = 1;
        }
        else
        {
          skip_next = 1;
          havemode = 1; //f'd up "mode" - handled in next loop
        }
      }
      else if (run_level==0)
      {
        client->quietmode = 0;
        ModeReqClear(-1); /* clear all */
        ModeReqSet( MODEREQ_CMDLINE_HELP );
        ModeReqSetArg(MODEREQ_CMDLINE_HELP,(const void *)thisarg);
        inimissing = 0; // don't need an .ini file if we just want help
        havemode = 0;
        break;
      }
      if (run_level!=0 && (missing_value || invalid_value) && logging_is_initialized)
        LogScreenRaw ("%s option ignored. (argument %s)\n", thisarg,
          ((missing_value)?("missing"):("invalid")) );
    }
  }

  TRACE_OUT((-1,"ParseCommandline(P3,%d,%d)\n",terminate_app,havemode));

  //-----------------------------------
  // In the final loop we parse the "modes".
  //-----------------------------------

  if (!terminate_app && havemode && run_level == 0)
  {
    for (pos = 1; pos < argc; pos += (1+skip_next))
    {
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      argvalue = ((pos < (argc-1))?(argv[pos+1]):((char *)NULL));
      skip_next = 0;

      if ( thisarg == NULL )
        ; // nothing
      else if (*thisarg == 0)
        ; // nothing
      else if (( strcmp( thisarg, "-fetch" ) == 0 ) ||
          ( strcmp( thisarg, "-forcefetch" ) == 0 ) ||
          ( strcmp( thisarg, "-flush"      ) == 0 ) ||
          ( strcmp( thisarg, "-forceflush" ) == 0 ) ||
          ( strcmp( thisarg, "-update"     ) == 0 ))
      {
        if (!inimissing)
        {
          client->quietmode = 0;
          int do_mode = 0;

          if ( strcmp( thisarg, "-update" ) == 0)
            do_mode = MODEREQ_FETCH | MODEREQ_FLUSH;
          else if ( strcmp( thisarg, "-fetch" ) == 0 || strcmp( thisarg, "-forcefetch" ) == 0 )
            do_mode = MODEREQ_FETCH;
          else
            do_mode = MODEREQ_FLUSH;

          ModeReqClear(-1); //clear all - only do -fetch/-flush/-update
          ModeReqSet( do_mode );
          break;
        }
      }
      else if ( strcmp(thisarg, "-ident" ) == 0)
      {
        client->quietmode = 0;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -ident
        ModeReqSet( MODEREQ_IDENT );
        break;
      }
      else if ( strcmp( thisarg, "-cpuinfo" ) == 0 )
      {
        client->quietmode = 0;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -cpuinfo
        ModeReqSet( MODEREQ_CPUINFO );
        break;
      }
      else if ( strcmp( thisarg, "-benchmark" ) == 0  ||
                strcmp( thisarg, "-bench" ) == 0 ||
                strcmp( thisarg, "-benchmark2" ) == 0 ||
                strcmp( thisarg, "-test" ) == 0 )
      {
        int do_mode = MODEREQ_BENCHMARK;
        inimissing = 0; // Don't need ini
        client->quietmode = 0;

        if (strcmp( thisarg, "-benchmark2"  ) == 0)
          do_mode = MODEREQ_BENCHMARK_QUICK;
        else if (strcmp( thisarg, "-bench"  ) == 0)
          do_mode = MODEREQ_BENCHMARK_ALLCORE;
        else if (strcmp( thisarg, "-test"  ) == 0)
          do_mode = MODEREQ_TEST_ALLCORE;

        ModeReqClear(-1); //clear all - only do benchmark/test
        ModeReqSet( do_mode );

        if (argvalue)
        {
          int contest = __arg2cname(argvalue,CONTEST_COUNT);
          if (contest < CONTEST_COUNT)
          {
            skip_next = 1;
            ModeReqLimitProject(do_mode, contest);
          }
        }
        break;
      }
      else if ( strcmp( thisarg, "-forceunlock" ) == 0 )
      {
        if (!inimissing && argvalue)
        {
          client->quietmode = 0;
          skip_next = 1;
          ModeReqClear(-1); //clear all - only do -forceunlock
          ModeReqSet(MODEREQ_UNLOCK);
          ModeReqSetArg(MODEREQ_UNLOCK,(const void *)argvalue);
          break;
        }
      }
      else if ( strcmp( thisarg, "-import" ) == 0 )
      {
        if (!inimissing && argvalue)
        {
          client->quietmode = 0;
          skip_next = 1;
          ModeReqClear(-1); //clear all - only do -import
          ModeReqSet(MODEREQ_IMPORT);
          ModeReqSetArg(MODEREQ_IMPORT,(const void *)argvalue);
          break;
        }
      }
      else if ( strcmp( thisarg, "-config" ) == 0 )
      {
        client->quietmode = 0;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -config
        ModeReqSet( MODEREQ_CONFIG );
        ModeReqSetArg( MODEREQ_CONFIG, (const void *)thisarg /* anything */);
        break;
      }
    }
  }

  //-----------------------------------------
  // done. set the inimissing bit if appropriate;
  // if hidden and a unix host, fork a new process with >/dev/null
  // -----------------------------------------

  if (!terminate_app && inimissing && run_level == 0)
  {
    client->quietmode = 0;
    ModeReqSet( MODEREQ_CONFIG );
  }
  /* BeOS gcc defines __unix__ for some strange reason.  But this works under BeOS, so keep it. */
  #if defined(__unix__) && (CLIENT_OS != OS_NEXTSTEP) && !defined(__EMX__)
  else if (!terminate_app && run_level==0 && (ModeReqIsSet(-1)==0) &&
           client->quietmode)
  {
    pid_t x = fork();
    if (x) //Parent gets pid or -1, child gets 0
    {
      terminate_app = 1;
      if (x == -1) //Error
        ConOutErr("fork() failed.  Unable to start quiet/hidden.");
    }
    else /* child */
    {
      int fd;

      if (setsid() == -1)
      {
        terminate_app = 1;
        ConOutErr("setsid() failed. Unable to start quiet/hidden.");
      }
      else
      {
        if ((fd = open("/dev/null", O_RDWR, 0)) != -1)
        {
          (void) dup2(fd, 0);
          (void) dup2(fd, 1);
          (void) dup2(fd, 2);
          if (fd > 2)
            (void) close(fd);
        }
      }
    }
  }
  #endif /* __unix__ */

  if (retcodeP)
    *retcodeP = 0;

  TRACE_OUT((-1,"ParseCommandline(%d,%d)\n",run_level,argc));

  return terminate_app;
}
