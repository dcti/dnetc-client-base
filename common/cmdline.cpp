/* Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -------------------------------------------------------------------
 * *All* option handling is performed by ParseCommandLine(), including
 * options loaded from an external .ini.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>                         
 *
 * Note to porters: your port can be expected to break frequently if your 
 * implementation does not call this or does start the client via the 
 * Client::Main() in client.cpp
 * -------------------------------------------------------------------
*/
const char *cmdline_cpp(void) {
return "@(#)$Id: cmdline.cpp,v 1.133.2.13 1999/06/13 16:07:22 cyp Exp $"; }

//#define TRACE

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "pathwork.h"  // InitWorkingDirectoryFromSamplePaths();
#include "lurk.h"      // dialup object
#include "util.h"      // trace
#include "sleepdef.h"  // usleep()
#include "modereq.h"   // get/set/clear mode request bits
#include "console.h"   // ConOutErr()
#include "clitime.h"   // CliTimer() for -until setting
#include "confrwv.h"   // ValidateConfig()
#include "clicdata.h"  // CliGetContestNameFromID()
#include "cmdline.h"   // ourselves

#if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_FREEBSD) || \
    (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_OPENBSD)
#include <dirent.h> /* for direct read of /proc/ */
#endif
    
/* -------------------------------------- */

int Client::ParseCommandline( int run_level, int argc, const char *argv[], 
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
    inifilename[0] = 0; //so we know when it changes
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
      if (argvalue && *argvalue == '-')
        argvalue = NULL; 
      skip_next = 0;
    
      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
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
          strcpy( inifilename, argvalue );
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
        #if defined(__unix__)
        {
          char buffer[1024];
          int sig = SIGHUP; char *dowhat_descrip = "-HUP'ed";
          unsigned int kill_ok = 0, kill_failed = 0; 
          int last_errno = 0, kill_found = 0;
          const char *binname = (const char *)strrchr( argv[0], '/' );
          char altbinname[] = {'r','c','5','d','e','s','\0'};
          binname = ((binname==NULL)?(argv[0]):(binname+1));
            
          if ( strcmp( thisarg, "-kill" ) == 0 ||
               strcmp( thisarg, "-shutdown") == 0 )
          { sig = SIGTERM; dowhat_descrip = "shutdown"; }
          else if (strcmp( thisarg, "-pause" ) == 0)
          { sig = SIGTSTP; dowhat_descrip = "paused";  }
          else if (strcmp( thisarg, "-unpause" ) == 0)
          { sig = SIGCONT; dowhat_descrip = "unpaused"; }

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
	      /* useless for OSs that do argv[0]="rc5des" in client.cpp */
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
		binname = (const char *)&realbinname[0];
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
                //printf("%s: %s (binname:%s,altbinname:%s)\n",dp->d_name,procname,binname,altbinname);
                if (!strcmp(procname,binname) || !strcmp(procname,altbinname))
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
                }
              }
            }  
            closedir(dirp);
          }  
          #endif
          #if (CLIENT_OS != OS_LINUX)
          // this part is only needed for OSs that do not read /proc OR
          // do not have a reliable method to set the name as read from /proc
          // (as opposed to reading it from ps output)
          const char *pscmd = NULL;
          #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_OPENBSD) || \
              (CLIENT_OS == OS_NETBSD) || (CLIENT_OS == OS_LINUX)
          pscmd = "ps ax|awk '{print$1\" \"$5}' 2>/dev/null"; /* bsd, no -o */
          //fbsd: "ps ax -o pid -o command 2>/dev/null";  /* bsd + -o ext */
          //lnux: "ps ax --format pid,comm 2>/dev/null";  /* bsd + gnu -o */
          #elif (CLIENT_OS == OS_SOLARIS) || (CLIENT_OS == OS_SUNOS) || \
                (CLIENT_OS == OS_DEC_UNIX) || (CLIENT_OS == OS_AIX) 
          pscmd = "/usr/bin/ps -ef -o pid -o comm 2>/dev/null"; /*svr4/posix*/
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
		      unsigned int i;
		      for (i=0;i<sigd_count;i++)
		      {
		        if (already_sigd[i] == thatpid)
			{
                          thatpid = 0;
			  break;
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
		      if (strcmp(procname,binname) && strcmp(procname,altbinname))
		        thatpid = 0;
		    }
		  }
		  if (thatpid != 0)
		  {
                    kill_found++;
                    if ( kill( thatpid, sig ) == 0)
                      kill_ok++;
                    else if ((errno != ESRCH) && (errno != ENOENT))
                    {
                      kill_failed++;
                      last_errno = errno;
                    }
		  }
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
        #elif ((CLIENT_OS == OS_WIN16 || CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN32S))
        {
          int rc, cmd = IDM_RESTART;
          const char *dowhat_descrip = "restarted";
          
          if ( strcmp( thisarg, "-kill" ) == 0 ||
               strcmp( thisarg, "-shutdown") == 0 )
          {
            cmd = IDM_SHUTDOWN;
            dowhat_descrip = "shutdown";
          }
          else if (strcmp( thisarg, "-pause" ) == 0)
          {
            cmd = IDM_PAUSE;
            dowhat_descrip = "paused";
          }
          else if (strcmp( thisarg, "-unpause" ) == 0)
          {
            cmd = IDM_UNPAUSE;
            dowhat_descrip = "unpaused";
          }
          
          rc = w32ConSendIDMCommand( cmd );
          terminate_app = 1;
          if (!loop0_quiet)
          {
            char scratch[128];
            if (rc < 0)
              sprintf(scratch,"No distributed.net clients are currently running.\n"
                              "None were %s.", dowhat_descrip);
            else if (rc > 0)
              sprintf(scratch,"A distributed.net client was found but "
                              "could not be %s.\n", dowhat_descrip);
            else
              sprintf(scratch,"The distributed.net client has been %s.", dowhat_descrip);
            ConOutModal(scratch);
          }
        }
        #else
          not_supported = 1;
        #endif
      }
      else if ( strcmp(thisarg, "-install" ) == 0)
      {
        #if (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
        winInstallClient(loop0_quiet); /*w32pre.cpp*/
        terminate_app = 1;
        #elif (CLIENT_OS == OS_OS2)
        extern int os2CliInstallClient(int quiet, const char *exename);
        os2CliInstallClient(loop0_quiet, argv[0]); /* os2inst.cpp */
        terminate_app = 1;
        #else
        not_supported = 1;
        #endif
      }
      else if ( strcmp(thisarg, "-uninstall" ) == 0)
      {
        #if (CLIENT_OS == OS_OS2)
        extern int os2CliUninstallClient(int /*do it without feedback*/);
        os2CliUninstallClient(loop0_quiet); /* os2inst.cpp */
        terminate_app = 1;
        #elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
        winUninstallClient(loop0_quiet); /*w32pre.cpp*/
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
    if (inifilename[0]==0) // determine the filename of the ini file
    {
      char * inienvp = getenv( "RC5INI" );
      if ((inienvp != NULL) && (strlen( inienvp ) < sizeof(inifilename)))
        strcpy( inifilename, inienvp );
      else
      {
        #if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_DOS) || \
            (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S) || \
            (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
        //not really needed for netware (appname in argv[0] won't be anything 
        //except what I tell it to be at link time.)
        inifilename[0] = 0;
        if (argv[0]!=NULL && ((strlen(argv[0])+5) < sizeof(inifilename)))
        {
          strcpy( inifilename, argv[0] );
          char *slash = strrchr( inifilename, '/' );
          char *slash2 = strrchr( inifilename, '\\');
          if (slash2 > slash ) slash = slash2;
          slash2 = strrchr( inifilename, ':' );
          if (slash2 > slash ) slash = slash2;
          if ( slash == NULL ) slash = inifilename;
          if ( ( slash2 = strrchr( slash, '.' ) ) != NULL ) // ie > slash
            strcpy( slash2, ".ini" );
          else if ( strlen( slash ) > 0 )
           strcat( slash, ".ini" );
        }
        if ( inifilename[0] == 0 )
          strcpy( inifilename, "rc5des.ini" );
        #elif (CLIENT_OS == OS_VMS)
        strcpy( inifilename, "rc5des" EXTN_SEP "ini" );
        #else
        strcpy( inifilename, argv[0] );
        strcat( inifilename, EXTN_SEP "ini" );
        #endif
      }
    } // if (inifilename[0]==0)

    TRACE_OUT((+1,"InitWorkingDirectoryFromSamplePaths()\n"));
    InitWorkingDirectoryFromSamplePaths( inifilename, argv[0] );
    TRACE_OUT((-1,"InitWorkingDirectoryFromSamplePaths()\n"));
    
    if ( (pos = ReadConfig(this)) != 0)
    {
      if (pos < 0) /* fatal */
        terminate_app = 1;
      else
      {
        stopiniio = 1; /* client class */
        ModeReqSet( MODEREQ_CONFIG );
        inimissing = 1;
      }
    }
  } 

  TRACE_OUT((-1,"ParseCommandline(P2,%s)\n",inifilename));

  //-----------------------------------
  // In the next loop we parse the other options
  //-----------------------------------

  TRACE_OUT((+1,"ParseCommandline(P3)\n"));

  if (!terminate_app && ((run_level == 0) || (logging_is_initialized)))
  {
    for (pos = 1; pos < argc; pos += (1+skip_next))
    {
      int missing_value = 0;
      skip_next = 0;
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      argvalue = ((pos < (argc-1))?(argv[pos+1]):((char *)NULL));
      if (argvalue && *argvalue == '-')
        argvalue = NULL;

      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
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
      else if ( strcmp( thisarg, "-guistart" ) == 0) 
      {
        #if (CLIENT_OS == OS_WIN32)
        //handled by GUI command line parser
        #endif
      }
      else if ( strcmp( thisarg, "-guirestart" ) == 0) 
      {          // See if are restarting (hence less banners wanted)
        #if (CLIENT_OS == OS_RISCOS)
        if (run_level == 0)
          guirestart = 1;
        #endif
      }
      else if ( strcmp( thisarg, "-hide" ) == 0 ||   
                strcmp( thisarg, "-quiet" ) == 0 )
      {
        if (run_level == 0)
          quietmode = 1;
      }
      else if ( strcmp( thisarg, "-noquiet" ) == 0 )      
      {
        if (run_level == 0)
          quietmode = 0;
      }
      else if ( strcmp(thisarg, "-percentoff" ) == 0)
      {
        if (run_level == 0)
          percentprintingoff = 1;
      }
      else if ( strcmp( thisarg, "-nofallback" ) == 0 )   
      {
        if (run_level == 0)
          nofallback = 1;
      }
      else if ( strcmp( thisarg, "-lurk" ) == 0 )
      {
        #if defined(LURK)
        if (run_level == 0)
          dialup.lurkmode=CONNECT_LURK;      // Detect modem connections
        #endif
      }
      else if ( strcmp( thisarg, "-lurkonly" ) == 0 )
      {
        #if defined(LURK)
        if (run_level == 0)
          dialup.lurkmode=CONNECT_LURKONLY;  // Only connect when modem connects
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
                             dialup.connifacemask );
          }
          else
          {
            strncpy(dialup.connifacemask, argvalue, sizeof(dialup.connifacemask) );
            dialup.connifacemask[sizeof(dialup.connifacemask)-1] = 0;
          }
          #endif
        }
      }
      else if ( strcmp( thisarg, "-noexitfilecheck" ) == 0 )
      {
        if (run_level == 0)
          noexitfilecheck=1;             // Change network timeout
      }
      else if ( strcmp( thisarg, "-runoffline" ) == 0 || 
                strcmp( thisarg, "-runonline" ) == 0) 
      {
        if (run_level != 0)
        {
          if (logging_is_initialized)
            LogScreenRaw("Client will run with%s network access.\n", 
                       ((offlinemode)?("out"):("")) );
        }
        else 
          offlinemode = ((strcmp( thisarg, "-runoffline" ) == 0)?(1):(0));
      }
      else if (strcmp(thisarg,"-runbuffers")==0 || strcmp(thisarg,"-run")==0) 
      {
        if (run_level != 0)
        {
          if (logging_is_initialized)
          {
            LogScreenRaw("Warning: %s is obsolete.\n"
                         "         Active settings: -runo%sline and -n %d%s.\n",
              thisarg, ((offlinemode)?("ff"):("n")), 
              ((blockcount<0)?(-1):((int)blockcount)),
              ((blockcount<0)?(" (exit on empty buffers)"):("")) );
          }
        }
        else
        {
          if (strcmp(thisarg,"-run")==0)
          {
            offlinemode = 0;
            if (blockcount < 0)
              blockcount = 0;
          }
          else /* -runbuffers */
          {
            offlinemode = 1;
            blockcount = -1;
          }
        }
      }
      else if ( strcmp( thisarg, "-nodisk" ) == 0 ) 
      {
        if (run_level == 0)
          nodiskbuffers=1;              // No disk buff-*.rc5 files.
        inimissing = 0; // Don't complain if the inifile is missing        
      }
      else if ( strcmp(thisarg, "-frequent" ) == 0)
      {
        if (run_level!=0)
        {
          if (logging_is_initialized && connectoften)
            LogScreenRaw("Buffer thresholds will be checked frequently.\n");
        }
        else
          connectoften = 1;
      }
      else if ( strcmp( thisarg, "-b" ) == 0 || strcmp( thisarg, "-b2" ) == 0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-b2" ) == 0 ) ? (1) : (0));
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s buffer thresholds to %u\n",
                   CliGetContestNameFromID(conid), (unsigned int)inthreshold[conid] );
          }
          else if ( atoi( argvalue ) > 0)
          {
            inimissing = 0; // Don't complain if the inifile is missing
            outthreshold[conid] = inthreshold[conid] = (s32) atoi( argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-bin" ) == 0 || strcmp( thisarg, "-bin2")==0)
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-bin2" ) == 0 ) ? (1) : (0));
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s in-buffer threshold to %u\n",
                 CliGetContestNameFromID(conid), (unsigned int)inthreshold[conid] );
          }
          else if ( atoi( argvalue ) > 0)
          {
            inimissing = 0; // Don't complain if the inifile is missing
            inthreshold[conid] = (s32) atoi( argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-bout" ) == 0 || strcmp( thisarg, "-bout2")==0)
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-bout2" ) == 0 ) ? (1) : (0));
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s out-buffer threshold to %u\n",
                  CliGetContestNameFromID(conid), (unsigned int)outthreshold[conid] );
          }
          else if ( atoi( argvalue ) > 0)
          {
            inimissing = 0; // Don't complain if the inifile is missing
            outthreshold[conid] = (s32) atoi( argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-inbase" ) == 0 || strcmp( thisarg, "-outbase")==0 )
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          int out = ( strcmp( thisarg, "-outbase" ) == 0 );
          char *p = (out ? out_buffer_basename : in_buffer_basename);
          skip_next = 1;
          if ( run_level == 0 )
          {
            strncpy( p, argvalue, sizeof(in_buffer_basename) );
            p[sizeof(in_buffer_basename)-1]=0;
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
              LogScreenRaw("Setting uue/http mode to %u\n",(unsigned int)uuehttpmode);
          }
          else
          {
            uuehttpmode = atoi( argvalue );
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
              LogScreenRaw("Setting keyserver to %s\n", keyproxy );
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            autofindkeyserver = 0;
            strcpy( keyproxy, argvalue );
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
              LogScreenRaw("Setting keyserver port to %u\n",(unsigned int)keyport);
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            keyport = atoi( argvalue );
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
              LogScreenRaw("Setting SOCKS/HTTP proxy to %s\n", httpproxy);
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            strcpy( httpproxy, argvalue );
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
              LogScreenRaw("Setting SOCKS/HTTP proxy port to %u\n",(unsigned int)httpport);
          }
          else
          {
            inimissing = 0; // Don't complain if the inifile is missing
            httpport = (s32) atoi( argvalue );
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
              LogScreenRaw("Setting log file to %s\n", logname );
          }
          else
          {
            strcpy( logname, argvalue );
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
              LogScreenRaw("Setting Mail message length to %u\n", (unsigned int)messagelen );
          }
          else
          {
            messagelen = (s32) atoi(argvalue);
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
              LogScreenRaw("Setting smtp port to %u\n", (unsigned int)smtpport);
          }
          else
          {
            smtpport = (s32) atoi(argvalue);
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
              LogScreenRaw("Setting SMTP relay host to %s\n", smtpsrvr);
          }
          else
          {
            strcpy( smtpsrvr, argvalue );
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
              LogScreenRaw("Setting mail 'from' address to %s\n", smtpfrom );
          }
          else
            strcpy( smtpfrom, argvalue );
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
              LogScreenRaw("Setting mail 'to' address to %s\n", smtpdest );
          }
          else
          {
            strcpy( smtpdest, argvalue );
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
              LogScreenRaw("Setting distributed.net ID to %s\n", id );
          }
          else
          {
            strcpy( id, argvalue );
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
                                             (unsigned int)(nettimeout));
          }
          else
          {
            nettimeout = atoi(argvalue);
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
      else if ( strcmp( thisarg, "-c" ) == 0 || strcmp( thisarg, "-cputype" ) == 0)
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting cputype to %d\n", (int)cputype);
          }
          else
          {
            cputype = (s32) atoi( argvalue );
            inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      }
      else if ( strcmp( thisarg, "-nice" ) == 0 ) // Nice level
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting priority to %u\n", priority );
          }
          else
          {
            priority = (s32) atoi( argvalue );
            priority = ((priority==2)?(8):((priority==1)?(4):(0)));
            inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      }
      else if ( strcmp( thisarg, "-priority" ) == 0 ) // Nice level
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting priority to %u\n", priority );
          }
          else
          {
            priority = (s32) atoi( argvalue );
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
              else if (minutes == 0)
                LogScreenRaw("Setting time limit to zero (no limit).\n");
              else
              {
                struct timeval tv; CliTimer(&tv); tv.tv_sec+=(time_t)(minutes*60);
                LogScreenRaw("Setting time limit to %u:%02u hours (stops at %s)\n",
                             minutes/60, minutes%60, CliGetTimeString(&tv,1) );
              }
            }
          }
          else if (isok)
          {  
            minutes = ((h*60)+m);
            if (isuntil)
            {
              time_t timenow = CliTimer(NULL)->tv_sec;
              struct tm *ltm = localtime( &timenow );
              if (ltm->tm_hour > h || (ltm->tm_hour == h && ltm->tm_min >= m))
                minutes+=(24*60);
              minutes -= (((ltm->tm_hour)*60)+(ltm->tm_min));
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
              if (blockcount < 0)
                LogScreenRaw("Client will exit when buffers are empty.\n");
              else
                LogScreenRaw("Setting block completion limit to %u%s\n",
                    (unsigned int)blockcount, 
                    ((blockcount==0)?(" (no limit)"):("")));
            }
          }
          else if ( (blockcount = atoi( argvalue )) < 0)
            blockcount = -1;
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
            numcpu = (s32) atoi(argvalue);
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
                                                 checkpoint_file );
          }
          else
          {
            strcpy(checkpoint_file, argvalue );
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
              LogScreenRaw("Setting pause file to %s\n",pausefile);
          }
          else
          {
            strcpy(pausefile, argvalue );
          }
        }
      }
      else if ( strcmp( thisarg, "-blsize" ) == 0)
      {
        if (!argvalue)
          missing_value = 1;
        else
        {
          skip_next = 1;
          if (run_level != 0)
          {
            if (logging_is_initialized)
              LogScreenRaw("Setting preferred blocksize to 2^%d\n",preferred_blocksize);
          }
          else
          {
            preferred_blocksize = (s32) atoi(argvalue);
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
          ( strcmp( thisarg, "-test"        ) == 0 ) ||
          ( strcmp( thisarg, "-config"      ) == 0 ) ||
          ( strncmp( thisarg, "-benchmark", 10 ) == 0))
      {
        havemode = 1; //nothing - handled in next loop
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
        quietmode = 0;
        ModeReqClear(-1); /* clear all */
        ModeReqSet( MODEREQ_CMDLINE_HELP );
        ModeReqSetArg(MODEREQ_CMDLINE_HELP,(void *)thisarg);
        inimissing = 0; // don't need an .ini file if we just want help
        havemode = 0;
        break;
      }
      if (run_level!=0 && missing_value && logging_is_initialized)
        LogScreenRaw ("%s option ignored. (argument missing)\n", thisarg );
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
      if (argvalue && *argvalue == '-')
        argvalue = NULL;
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
          quietmode = 0;
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
        quietmode = 0;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -ident
        ModeReqSet( MODEREQ_IDENT );
        break;
      }
      else if ( strcmp( thisarg, "-cpuinfo" ) == 0 )
      {
        quietmode = 0;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -cpuinfo
        ModeReqSet( MODEREQ_CPUINFO );
        break;
      }
      else if ( strcmp( thisarg, "-test" ) == 0 )
      {
        quietmode = 0;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -test
        ModeReqSet( MODEREQ_TEST );
        break;
      }
      else if (strncmp( thisarg, "-benchmark", 10 ) == 0)
      {
        quietmode = 0;
        int do_mode = 0;
        thisarg += 10;

        if (*thisarg == '2')
        {
          do_mode |= MODEREQ_BENCHMARK_QUICK;
          thisarg++;
        }
        if ( strcmp( thisarg, "rc5" ) == 0 )  
          do_mode |= MODEREQ_BENCHMARK_RC5;
        else if ( strcmp( thisarg, "des" ) == 0 )
           do_mode |= MODEREQ_BENCHMARK_DES;
        else 
          do_mode |= (MODEREQ_BENCHMARK_DES | MODEREQ_BENCHMARK_RC5);

        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do benchmark
        ModeReqSet( do_mode );
        break;
      }
      else if ( strcmp( thisarg, "-forceunlock" ) == 0 )
      {
        if (!inimissing && argvalue)
        {
          quietmode = 0;
          skip_next = 1;
          ModeReqClear(-1); //clear all - only do -forceunlock
          ModeReqSet(MODEREQ_UNLOCK);
          ModeReqSetArg(MODEREQ_UNLOCK,(void *)argvalue);
          break;
        }
      }
      else if ( strcmp( thisarg, "-import" ) == 0 )
      {
        if (!inimissing && argvalue)
        {
          quietmode = 0;
          skip_next = 1;
          ModeReqClear(-1); //clear all - only do -import
          ModeReqSet(MODEREQ_IMPORT);
          ModeReqSetArg(MODEREQ_IMPORT,(void *)argvalue);
          break;
        }
      }
      else if ( strcmp( thisarg, "-config" ) == 0 )
      {
        quietmode = 0;
        ModeReqClear(-1); //clear all - only do -config
        inimissing = 1; //force run config
        break;
      }
    }
  }

  //-----------------------------------------
  // done. set the inimissing bit if appropriate;
  // if hidden and a unix host, fork a new process with >/dev/null
  // -----------------------------------------

  if (inimissing && run_level == 0)
  {
    quietmode = 0;
    ModeReqSet( MODEREQ_CONFIG );
  }
  #if defined(__unix__)
  else if (!terminate_app && run_level==0 && (ModeReqIsSet(-1)==0) && quietmode)
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
      /* don't/can't use these anymore */
      #ifndef _PATH_DEVNULL //paths.h
      #define _PATH_DEVNULL "/dev/null"
      #endif
      //if (isatty(fileno(stdin)))
        freopen(_PATH_DEVNULL,"r",stdin);
      if (isatty(fileno(stdout)))
        freopen(_PATH_DEVNULL,"w",stdout);
      if (isatty(fileno(stderr)))
        freopen(_PATH_DEVNULL,"w",stderr);
    }
  }
  #endif
  
  if (retcodeP) 
    *retcodeP = 0;

  TRACE_OUT((-1,"ParseCommandline(%d,%d)\n",run_level,argc));

  return terminate_app;
}

