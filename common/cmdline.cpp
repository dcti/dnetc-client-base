// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cmdline.cpp,v $
// Revision 1.107  1998/12/10 18:52:30  cyp
// Fixed a LogScreen() that should have been a LogScreenRaw()
//
// Revision 1.106  1998/12/05 22:26:18  cyp
// Added -kill/-shutdown and -hup/-restart support for win32/win16
// (aka "waaah, how do I stop a hidden client?") and for *nixes.
// Yes, unix support is ugly, but it works.
//
// Revision 1.105  1998/12/01 15:04:44  cyp
// Both -run and -runbuffers give obsolete warnings and show the current
// state of offlinemode and blockcount.
//
// Revision 1.104  1998/11/28 19:48:58  cyp
// Added win16 -install/-uninstall support.
//
// Revision 1.103  1998/11/26 07:23:03  cyp
// Added stopiniio flag to client object. Ini should not be updated with
// randomprefix number if it didn't exist to begin with.
//
// Revision 1.102  1998/11/22 14:54:35  cyp
// Adjusted to reflect changed -runonline, -runoffline, -n behaviour
//
// Revision 1.101  1998/11/21 13:08:09  remi
// Fixed "Setting cputype to" when cputype < 0.
//
// Revision 1.100  1998/11/19 20:48:53  cyp
// Rewrote -until/-h handling. Did away with useless client.hours (time-to-die
// is handled by client.minutes anyway). -until/-h/hours all accept "hh:mm"
// format now (although they do continue to support the asinine "hh.mm").
//
// Revision 1.99  1998/11/15 11:49:42  cyp
// ModeReq(-1) == 0 (not != 0) before the unix the system call.
//
// Revision 1.98  1998/11/15 11:00:16  remi
// Moved client->SelectCore() for -test and -benchmark* from cmdline.cpp to
// modereq.cpp and told it to not be quiet.
//
// Revision 1.97  1998/11/14 13:51:07  cyp
// -hidden/-quiet spawn of a new process for unix hosts is done here instead
// of from main.
//
// Revision 1.96  1998/11/10 21:45:59  cyp
// ParseCommandLine() is now one-pass. (a second pass is available so that
// lusers can see the override messages on the screen)
//
// Revision 1.95  1998/11/09 20:05:20  cyp
// Did away with client.cktime altogether. Time-to-Checkpoint is calculated
// dynamically based on problem completion state and is now the greater of 1
// minute and time_to_complete_1_percent (an average change of 1% that is).
//
// Revision 1.94  1998/11/09 01:59:20  remi
// Fixed a bug where ./rc5des -help will print the help text and then go to
// config mode if there isn't any .ini file.
//
// Revision 1.93  1998/11/08 19:03:22  cyp
// -help (and invalid command line options) are now treated as "mode" requests.
//
// Revision 1.92  1998/11/04 21:28:19  cyp
// Removed redundant ::hidden option. ::quiet was always equal to ::hidden.
//
// Revision 1.91  1998/10/26 03:19:57  cyp
// More tags fun.
//
// Revision 1.9  1998/10/19 12:59:51  cyp
// completed implementation of 'priority'.
//
// Revision 1.8  1998/10/19 12:42:18  cyp
// win16 changes
//
// Revision 1.7  1998/10/11 00:41:22  cyp
// Implemented ModeReq
//
// Revision 1.6  1998/10/08 20:54:39  cyp
// Added buffwork.h to include list for UnlockBuffer() prototype.
//
// Revision 1.5  1998/10/04 20:38:45  remi
// -benchmark shouldn't ask for something.
//
// Revision 1.4  1998/10/03 23:12:58  remi
// -nommx is for both DES and RC5 mmx cores.
// we don't need any .ini file for -ident, -cpuinfo, -test and -benchmark*
//
// Revision 1.3  1998/10/03 12:32:19  cyp
// Removed a trailing ^Z
//
// Revision 1.2  1998/10/03 03:56:51  cyp
// running "modes", excluding -fetch/-flush but including -config (or a
// missing ini file) disables -hidden and -quiet. -install and -uninstall
// are now run before all other checks. trap for argv[x]=="" added. -noquiet
// now negates xxhidden as well as quiet.
//
// Revision 1.1  1998/08/28 21:35:42  cyp
// Created (complete rewrite). The command line is now "reusable", and allows
// main() to be re-startable. *All* option handling is done here.
//

#if (!defined(lint) && defined(__showids__))
const char *cmdline_cpp(void) {
return "@(#)$Id: cmdline.cpp,v 1.107 1998/12/10 18:52:30 cyp Exp $"; }
#endif

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "pathwork.h"  // InitWorkingDirectoryFromSamplePaths();
#include "modereq.h"   // get/set/clear mode request bits
#include "console.h"   // ConOutErr()
#include "clitime.h"   // CliTimer() for -until setting
#include "cmdline.h"   // ourselves

/* -------------------------------------- */

int Client::ParseCommandline( int run_level, int argc, const char *argv[], 
                              int *retcodeP, int logging_is_initialized )
{
  int inimissing, pos, skip_next, terminate_app, havemode;
  const char *thisarg, *nextarg;

  terminate_app = 0;
  inimissing = 0;
  havemode = 0;

  //-----------------------------------
  // In the first loop we (a) get the ini filename and 
  // (b) get switches that won't be overriden by the ini
  //-----------------------------------

  if (run_level == 0)
    {
    inifilename[0] = 0; //so we know when it changes
    ModeReqClear(-1);   // clear all mode request bits

    skip_next = 0;
    for (pos = 1;((terminate_app==0) && (pos<argc));pos+=(1+skip_next))
      {
      int not_supported = 0;
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;
    
      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
      else if ( strcmp(thisarg, "-ini" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1; 
          strcpy( inifilename, nextarg );
          }
        else
          terminate_app = 1;
        }
      else if ( ( strcmp( thisarg, "-restart" ) == 0) || 
                ( strcmp( thisarg, "-hup" ) == 0 ) ||
                ( strcmp( thisarg, "-kill" ) == 0 ) ||
                ( strcmp( thisarg, "-shutdown" ) == 0 ) )
        {
        #if ((CLIENT_OS == OS_DEC_UNIX)    || (CLIENT_OS == OS_HPUX)    || \
             (CLIENT_OS == OS_QNX)         || (CLIENT_OS == OS_OSF1)    || \
             (CLIENT_OS == OS_BSDI)        || (CLIENT_OS == OS_SOLARIS) || \
             (CLIENT_OS == OS_IRIX)        || (CLIENT_OS == OS_SCO)     || \
             (CLIENT_OS == OS_LINUX)       || (CLIENT_OS == OS_NETBSD)  || \
             (CLIENT_OS == OS_UNIXWARE)    || (CLIENT_OS == OS_DYNIX)   || \
             (CLIENT_OS == OS_MINIX)       || (CLIENT_OS == OS_MACH10)  || \
             (CLIENT_OS == OS_AIX)         || (CLIENT_OS == OS_AUX)     || \
             (CLIENT_OS == OS_OPENBSD)     || (CLIENT_OS == OS_SUNOS)   || \
             (CLIENT_OS == OS_ULTRIX)      || (CLIENT_OS == OS_DGUX))
          {
          terminate_app = 1;
          char buffer[1024];
          int dokill = ( strcmp( thisarg, "-kill" ) == 0 ||
                         strcmp( thisarg, "-shutdown") == 0 );
          if (nextarg != NULL && *nextarg == '[')
            {
            int sig = ((dokill)?(SIGTERM):(SIGHUP));
            pid_t ourpid[2];
            ourpid[0] = atol( nextarg+1 );
            ourpid[1] = getpid();
            pos+=2;
            if (!(pos<argc && argv[pos]!=NULL && isdigit(argv[pos][0])))
              {
              ConOutErr( ((pos==argc || argv[pos]==NULL)?
                   ("Unable to get pid list."):(argv[pos])) );
              }
            else
              {
              unsigned int kill_ok = 0;
              unsigned int kill_failed = 0;
              int last_errno = 0;
              do{
                pid_t tokill = atol( argv[pos++] );
                if (tokill!=ourpid[0] && tokill!=ourpid[1] && tokill!=0)
                  {
                  if ( kill( tokill, sig ) == 0)
                    kill_ok++;
                  else
                    {
                    kill_failed++;
                    last_errno = errno;
                    }
                  }
                } while (pos<argc && argv[pos]!=NULL && isdigit(argv[pos][0]));
              if ((kill_ok + kill_failed) == 0)    
                {
                sprintf(buffer,"No distributed.net clients are currently running.\n"
                               "None were %s.", ((dokill)?("killed"):("-HUP'ed")));
                ConOutErr(buffer);
                }
              else 
                {
                sprintf(buffer,"%u distributed.net client%s %s. %u failure%s%s%s%s.",
                           kill_ok, 
                           ((kill_ok==1)?(" was"):("s were")),
                           ((dokill)?("killed"):("-HUP'ed")), 
                           kill_failed, (kill_failed==1)?(""):("s"),
                           ((kill_failed==0)?(""):(" (")),
                           ((kill_failed==0)?(""):(strerror(last_errno))),
                           ((kill_failed==0)?(""):(")")) );
                ConOutErr(buffer);
                }
              }
            }
          else
            {
            const char *p = "pidof";
            const char *q = (const char *)strrchr( argv[0], '/' );
            q = ((q==NULL)?(argv[0]):(q+1));
            sprintf(buffer, "%s %s [%lu] `%s %s`", argv[0], thisarg,
                    (unsigned long)getpid(), p, q );
            if (system( buffer ) != 0)
              {
              sprintf(buffer, "%s failed. Unable to get pid list.", thisarg );
              ConOutErr( buffer );
              }
            }
          }
        #elif ((CLIENT_OS==OS_WIN16 || CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN32S))
          {
          terminate_app = 1;
          char scratch[128];
          int dokill = ( strcmp( thisarg, "-kill" ) == 0 ||
                         strcmp( thisarg, "-shutdown") == 0 );
          thisarg = ((dokill)?("shut down"):("restarted"));
          if (w32ConSendIDMCommand( ((dokill)?(IDM_SHUTDOWN):(IDM_RESTART)) )!=0)
            {
            sprintf(scratch,"No distributed.net clients are currently running.\n"
                            "None were %s.", thisarg);
            ConOutErr(scratch);
            }
          else
            {
            sprintf(scratch,"The distributed.net client has been %s.", thisarg);
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
        winInstallClient(0); /*w32pre.cpp*/
        terminate_app = 1;
        #elif (CLIENT_OS == OS_OS2)
        os2CliInstallClient(0);
        terminate_app = 1;
        #else
        not_supported = 1;
        #endif
        }
      else if ( strcmp(thisarg, "-uninstall" ) == 0)
        {
        #if (CLIENT_OS == OS_OS2)
        os2CliUninstallClient(0);
        terminate_app = 1;
        #elif (CLIENT_OS==OS_WIN32) || (CLIENT_OS==OS_WIN16) || (CLIENT_OS==OS_WIN32S)
        winUninstallClient(0); /*w32pre.cpp*/
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

  //-----------------------------------
  // In the next section we get inifilename defaults
  // and load the config from file
  //-----------------------------------
    
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
        if (argv[0] != NULL && ((strlen(argv[0])+5) < sizeof(inifilename)))
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

    InitWorkingDirectoryFromSamplePaths( inifilename, argv[0] );

    if ( ReadConfig() != 0)
      {
      stopiniio = 1; /* client class */
      ModeReqSet( MODEREQ_CONFIG );
      inimissing = 1;
      }
    } 

  //-----------------------------------
  // In the next loop we parse the other options
  //-----------------------------------

  if (!terminate_app && ((run_level == 0) || (logging_is_initialized)))
    {
    for (pos = 1;pos<argc; pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;

      if ( thisarg == NULL )
        { //nothing
        }
      else if (*thisarg == 0)
        { //nothing
        }
      else if ( strcmp( thisarg, "-ini" ) == 0) 
        {
        //we already did this so skip it
        if (nextarg)
          skip_next = 1;
        }
      else if ( strcmp( thisarg, "-guiriscos" ) == 0) 
        {                       
        #if (CLIENT_OS == OS_RISCOS)
        if (run_level == 0)
          guiriscos=1;
        #endif
        }
      else if ( strcmp( thisarg, "-guirestart" ) == 0) 
        {          // See if are restarting (hence less banners wanted)
        #if (CLIENT_OS == OS_RISCOS)
        if (run_level == 0)
          guirestart=1;
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
          dialup.lurkmode=1;               // Detect modem connections
        #endif
        }
      else if ( strcmp( thisarg, "-lurkonly" ) == 0 )
        {               
        #if defined(LURK)
        if (run_level == 0)
          dialup.lurkmode=2;              // Only connect when modem connects
        #endif
        }
      else if ( strcmp( thisarg, "-noexitfilecheck" ) == 0 )
        {                          
        if (run_level == 0)
          noexitfilecheck=1;             // Change network timeout
        }
      else if ( strcmp( thisarg, "-runoffline" ) == 0 || 
                strcmp( thisarg, "-runonline" ) == 0) 
        {
        if (run_level!=0)
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
        if (run_level!=0)
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
        }
      else if ( strcmp(thisarg, "-frequent" ) == 0)
        {
        if (run_level!=0)
          {
          if (logging_is_initialized && connectoften)
            LogScreenRaw("Buffer thresholds will be checked frequently.\n");
          }
        else
          connectoften=1;
        }
      else if ( strcmp(thisarg, "-nommx" ) == 0)
        {
        #if (CLIENT_CPU == CPU_X86) && (defined(MMX_BITSLICER) || defined(MMX_RC5))
        if (run_level == 0)
          usemmx=0;
        //we don't print a message because usemmx is 
        //internal/undocumented and for developer use only
        #endif
        }
      else if ( strcmp( thisarg, "-b" ) == 0 || strcmp( thisarg, "-b2" ) == 0 )
        { 
        if (nextarg)
          {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-b2" ) == 0 ) ? (1) : (0));
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s buffer thresholds to %u\n",
                   ((conid)?("DES"):("RC5")), (unsigned int)inthreshold[conid] );
            }
          else if ( atoi( nextarg ) > 0)
            {
            inimissing = 0; // Don't complain if the inifile is missing
            outthreshold[conid] = inthreshold[conid] = (s32) atoi( nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-bin" ) == 0 || strcmp( thisarg, "-bin2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-bin2" ) == 0 ) ? (1) : (0));
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s in-buffer threshold to %u\n",
                 ((conid)?("DES"):("RC5")), (unsigned int)inthreshold[conid] );
            }
          else if ( atoi( nextarg ) > 0)
            {
            inimissing = 0; // Don't complain if the inifile is missing
            inthreshold[conid] = (s32) atoi( nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-bout" ) == 0 || strcmp( thisarg, "-bout2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-bout2" ) == 0 ) ? (1) : (0));
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s out-buffer threshold to %u\n",
                  ((conid)?("DES"):("RC5")), (unsigned int)outthreshold[conid] );
            }
          else if ( atoi( nextarg ) > 0)
            {
            inimissing = 0; // Don't complain if the inifile is missing
            outthreshold[conid] = (s32) atoi( nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-in" ) == 0 || strcmp( thisarg, "-in2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-in2" ) == 0 ) ? (1) : (0));
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s in-buffer file to %s\n",
                    ((conid)?("DES"):("RC5")), in_buffer_file[conid] );
            }
          else
            {
            inimissing = 0; // Don't complain if the inifile is missing
            in_buffer_file[conid][sizeof(in_buffer_file[0])-1]=0;
            strncpy(in_buffer_file[conid], nextarg, sizeof(in_buffer_file[0]) );
            }
          }
        }
      else if ( strcmp( thisarg, "-out" ) == 0 || strcmp( thisarg, "-out2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-out2" ) == 0 ) ? (1) : (0));
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s out-buffer file to %s\n",
                    ((conid)?("DES"):("RC5")), out_buffer_file[conid] );
            }
          else
            {
            inimissing = 0; // Don't complain if the inifile is missing
            out_buffer_file[conid][sizeof(out_buffer_file[0])-1]=0;
            strncpy(out_buffer_file[conid], nextarg, sizeof(out_buffer_file[0]) );
            }
          }
        }
      else if ( strcmp( thisarg, "-u" ) == 0 ) // UUE/HTTP Mode
        {                    
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting uue/http mode to %u\n",(unsigned int)uuehttpmode);
            }
          else
            {
            uuehttpmode = (s32) atoi( nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-a" ) == 0 ) // Override the keyserver name
        {                    
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting keyserver to %s\n", keyproxy );
            }
          else
            {
            inimissing = 0; // Don't complain if the inifile is missing
            strcpy( keyproxy, nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-p" ) == 0 ) // UUE/HTTP Mode
        {                    
        if (nextarg)
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
            keyport = (s32) atoi( nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-ha" ) == 0 ) // Override the http proxy name
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting SOCKS/HTTP proxy to %s\n", httpproxy);
            }
          else
            {
            inimissing = 0; // Don't complain if the inifile is missing
            strcpy( httpproxy, nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-hp" ) == 0 ) // Override the socks/http proxy port
        {                    
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting SOCKS/HTTP proxy port to %u\n",(unsigned int)httpport);
            }
          else
            {
            inimissing = 0; // Don't complain if the inifile is missing
            httpport = (s32) atoi( nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-l" ) == 0 ) // Override the log file name
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting log file to %s\n", logname );
            }
          else
            {
            strcpy( logname, nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-smtplen" ) == 0 ) // Override the mail message length
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting Mail message length to %u\n", (unsigned int)messagelen );
            }
          else
            {
            messagelen = (s32) atoi(nextarg);
            }
          }
        }
      else if ( strcmp( thisarg, "-smtpport" ) == 0 ) // Override the smtp port for mailing
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting smtp port to %u\n", (unsigned int)smtpport);
            }
          else
            {
            smtpport = (s32) atoi(nextarg);
            }
          }
        }
      else if ( strcmp( thisarg, "-smtpsrvr" ) == 0 ) // Override the smtp server name
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting SMTP relay host to %s\n", smtpsrvr);
            }
          else
            {
            strcpy( smtpsrvr, nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-smtpfrom" ) == 0 ) // Override the smtp source id
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting mail 'from' address to %s\n", smtpfrom );
            }
          else
            strcpy( smtpfrom, nextarg );
          }
        }
      else if ( strcmp( thisarg, "-smtpdest" ) == 0 ) // Override the smtp source id
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting mail 'to' address to %s\n", smtpdest );
            }
          else
            {
            strcpy( smtpdest, nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-e" ) == 0 )     // Override the email id
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting distributed.net ID to %s\n", id );
            }
          else
            {
            strcpy( id, nextarg );
            inimissing = 0; // Don't complain if the inifile is missing
            }
          }
        }
      else if ( strcmp( thisarg, "-nettimeout" ) == 0 ) // Change network timeout
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting network timeout to %u\n", 
                                             (unsigned int)(nettimeout));
            }
          else
            {
            nettimeout = atoi(nextarg);
            }
          }
        }
      else if ( strcmp( thisarg, "-exitfilechecktime" ) == 0 ) 
        {                                        
        if (nextarg)
          {
          skip_next = 1;
          /* obsolete */
          }
        }
      else if ( strcmp( thisarg, "-c" ) == 0 || strcmp( thisarg, "-cputype" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting cputype to %d\n", (int)cputype);
            }
          else
            {
            cputype = (s32) atoi( nextarg );
            inimissing = 0; // Don't complain if the inifile is missing
            }
          }
        }
      else if ( strcmp( thisarg, "-nice" ) == 0 ) // Nice level
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting priority to %u\n", priority );
            }
          else
            {
            #ifdef OLDNICENESS
            niceness = (s32) atoi( nextarg );
            #else          
            priority = (s32) atoi( nextarg );
            priority = ((priority==2)?(8):((priority==1)?(4):(0)));
            #endif
            inimissing = 0; // Don't complain if the inifile is missing
            }
          }
        }
      else if ( strcmp( thisarg, "-priority" ) == 0 ) // Nice level
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting priority to %u\n", priority );
            }
          else
            {
            #ifndef OLDNICENESS
            priority = (s32) atoi( nextarg );
            inimissing = 0; // Don't complain if the inifile is missing
            #endif
            }
          }
        }
      else if ( strcmp( thisarg, "-h" ) == 0 || strcmp( thisarg, "-until" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          int isuntil = (strcmp( thisarg, "-until" ) == 0);
          int h=0, m=0, pos, isok = 0, dotpos=0;
          if (isdigit(*nextarg))
            {
            isok = 1;
            for (pos = 0; nextarg[pos] != 0; pos++)
              {
              if (!isdigit(nextarg[pos]))
                {
                if (dotpos!=0 || (nextarg[pos]!=':' && nextarg[pos]!='.'))
                  {
                  isok = 0;
                  break;
                  }
                dotpos=pos;
                }
              }
            if (isok)
              {
              if ((h = atoi( nextarg )) < 0)
                isok = 0;
              else if (isuntil && h > 23)
                isok = 0;
              else if (dotpos == 0 && isuntil)
                isok = 0;
              else if (dotpos != 0 && strlen(nextarg+dotpos+1) != 2)
                isok = 0;
              else if (dotpos != 0 && ((m = atoi(nextarg+dotpos+1)) > 59))
                isok = 0;
              }
            }
          if (run_level!=0)
            {
            if (logging_is_initialized)
              {
              if (!isok)
                LogScreenRaw("%s option is invalid and was ignored.\n",thisarg);
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
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
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
          else if ( (blockcount = atoi( nextarg )) < 0)
            blockcount = -1;
          }
        }
      else if ( strcmp( thisarg, "-numcpu" ) == 0 ) // Override the number of cpus
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            }
          else
            {
            numcpu = (s32) atoi(nextarg);
            inimissing = 0; // Don't complain if the inifile is missing
            }
          }
        }
      else if ( strcmp( thisarg, "-ckpoint" ) == 0 || strcmp( thisarg, "-ckpoint2" ) == 0 )
        {
        if (nextarg)
          {
          skip_next = 1;
          int conid = (( strcmp( thisarg, "-ckpoint2" ) == 0 ) ? (1) : (0));
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting %s checkpoint file to %s\n", 
                    ((conid)?("DES"):("RC5")), checkpoint_file[conid] );
            }
          else
            {
            strcpy(checkpoint_file[conid], nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-cktime" ) == 0 )
        {
        if (nextarg)
          {
          skip_next = 1;
          /* obsolete */
          }
        }
      else if ( strcmp( thisarg, "-pausefile" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting pause file to %s\n",pausefile);
            }
          else
            {
            strcpy(pausefile, nextarg );
            }
          }
        }
      else if ( strcmp( thisarg, "-blsize" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Setting preferred blocksize to 2^%d\n",preferred_blocksize);
            }
          else
            {
            preferred_blocksize = (s32) atoi(nextarg);
            }
          }
        }
      else if ( strcmp(thisarg, "-processdes" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          if (run_level!=0)
            {
            if (logging_is_initialized)
              LogScreenRaw("Client will now%s compete in DES contest(s).\n",
                      ((preferred_contest_id==0)?(" NOT"):("")) );
            }
          else
            {
            preferred_contest_id = (s32) atoi(nextarg);
            if (preferred_contest_id != 0)
              preferred_contest_id = 1;
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
        ; //nothing - handled in next loop
        havemode = 1;
        }
      else if ( strcmp( thisarg, "-forceunlock" ) == 0 ) 
        {
        ; //f'd up "mode" - handled in next loop
        skip_next = 1;
        havemode = 1;
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
      }
    ValidateConfig();
    }
        
  //-----------------------------------
  // In the final loop we parse the "modes".
  //-----------------------------------

  if (!terminate_app && havemode && run_level == 0)
    {          
    for (pos = 1;pos<argc;pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      if (thisarg && *thisarg=='-' && thisarg[1]=='-')
        thisarg++;
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;
  
      if ( thisarg == NULL )
        ; //nothing
      else if (*thisarg == 0)
        ; //nothing
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
          
          if ( strcmp( thisarg, "-fetch" ) == 0 )           
            do_mode = MODEREQ_FETCH;
          else if ( strcmp( thisarg, "-flush" ) == 0 )      
            do_mode = MODEREQ_FLUSH;
          else if ( strcmp( thisarg, "-forcefetch" ) == 0 )
            do_mode = MODEREQ_FETCH | MODEREQ_FFORCE;
          else if ( strcmp( thisarg, "-forceflush" ) == 0 )
            do_mode = MODEREQ_FLUSH | MODEREQ_FFORCE;
          else /* ( strcmp( thisarg, "-update" ) == 0) */
            do_mode = MODEREQ_FETCH | MODEREQ_FLUSH | MODEREQ_FFORCE;
          
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
        if (!inimissing && nextarg)
          {
          quietmode = 0;
          skip_next = 1;
          ModeReqClear(-1); //clear all - only do -forceunlock
          ModeReqSet(MODEREQ_UNLOCK);
          ModeReqSetArg(MODEREQ_UNLOCK,(void *)nextarg);
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

  if (inimissing  && run_level == 0)
    {
    quietmode = 0;
    ModeReqSet( MODEREQ_CONFIG );
    }
  #if ((CLIENT_OS == OS_DEC_UNIX)    || (CLIENT_OS == OS_HPUX)    || \
       (CLIENT_OS == OS_QNX)         || (CLIENT_OS == OS_OSF1)    || \
       (CLIENT_OS == OS_BSDI)        || (CLIENT_OS == OS_SOLARIS) || \
       (CLIENT_OS == OS_IRIX)        || (CLIENT_OS == OS_SCO)     || \
       (CLIENT_OS == OS_LINUX)       || (CLIENT_OS == OS_NETBSD)  || \
       (CLIENT_OS == OS_UNIXWARE)    || (CLIENT_OS == OS_DYNIX)   || \
       (CLIENT_OS == OS_MINIX)       || (CLIENT_OS == OS_MACH10)  || \
       (CLIENT_OS == OS_AIX)         || (CLIENT_OS == OS_AUX)     || \
       (CLIENT_OS == OS_OPENBSD)     || (CLIENT_OS == OS_SUNOS)   || \
       (CLIENT_OS == OS_ULTRIX)      || (CLIENT_OS == OS_DGUX))
  else if (run_level == 0 && (ModeReqIsSet(-1) == 0) && 
          quietmode && isatty(fileno(stdout)))
    {
    terminate_app = 1; //always return !0
    int i,quoteit;
    char cmd_buffer[1024];
    char redir[] = ">/dev/null &";
    strcpy(cmd_buffer,argv[0]);
    for (i=1;i<=argc;i++)
      {
      if (argv[i])
        {
        if ((strlen(cmd_buffer)+strlen(argv[i])+4) >= 
            ((sizeof(cmd_buffer)-sizeof(redir))+2))
          break;
        quoteit=(argv[i][0]==0 || strchr(argv[i],' ')!=NULL || 
                                  strchr(argv[i],'\t')!=NULL);
        strcat(cmd_buffer," ");
        if (quoteit) strcat(cmd_buffer,"\"");
        strcat(cmd_buffer,argv[i]);
        if (quoteit) strcat(cmd_buffer,"\"");
        }
      }    
    strcat(cmd_buffer,redir);
    if (system(cmd_buffer) != 0)
      {
      sprintf(cmd_buffer,"Unable to start quiet/hidden. "
              "Use \"%s\" instead.",redir);
      ConOutErr(cmd_buffer);
      }
    }
  #endif
 
  
  if (retcodeP) 
    *retcodeP = 0;
  return terminate_app;
}

