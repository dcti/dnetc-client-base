// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cmdline.cpp,v $
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
return "@(#)$Id: cmdline.cpp,v 1.92 1998/11/04 21:28:19 cyp Exp $"; }
#endif

#include "cputypes.h"
#include "client.h"    // Client class
#include "baseincs.h"  // basic (even if port-specific) #includes
#include "logstuff.h"  // Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "pathwork.h"  // InitWorkingDirectoryFromSamplePaths();
#include "buffwork.h"  // UnlockBuffer()
#include "modereq.h"   // get/set/clear mode request bits

/* ------------------------------------------------------------------------
 * runlevel == 0 = pre-anything    (-quiet, -ini, -guistart etc done here)
 *          >= 1 = post-readconfig (override ini options)
 *          == 2 = run "modes"
 *
 * Sequence of events:
 *
 *   ParseCommandLine( 0, argc, argv, NULL, NULL, 0 );
 *   if ( InitializeLogging() == 0 ) //let -quiet take affect
 *     {
 *     inimissing = ReadConfig();
 *     if (ParseCommandLine( X, argc, argv, &inimissing, &retcode, x )==0)
 *       {                   |                                     
 *       if (inimissing)     `-- X==2 for OS_xxx that do "modes", 1 for others
 *         { 
 *         Configure() ...  
 *         }
 *       else 
 *         {
 *         if ( RunStartup() == 0 )
 *           {
 *           ValidateConfig()
 *           Run();
 *           RunShutdown();
 *           }
 *         }
 *       }
 *     DeinitializeLogging();
 *     }
 *   
 *------------------------------------------------------------------------ */

int Client::ParseCommandline( int runlevel, int argc, const char *argv[], 
                              int *retcodeP, int logging_is_initialized )
{
  int inimissing, pos, skip_next = 0, do_break = 0, retcode = 0;
  const char *thisarg, *nextarg;

  //---------------------------------------
  //first handle the options that affect option handling
  //--------------------------------------

  if (runlevel >= 0) // this is only to protect against an invalid runlevel
    {
    int quietmode_override = -1;
    
    inifilename[0] = 0; //so we know when it changes
    if (runlevel == 0)
      ModeReqClear(-1); // clear all mode request bits

    for (pos = 1;((!do_break) && (pos<argc)); pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;
  
      if ( strcmp( thisarg, "-guiriscos" ) == 0) 
        {                       
        #if (CLIENT_OS == OS_RISCOS)
        guiriscos=1;
        #endif
        }
      else if ( strcmp( thisarg, "-guirestart" ) == 0) 
        {                 // See if are restarting (hence less banners wanted)
        #if (CLIENT_OS == OS_RISCOS)
        guirestart=1;
        #endif
        }
      else if ( strcmp(thisarg, "-install" ) == 0)
        {
        #if (CLIENT_OS == OS_WIN32)
          win32CliInstallService(0);
          do_break = -1;
          retcode = 0;
        #endif
        #if (CLIENT_OS == OS_OS2)
          os2CliInstallClient(0);
          do_break = -1;
          retcode = 0;
        #endif
        }
      else if ( strcmp(thisarg, "-uninstall" ) == 0)
        {
        #if (CLIENT_OS == OS_OS2)
          os2CliUninstallClient(0);
          do_break = -1;
          retcode = 0;
        #endif
        #if (CLIENT_OS == OS_WIN32) 
          win32CliUninstallService(0);
          do_break = -1;
          retcode = 0;
        #endif
        }
      else if ( strcmp(thisarg, "-ini" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1; 
          strcpy( inifilename, nextarg );
          }
        }
      else if ( strcmp( thisarg, "-hide" ) == 0 ||   
                strcmp( thisarg, "-quiet" ) == 0 )
        {
        quietmode_override = 1;
        }
      else if ( strcmp( thisarg, "-noquiet" ) == 0 )      
        {
        quietmode_override = 0;
        }
      } //for ... next

    if (!inifilename[0] && !do_break) //we don't have an ini filename
      {
      // determine the filename of the ini file
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
    
    if (!do_break)
      {  
      InitWorkingDirectoryFromSamplePaths( inifilename, argv[0] );

      if (runlevel == 0)
        {
        if ( ReadConfig() != 0)
          ModeReqSet( MODEREQ_CONFIG );
        if ( quietmode_override >= 0)
          quietmode = quietmode_override;
        }
      } //if (!do_break)
    } //if (runlevel >= 0)

  inimissing = (ModeReqIsSet( MODEREQ_CONFIG ) != 0);

  //---------------------------------------
  // handle the other options 
  //--------------------------------------

  static const char *ignoreX[]={ /* options handled in other loops */
  "+ini","-trace","-guiriscos","-guirestart","-hide","-quiet","-noquiet",
  "-ident","-cpuinfo","-test","-config","-install","-uninstall","+forceunlock",
  "-benchmark2rc5","-benchmark2des","-benchmark2","-benchmarkrc5","-benchmarkdes",
  "-benchmark","-fetch","-forcefetch","-flush","-forceflush","-update","" };

  if (runlevel >= 1 && !do_break)
    {
    for (pos = 1;((!do_break) && (pos<argc)); pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
  
      skip_next = -1;
      for (unsigned int i=0;i<(sizeof(ignoreX)/sizeof(ignoreX[1]));i++)
        {
        if (*thisarg == '-' && strcmp( thisarg+1, ignoreX[i]+1 )==0 )
          {
          skip_next++;
          if (*(ignoreX[i])=='+' && nextarg)
            skip_next++;
          break;
          }
        }
      if (skip_next != -1) //not found in ignoreX
        continue;
      skip_next = 0;
      
      if ( strcmp(thisarg, "-percentoff" ) == 0)          
        {
        percentprintingoff = 1;
        }
      else if ( strcmp( thisarg, "-nofallback" ) == 0 )   
        {
        nofallback = 1;
        }
      else if ( strcmp( thisarg, "-lurk" ) == 0 )
        {                                
        #if defined(LURK)
        dialup.lurkmode=1;               // Detect modem connections
        #endif
        }
      else if ( strcmp( thisarg, "-lurkonly" ) == 0 )
        {               
        #if defined(LURK)
        dialup.lurkmode=2;              // Only connect when modem connects
        #endif
        }
      else if ( strcmp( thisarg, "-noexitfilecheck" ) == 0 )
        {                          
        noexitfilecheck=1;             // Change network timeout
        }
      else if ( strcmp( thisarg, "-runoffline" ) == 0 ) 
        {
        offlinemode=1;                // Run offline
        }
      else if ( strcmp( thisarg, "-runbuffers" ) == 0 ) 
        {
        offlinemode=2;                // Run offline & exit when buffer empty
        }
      else if ( strcmp( thisarg, "-run" ) == 0 ) 
        {
        offlinemode=0;                // Run online
        }
      else if ( strcmp( thisarg, "-nodisk" ) == 0 ) 
        {
        nodiskbuffers=1;              // No disk buff-*.rc5 files.
        strcpy(checkpoint_file[0],"none");
        strcpy(checkpoint_file[1],"none");
        #ifdef DONT_USE_PATHWORK
        strcpy(ini_checkpoint_file[0],"none");
        strcpy(ini_checkpoint_file[1],"none");
        #endif
        }
      else if ( strcmp(thisarg, "-frequent" ) == 0)
        {
        connectoften=1;
        if (logging_is_initialized)
          LogScreenRaw("Setting connections to frequent\n");
        }
      else if ( strcmp(thisarg, "-nommx" ) == 0)
        {
        #if (CLIENT_CPU == CPU_X86) && (defined(MMX_BITSLICER) || defined(MMX_RC5))
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
          if ( atoi( nextarg ) > 0)
            {
            inimissing = 0; // Don't complain if the inifile is missing
            int conid = (( strcmp( thisarg, "-b2" ) == 0 ) ? (1) : (0));
            outthreshold[conid] = inthreshold[conid] = (s32) atoi( nextarg );
            if (logging_is_initialized)
              LogScreenRaw("Setting %s buffer thresholds to %u\n",
                   ((conid)?("DES"):("RC5")), (unsigned int)inthreshold[conid] );
            }
          }
        }
      else if ( strcmp( thisarg, "-bin" ) == 0 || strcmp( thisarg, "-bin2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          if ( atoi( nextarg ) > 0)
            {
            inimissing = 0; // Don't complain if the inifile is missing
            int conid = (( strcmp( thisarg, "-bin2" ) == 0 ) ? (1) : (0));
            inthreshold[conid] = (s32) atoi( nextarg );
            if (logging_is_initialized)
              LogScreenRaw("Setting %s in-buffer threshold to %u\n",
                    ((conid)?("DES"):("RC5")), (unsigned int)inthreshold[conid] );
            }
          }
        }
      else if ( strcmp( thisarg, "-bout" ) == 0 || strcmp( thisarg, "-bout2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          if ( atoi( nextarg ) > 0)
            {
            inimissing = 0; // Don't complain if the inifile is missing
            int conid = (( strcmp( thisarg, "-bout2" ) == 0 ) ? (1) : (0));
            outthreshold[conid] = (s32) atoi( nextarg );
            if (logging_is_initialized)
              LogScreenRaw("Setting %s out-buffer threshold to %u\n",
                  ((conid)?("DES"):("RC5")), (unsigned int)outthreshold[conid] );
            }
          }
        }
      else if ( strcmp( thisarg, "-in" ) == 0 || strcmp( thisarg, "-in2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          int conid = (( strcmp( thisarg, "-in2" ) == 0 ) ? (1) : (0));
          in_buffer_file[conid][sizeof(in_buffer_file[0])-1]=0;
          strncpy(in_buffer_file[conid], nextarg, sizeof(in_buffer_file[0]) );
          #ifdef DONT_USE_PATHWORK
            strcpy(ini_in_buffer_file[conid], in_buffer_file[conid]);
          #endif
          if (logging_is_initialized)
            LogScreenRaw("Setting %s in-buffer file to %s\n",
                    ((conid)?("DES"):("RC5")), in_buffer_file[conid] );
          }
        }
      else if ( strcmp( thisarg, "-out" ) == 0 || strcmp( thisarg, "-out2")==0)
        {                          
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          int conid = (( strcmp( thisarg, "-out2" ) == 0 ) ? (1) : (0));
          out_buffer_file[conid][sizeof(out_buffer_file[0])-1]=0;
          strncpy(out_buffer_file[conid], nextarg, sizeof(out_buffer_file[0]) );
          #ifdef DONT_USE_PATHWORK
            strcpy(ini_out_buffer_file[conid], out_buffer_file[conid]);
          #endif
          if (logging_is_initialized)
            LogScreenRaw("Setting %s out-buffer file to %s\n",
                    ((conid)?("DES"):("RC5")), out_buffer_file[conid] );
          }
        }
      else if ( strcmp( thisarg, "-u" ) == 0 ) // UUE/HTTP Mode
        {                    
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          uuehttpmode = (s32) atoi( nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting uue/http mode to %u\n",(unsigned int)uuehttpmode);
          }
        }
      else if ( strcmp( thisarg, "-a" ) == 0 ) // Override the keyserver name
        {                    
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          strcpy( keyproxy, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting keyserver to %s\n", keyproxy );
          }
        }
      else if ( strcmp( thisarg, "-p" ) == 0 ) // UUE/HTTP Mode
        {                    
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          keyport = (s32) atoi( nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting keyserver port to %u\n",(unsigned int)keyport);
          }
        }
      else if ( strcmp( thisarg, "-ha" ) == 0 ) // Override the http proxy name
        {
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          strcpy( httpproxy, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting SOCKS/HTTP proxy to %s\n", httpproxy);
          }
        }
      else if ( strcmp( thisarg, "-hp" ) == 0 ) // Override the socks/http proxy port
        {                    
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          httpport = (s32) atoi( nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting SOCKS/HTTP proxy port to %u\n",(unsigned int)httpport);
          }
        }
      else if ( strcmp( thisarg, "-l" ) == 0 ) // Override the log file name
        {
        if (nextarg)
          {
          skip_next = 1;
          strcpy( logname, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting log file to %s\n", logname );
          }
        }
      else if ( strcmp( thisarg, "-smtplen" ) == 0 ) // Override the mail message length
        {
        if (nextarg)
          {
          skip_next = 1;
          messagelen = (s32) atoi(nextarg);
          if (logging_is_initialized)
            LogScreenRaw("Setting Mail message length to %s\n", nextarg );
          }
        }
      else if ( strcmp( thisarg, "-smtpport" ) == 0 ) // Override the smtp port for mailing
        {
        if (nextarg)
          {
          skip_next = 1;
          smtpport = (s32) atoi(nextarg);
          if (logging_is_initialized)
            LogScreenRaw("Setting smtp port to %s\n", nextarg);
          }
        }
      else if ( strcmp( thisarg, "-smtpsrvr" ) == 0 ) // Override the smtp server name
        {
        if (nextarg)
          {
          skip_next = 1;
          strcpy( smtpsrvr, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting SMTP relay host to %s\n", smtpsrvr);
          }
        }
      else if ( strcmp( thisarg, "-smtpfrom" ) == 0 ) // Override the smtp source id
        {
        if (nextarg)
          {
          skip_next = 1;
          strcpy( smtpfrom, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting mail 'from' address to %s\n", smtpfrom );
          }
        }
      else if ( strcmp( thisarg, "-smtpdest" ) == 0 ) // Override the smtp source id
        {
        if (nextarg)
          {
          skip_next = 1;
          strcpy( smtpdest, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting mail 'to' address to %s\n", smtpdest );
          }
        }
      else if ( strcmp( thisarg, "-e" ) == 0 )     // Override the email id
        {
        if (nextarg)
          {
          skip_next = 1;
          strcpy( id, nextarg );
          inimissing = 0; // Don't complain if the inifile is missing
          if (logging_is_initialized)
            LogScreenRaw("Setting distributed.net ID to %s\n", id );
          }
        }
      else if ( strcmp( thisarg, "-nettimeout" ) == 0 ) // Change network timeout
        {
        if (nextarg)
          {
          skip_next = 1;
          int tmp = atoi( nextarg );
          nettimeout = ((tmp <= 5)?(5):((tmp>=300)?(300):(tmp)));
          if (logging_is_initialized)
            LogScreenRaw("Setting network timeout to %u\n", 
                                             (unsigned int)(nettimeout));
          }
        }
      else if ( strcmp( thisarg, "-exitfilechecktime" ) == 0 ) 
        {                                        
        if (nextarg)
          {
          skip_next = 1;
          exitfilechecktime = atoi(nextarg);
          if (exitfilechecktime < 5) exitfilechecktime=5;
          else if (exitfilechecktime > 600) exitfilechecktime=600;
          if (logging_is_initialized)
            LogScreenRaw("Setting exitfile check time to %u\n", 
                (unsigned int)(exitfilechecktime) );
          }
        }
      else if ( strcmp( thisarg, "-c" ) == 0)      // set cpu type
        {
        if (nextarg)
          {
          skip_next = 1;
          cputype = (s32) atoi( nextarg );
          inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      else if ( strcmp( thisarg, "-nice" ) == 0 ) // Nice level
        {
        if (nextarg)
          {
          skip_next = 1;
#ifdef OLDNICENESS
          niceness = (s32) atoi( nextarg );
#else          
          priority = (s32) atoi( nextarg );
          priority = ((priority==2)?(8):((priority==1)?(4):(0)));
#endif
          inimissing = 0; // Don't complain if the inifile is missing
          }
        }
      else if ( strcmp( thisarg, "-priority" ) == 0 ) // Nice level
        {
        if (nextarg)
          {
          skip_next = 1;
#ifndef OLDNICENESS
          priority = (s32) atoi( nextarg );
          inimissing = 0; // Don't complain if the inifile is missing
#endif
          }
        }
      else if ( strcmp( thisarg, "-h" ) == 0 ) // Hours to run
        {
        if (nextarg)
          {
          skip_next = 1;
          minutes = (s32) (60. * atol( nextarg ));
          if (logging_is_initialized)
            LogScreenRaw("Setting time limit to %ul minutes\n",
                                                       (unsigned long)(minutes));
          }
        }
      else if ( strcmp( thisarg, "-n" ) == 0 ) // Blocks to complete in a run
        {
        if (nextarg)
          {
          skip_next = 1;
          if ( (blockcount = atoi( nextarg )) < 0)
            blockcount = 0;
          if (logging_is_initialized)
            LogScreenRaw("Setting block completion limit to %u\n",
                                                    (unsigned int)blockcount);
          }
        }
      else if ( strcmp( thisarg, "-until" ) == 0 ) // Exit time
        {
        if (nextarg)
          {
          skip_next = 1;
          time_t timenow = time( NULL );
          struct tm *gmt = localtime(&timenow );
          minutes = atoi( nextarg );
          minutes = (int)( ( ((int)(minutes/100))*60 + (minutes%100) ) - 
                                       ((60. * gmt->tm_hour) + gmt->tm_min));
          if (minutes<0) minutes += 24*60;
          if (minutes<0) minutes = 0;
          if (logging_is_initialized)
            LogScreenRaw("Setting time limit to %d minutes\n",minutes);
          sprintf(hours,"%u.%02u",(unsigned int)(minutes/60),
                                    (unsigned int)(minutes%60));
          //was sprintf(hours,"%f",minutes/60.); -> "0.000000" which looks silly
          //and could cause a NetWare 3.x client to raise(SIGABRT)
          }
        }
      else if ( strcmp( thisarg, "-numcpu" ) == 0 ) // Override the number of cpus
        {
        if (nextarg)
          {
          skip_next = 1;
          numcpu = (s32) atoi(nextarg);
          inimissing = 0; // Don't complain if the inifile is missing
          //LogScreenRaw("Configuring for %s CPUs\n",Argv[i+1]);
          //Message appears in SelectCore()
          }
        }
      else if ( strcmp( thisarg, "-ckpoint" ) == 0 || strcmp( thisarg, "-ckpoint2" ) == 0 )
        {
        if (nextarg)
          {
          skip_next = 1;
          inimissing = 0; // Don't complain if the inifile is missing
          int conid = (( strcmp( thisarg, "-ckpoint2" ) == 0 ) ? (1) : (0));
          strcpy(checkpoint_file[conid], nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting RC5 checkpoint file to %s\n", nextarg );
          }
        }
      else if ( strcmp( thisarg, "-cktime" ) == 0 )
        {
        if (nextarg)
          {
          skip_next = 1;
          int tmp = atoi(nextarg);
          inimissing = 0; // Don't complain if the inifile is missing
          checkpoint_min=((tmp <=2)?(2):(tmp));
          if (logging_is_initialized)
            LogScreenRaw("Setting checkpointing to %u minutes\n", (unsigned int)(checkpoint_min));
          }
        }
      else if ( strcmp( thisarg, "-pausefile" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          strcpy(pausefile, nextarg );
          if (logging_is_initialized)
            LogScreenRaw("Setting pause file to %s\n",pausefile);
          }
        }
      else if ( strcmp( thisarg, "-blsize" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          preferred_blocksize = (s32) atoi(nextarg);
          if (preferred_blocksize < 28) preferred_blocksize = 28;
          if (preferred_blocksize > 31) preferred_blocksize = 31;
          if (logging_is_initialized)
            LogScreenRaw("Setting preferred blocksize to 2^%d\n",preferred_blocksize);
          }
        }
      else if ( strcmp(thisarg, "-processdes" ) == 0)
        {
        if (nextarg)
          {
          skip_next = 1;
          preferred_contest_id = (s32) atoi(nextarg);
          if (preferred_contest_id != 0)
            preferred_contest_id = 1;
          if (logging_is_initialized)
            LogScreenRaw("Client will now%s compete in DES contest(s).\n",
                      ((preferred_contest_id==0)?(" NOT"):("")) );
          }
        }
      else if (runlevel > 0)
        {
        quietmode = 0;
        DisplayHelp(thisarg);
        retcode = 1;
        do_break = 1;
        }
      }
    if (!do_break)
      ValidateConfig();
    }
      
  //---------------------------------------
  // handle the run modes
  //--------------------------------------
  
  if (runlevel >= 2 && !do_break)
    {
    for (pos = 1;((!do_break) && (pos<argc)); pos+=(1+skip_next))
      {
      thisarg = argv[pos];
      nextarg = ((pos < (argc-1))?(argv[pos+1]):(NULL));
      skip_next = 0;
    
      if (( strcmp( thisarg, "-fetch" ) == 0 ) || 
          ( strcmp( thisarg, "-forcefetch" ) == 0 ) || 
          ( strcmp( thisarg, "-flush"      ) == 0 ) || 
          ( strcmp( thisarg, "-forceflush" ) == 0 ) || 
          ( strcmp( thisarg, "-update"     ) == 0 ))
        {
        if (!inimissing)
          {
          quietmode = 0;
          do_break = 1;
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
          }
        }
      else if ( strcmp(thisarg, "-ident" ) == 0)
        {
        quietmode = 0;
        do_break = 1;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -ident
        ModeReqSet( MODEREQ_IDENT );
        retcode = 0;
        }
      else if ( strcmp( thisarg, "-cpuinfo" ) == 0 )
        {
        quietmode = 0;
        do_break = 1;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -cpuinfo
        ModeReqSet( MODEREQ_CPUINFO );
        retcode = 0; //and break out of loop
        }
      else if ( strcmp( thisarg, "-test" ) == 0 )
        {
        quietmode = 0;
        do_break = 1;
        inimissing = 0; // Don't complain if the inifile is missing
        ModeReqClear(-1); //clear all - only do -test
        ModeReqSet( MODEREQ_TEST );
        SelectCore( 1 /* quietly */ );
        }
      else if (strncmp( thisarg, "-benchmark", 10 ) == 0)
        {
        quietmode = 0;
        do_break = 1;
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
        SelectCore( 1 /* quietly */ );
        }
      else if ( strcmp( thisarg, "-forceunlock" ) == 0 )
        {
        if (!inimissing)
          {
          quietmode = 0;
          do_break = 1;
          retcode = -1;
          ModeReqClear(-1); //clear all - only do -forceunlock
          if (nextarg)
            retcode = UnlockBuffer(nextarg);
          }
        }
      else if ( strcmp( thisarg, "-config" ) == 0 )
        {
        quietmode = 0;
        do_break = 1;
        retcode = 0;
        ModeReqClear(-1); //clear all - only do -config
        inimissing = 1; //this should force main to run config
        }
      }
    }

  if (inimissing)
    ModeReqSet( MODEREQ_CONFIG );
  if (!do_break && runlevel >= 1)
    do_break = ModeReqIsSet(-1);
  if (retcodeP && do_break) 
    *retcodeP = retcode;
  return do_break;
}
