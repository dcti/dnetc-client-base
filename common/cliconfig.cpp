// Copyright distributed.net 1997 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "client.h"


// --------------------------------------------------------------------------

#define OPTION_COUNT    27

char * options[OPTION_COUNT][4] = //name, desc, default, additional comments
{
//0
  { "id", "Email to report as", "rc5@distributed.net", "(64 characters max)"},
//1
  { "threshold", "RC5 Blocks to Buffer [in:out]", "10", "Input (max 1000)?"},
//2
  { "threshold2", "DES Blocks to Buffer [in:out]", "10", "Input (max 1000)?"},
//3
  { "count", "Blocks to complete in run", "0", "(0 = no limit)"},
//4
  { "hours", "Hours to complete in a run", "0.0", "(0 = no limit)"},
//5
  { "timeslice", "Keys per timeslice - for Macs, Win16, RISC OS, etc",
#if (CLIENT_OS == OS_WIN16)
    "200",
#else
    "65536",
#endif
    "(0 = default timeslicing)\n"
    "DO NOT TOUCH this unless you know what you're doing!!!"},
//6
  { "niceness", "Level of niceness to run at", "0",
    "\n  mode 0) (recomended) Very nice, should not interfere with any other process\n"
    "  mode 1) Nice, runs with slightly higher priority than idle processes\n"
    "          Same as mode 0 in OS/2 and Win32\n"
    "  mode 2) Normal, runs with same priority as normal user processes\n"},
//7
  { "logname", "File to log to", "", "(128 characters max, blank = no log)\n"},
//8
  { "firemode", "Network communication mode", "1",
    "\n  mode 1) I can communicate freely to the Internet on all ports.\n"
    "  mode 2) I can communicate freely on telnet ports\n"
    "  mode 3) I can communicate freely on telnet ports, but need uuencoding\n"
    "  mode 4) I have a local HTTP proxy that I can go through\n"
    "  mode 5) Let me specify custom network settings (expert mode)\n"},
//9
  { "keyproxy", "Preferred KeyServer Proxy\n    ",
    "us.v27.distributed.net", "(DNS or IP address)\n" },
//10
  { "keyport", "Preferred KeyServer Port", "2064", "(TCP/IP port on preferred proxy)"},
//11
  { "httpproxy", "Local HTTP/SOCKS proxy address\n    ",
    "wwwproxy.corporate.com", "(DNS or IP address)\n" },
//12
  { "httpport", "Local HTTP/SOCKS proxy port", "80", "(TCP/IP port on http proxy)"},
//13
  { "uuehttpmode", "UUE/HTTP/SOCKS mode", "0",
      "\n  mode 0) No special encoding\n"
      "  mode 1) UUE encoding (telnet proxies)\n"
      "  mode 2) HTTP encoding\n"
      "  mode 3) HTTP+UUE encoding\n"
      "  mode 4) SOCKS4 proxy\n"
      "  mode 5) SOCKS5 proxy\n" },
//14
  { "httpid", "HTTP/SOCKS proxy userid/password", "", "(Enter userid (. to reset it to empty) )"},
#if (CLIENT_CPU == CPU_X86)
//15
  { "cputype", "Optimize performance for CPU type", "-1",
      "\n   mode -1) Autodetect\n"
      "   mode 0) Intel Pentium, Intel Pentium MMX, Cyrix 486/5x86/MediaGX\n"
      "   mode 1) Intel 80386, Intel 80486\n"
      "   mode 2) Intel Pentium Pro, Intel Pentium II\n"
      "   mode 3) AMD 486, Cyrix 6x86/6x86MX/M2\n"
      "   mode 4) AMD K5\n"
      "   mode 5) AMD K6\n"
    },
#elif (CLIENT_CPU == CPU_STRONGARM)
  { "cputype", "Optimise performance for CPU type", "-1",
      "\n   mode -1) Autodetect\n"
      "   mode 0) ARM\n"
      "   mode 1) StrongARM\n"},

#elif (CLIENT_CPU == CPU_POWERPC && (CLIENT_OS == OS_LINUX || CLIENT_OS == OS_AIX))
// 15
  { "cputype", "Optimize performance for CPU type", "-1",
      "\n   mode -1) Autodetect\n"
      "   mode 0) PowerPC 601\n"
      "   mode 1) PowerPC 603/604/750\n"},
#else
//15
  { "cputype", "CPU type...not applicable in this client", "-1", "(default -1)"},
#endif
//16
  { "messagelen", "Message Mailing (bytes)", "0", "(0=no messages mailed.  10000 recommended.  125000 max.)\n"},
//17
  { "smtpsrvr", "SMTP Server", "your.smtp.server", "(128 characters max)"},
//18
  { "smtpport", "SMTP Port", "25", "(SMTP port on mail server -- default 25)"},
//19
  { "smtpfrom", "Mail ID that logs will be mailed from", "RC5notify", "\n(Some servers require this to be a real address)\n"},
//20
  { "smtpdest", "Mail ID that logs will be sent to\n    ", "you@your.site", "\n(Full name and site eg: you@your.site.  Comma delimited list permitted)\n"},
//21
#if ((CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_OS2))
  { "numcpu", "Number of CPUs in this machine", "-1 (autodetect)", "\n"},
#else
  { "numcpu", "Number of CPUs in this machine", "1", "\n"},
#endif
//22
  { "checkpointfile", "RC5 Checkpoint information filename","none","\n(Non-shared file required.  "
#if (CLIENT_OS == OS_RISCOS)
    "ckpoint/rc5"
#else
    "ckpoint.rc5"
#endif
    " recommended.  'none' to disable)\n"},
//23
  { "checkpointfile2", "DES Checkpoint information filename","none","\n(Non-shared file required.  "
#if (CLIENT_OS == OS_RISCOS)
    "ckpoint/des"
#else
    "ckpoint.des"
#endif
    " recommended.  'none' to disable)\n"},
//24
  { "randomprefix", "High order byte of random blocks","100","Do not change this"},
//25
  { "preferredblocksize", "Preferred Block Size (2^28 through 2^31) ","30","28 -- 31"},
//26
  { "preferredcontest", "Preferred Contest (1=RC5, 2=DES) ","2","2 recommended"}
};

#define CONF_ID 0
#define CONF_THRESHOLD 1
#define CONF_THRESHOLD2 2
#define CONF_COUNT 3
#define CONF_HOURS 4
#define CONF_TIMESLICE 5
#define CONF_NICENESS 6
#define CONF_LOGNAME 7
#define CONF_FIREMODE 8
#define CONF_KEYPROXY 9
#define CONF_KEYPORT 10
#define CONF_HTTPPROXY 11
#define CONF_HTTPPORT 12
#define CONF_UUEHTTPMODE 13
#define CONF_HTTPID 14
#define CONF_CPUTYPE 15
#define CONF_MESSAGELEN 16
#define CONF_SMTPSRVR 17
#define CONF_SMTPPORT 18
#define CONF_SMTPFROM 19
#define CONF_SMTPDEST 20
#define CONF_NUMCPU 21
#define CONF_CHECKPOINT 22
#define CONF_CHECKPOINT2 23
#define CONF_RANDOMPREFIX 24
#define CONF_PREFERREDBLOCKSIZE 25
#define CONF_PREFERREDCONTEST 26

// --------------------------------------------------------------------------

s32 Client::Configure( void )
{
  char parm[128],parm2[128];
  s32 choice;
  char *p;
#if (CLIENT_OS == OS_WIN32) && defined(MULTITHREAD)
  SYSTEM_INFO systeminfo;
#elif (CLIENT_OS == OS_BEOS)
  system_info the_info;
#endif

  while ( 1 )
  {
    // display menu
    printf("\nCLIENT CONFIG MENU\n"
           "------------------\n");
    char threshold[2][64];
    sprintf(threshold[0], "%d:%d", (int) inthreshold[0], (int) outthreshold[0]);
    sprintf(threshold[1], "%d:%d", (int) inthreshold[1], (int) outthreshold[1]);

    IniStringList list( id, &threshold[0], &threshold[1], blockcount, hours, timeslice, niceness,
                        logname, firemode, keyproxy, keyport,
                        httpproxy, httpport, uuehttpmode, httpid, cputype,
                        messagelen,smtpsrvr,smtpport,smtpfrom,smtpdest,numcpu,
                        checkpoint_file[0], checkpoint_file[1], randomprefix,
                        preferred_blocksize, preferred_contest_id);
    for ( choice = 0; choice < OPTION_COUNT; choice++ )
    {
      if ((choice >= 0 && choice < 1+CONF_FIREMODE) ||
          (choice == CONF_KEYPROXY && firemode >= 4) ||
          (choice == CONF_KEYPORT && firemode >= 5 && !CheckForcedKeyport()) ||
          ((choice == CONF_HTTPPROXY || choice == CONF_HTTPPORT || choice == CONF_HTTPID) &&
              (uuehttpmode > 1)) ||
          (choice == CONF_UUEHTTPMODE && firemode >= 5)
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_STRONGARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
          || (choice == CONF_CPUTYPE)
#endif
          || (choice == CONF_MESSAGELEN)
          || ((choice == CONF_SMTPSRVR) && (messagelen != 0))
          || ((choice == CONF_SMTPPORT) && (messagelen != 0))
          || ((choice == CONF_SMTPDEST) && (messagelen != 0))
          || ((choice == CONF_SMTPFROM) && (messagelen != 0))
#if defined(MULTITHREAD)
          || (choice == CONF_NUMCPU)
#endif
          || (choice == CONF_CHECKPOINT)
          || (choice == CONF_CHECKPOINT2)
          || (choice == CONF_PREFERREDBLOCKSIZE)
          || (choice == CONF_PREFERREDCONTEST)
          )
        printf("%d)  %s [default:%s] ==> %s\n",
          (int)(choice + 1), options[choice][1],
          options[choice][2], list[choice].c_str());
    }
    printf("0)  Quit and Save\n");


    // get choice from user
    while(1)
    {
      printf("Choice --> ");
      fflush( stdout );
      fgets(parm, 128, stdin);
      choice = atoi(parm);

      if (choice == 0) return 0;
      else choice--;

      if ((choice >= 0 && choice < 1+CONF_FIREMODE) ||
          (choice == CONF_KEYPROXY && firemode >= 4) ||
          (choice == CONF_KEYPORT && firemode >= 5 && !CheckForcedKeyport()) ||
          ((choice == CONF_HTTPPROXY || choice == CONF_HTTPPORT || choice == CONF_HTTPID) &&
              (uuehttpmode > 1)) ||
          (choice == CONF_UUEHTTPMODE && firemode >= 5)
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_STRONGARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
          || (choice == CONF_CPUTYPE)
#endif
          || (choice == CONF_MESSAGELEN)
          || ((choice == CONF_SMTPSRVR) && (messagelen != 0))
          || ((choice == CONF_SMTPPORT) && (messagelen != 0))
          || ((choice == CONF_SMTPDEST) && (messagelen != 0))
          || ((choice == CONF_SMTPFROM) && (messagelen != 0))
#if defined(MULTITHREAD)
          || (choice == CONF_NUMCPU)
#endif
          || (choice == CONF_CHECKPOINT)
          || (choice == CONF_CHECKPOINT2)
          || (choice == CONF_PREFERREDBLOCKSIZE)
          || (choice == CONF_PREFERREDCONTEST)
          )
        break;
    }



    // prompt for new value
    printf("%s %s [%s] --> ", options[choice][1],
                              options[choice][3], list[choice].c_str());
    fflush( stdout );
    fgets(parm, sizeof(parm), stdin);
    for ( p = parm; *p; p++ )
    {
      if ( !isprint(*p) )
      {
        *p = 0;
        break;
      }
    }
    if ( parm[0] || choice == CONF_LOGNAME )
    {
      switch ( choice )
      {
        case CONF_ID:
          strncpy( id, parm, sizeof(id) - 1 );
          break;
        case CONF_THRESHOLD:
          inthreshold[0]=atoi(parm);
          printf("Output (max 1000)?--> ");
          fflush( stdout );
          fgets(parm2, sizeof(parm2), stdin);
          for ( p = parm2; *p; p++ )
          {
            if ( !isprint(*p) )
             {
              *p = 0;
              break;
            }
          }
          outthreshold[0]=atoi(parm2);
          if ( inthreshold[0] < 1   ) inthreshold[0] = 1;
          if ( inthreshold[0] > 1000 ) inthreshold[0] = 1000;
          if ( outthreshold[0] < 1   ) outthreshold[0] = 1;
          if ( outthreshold[0] > 1000 ) outthreshold[0] = 1000;
          break;
        case CONF_THRESHOLD2:
          inthreshold[1]=atoi(parm);
          printf("Output (max 1000)?--> ");
          fflush( stdout );
          fgets(parm2, sizeof(parm2), stdin);
          for ( p = parm2; *p; p++ )
          {
            if ( !isprint(*p) )
             {
              *p = 0;
              break;
            }
          }
          outthreshold[1]=atoi(parm2);
          if ( inthreshold[1] < 1   ) inthreshold[1] = 1;
          if ( inthreshold[1] > 1000 ) inthreshold[1] = 1000;
          if ( outthreshold[1] < 1   ) outthreshold[1] = 1;
          if ( outthreshold[1] > 1000 ) outthreshold[1] = 1000;
          break;
        case CONF_COUNT:
          blockcount = atoi(parm);
          if (blockcount < 0)
            blockcount = 0;
          break;
        case CONF_HOURS:
          minutes = (s32) (60. * atol(parm));
          if ( minutes < 0 ) minutes = 0;
          strncpy( hours, parm, sizeof(hours) - 1 );
          break;
        case CONF_TIMESLICE:
          timeslice = atoi(parm);
          if (timeslice < 0x1)
            timeslice = 0x10000;
          break;
        case CONF_NICENESS:
          niceness = atoi(parm);
          if ( niceness < 0 || niceness > 2 )
            niceness = 0;
          break;
        case CONF_LOGNAME:
          strncpy( logname, parm, sizeof(logname) - 1 );
          break;
        case CONF_FIREMODE:
          firemode = atoi(parm);
          if ( firemode < 1 || firemode > 5 )
            firemode = 1;
          switch (firemode) {
            case 1: strcpy( keyproxy, "us.v27.distributed.net" ); keyport = 2064; uuehttpmode = 0; break;
            case 2: strcpy( keyproxy, "us23.v27.distributed.net" ); keyport = 23; uuehttpmode = 0; break;
            case 3: strcpy( keyproxy, "us23.v27.distributed.net" ); keyport = 23; uuehttpmode = 1; break;
            case 4: uuehttpmode = 3; break;
          }
          break;
        case CONF_KEYPROXY:
          strncpy( keyproxy, parm, sizeof(keyproxy) - 1 );
          CheckForcedKeyport();
          break;
        case CONF_KEYPORT:
          keyport = atoi(parm); break;
        case CONF_HTTPPROXY:
          strncpy( httpproxy, parm, sizeof(httpproxy) - 1); break;
        case CONF_HTTPPORT:
          httpport = atoi(parm); break;
        case CONF_HTTPID:
          if ( strcmp(parm,".") == 0) {
             strcpy(httpid,"");
          } else if (uuehttpmode == 4) {  // socks4
             strcpy(httpid, parm);
          } else {             // http & socks5
             printf("Enter password--> ");
             fflush( stdout );
             fgets(parm2, sizeof(parm2), stdin);
             for ( p = parm2; *p; p++ )
             {
               if ( !isprint(*p) )
               {
                 *p = 0;
                 break;
               }
             }
             if (uuehttpmode == 5)   // socks5
               sprintf(httpid, "%s:%s", parm, parm2);
             else                    // http
               strcpy(httpid,Network::base64_encode(parm, parm2));
          }
          break;
        case CONF_UUEHTTPMODE:
          uuehttpmode = atoi(parm);
          if ( uuehttpmode < 0 || uuehttpmode > 5 )
            uuehttpmode = 0;
          break;
#if (CLIENT_CPU == CPU_X86)
        case CONF_CPUTYPE:
          cputype = atoi(parm);
          if (cputype < -1 || cputype > 5)
            cputype = -1;
          break;
#elif (CLIENT_CPU == CPU_STRONGARM)
        case CONF_CPUTYPE:
          cputype = atoi(parm);
          if (cputype < -1 || cputype > 1)
            cputype = -1;
          break;
#elif ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX)) )
        case CONF_CPUTYPE:
          cputype = atoi(parm);
          if (cputype < -1 || cputype > 1)
            cputype = -1;
          break;
#endif
        case CONF_MESSAGELEN:
          messagelen = atoi(parm);
          if ( messagelen < 0 )
            messagelen = 0;
          if ( messagelen > MAXMAILSIZE) {
            messagelen = MAXMAILSIZE;
          }
          break;
        case CONF_SMTPPORT:
          smtpport = atoi(parm);
          if ( smtpport < 0 )
            smtpport = 0;
          break;
        case CONF_SMTPSRVR:
          strncpy( smtpsrvr, parm, sizeof(smtpsrvr) - 1 );
          break;
        case CONF_SMTPFROM:
          strncpy( smtpfrom, parm, sizeof(smtpfrom) - 1 );
          break;
        case CONF_SMTPDEST:
          strncpy( smtpdest, parm, sizeof(smtpdest) - 1 );
          break;
#if defined(MULTITHREAD)
        case CONF_NUMCPU:
          numcpu = atoi(parm);
  #if (CLIENT_OS == OS_BEOS)
          if (numcpu == -1)
          {
            get_system_info(&the_info);
            numcputemp = the_info.cpu_count;
            if (numcputemp < 1) {
              numcputemp = 1;
            }
            else if (numcputemp > MAXCPUS) {
              numcputemp = MAXCPUS;
            }
            if (numcputemp == the_info.cpu_count) {
              LogScreenf("Detected %d cpu(s)\n", numcputemp);
            }
            else {
              LogScreenf("Detected %d cpu(s); using %d cpu(s)\n",
              the_info.cpu_count, numcputemp);
            }
          } else if (numcpu < 1) {
            numcputemp = numcpu = 1;
          } else if (numcpu > MAXCPUS) {
            numcputemp = numcpu = MAXCPUS;
          } else {
            numcputemp = numcpu;
          }
  #else
          numcputemp = numcpu;
      #if (CLIENT_OS == OS_WIN32)
          if (numcpu == -1) {
            GetSystemInfo(&systeminfo);
            numcputemp=systeminfo.dwNumberOfProcessors;
            LogScreenf("Detected %d cpu(s)\n",numcputemp);
          } else {
            numcputemp=numcpu;
          }
      #elif (CLIENT_OS == OS_OS2)
          if (numcpu == -1)
          {
            int rc = DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                    &numcputemp, sizeof(numcputemp));
            // check if call is valid if not, default to one
            if(rc!=0 || numcputemp < 1 || numcputemp > MAXCPUS)
              numcputemp = numcpu;
            LogScreenf("Detected %d cpu(s)\n", numcputemp);
          }
      #elif ((CLIENT_OS == OS_NETWARE) && (CLIENT_CPU == CPU_X86))
          numcputemp = numcpu = CliValidateProcessorCount( numcpu );
      #endif
          if ( numcputemp < 1 )
            numcputemp = 1;
          if ( numcputemp > MAXCPUS )
            numcputemp = MAXCPUS;
      #if ((CLIENT_CPU != CPU_X86) && (CLIENT_CPU != CPU_88K) && (CLIENT_CPU != CPU_SPARC))
          if ( numcputemp > 1 )
          {
            numcputemp = 1;
            LogScreen("Core routines not yet updated for thread safe operation.  Using 1 cpu.\n");
          }
      #endif
          //numcputemp=numcpu;
  #endif
          break;
#endif
        case CONF_CHECKPOINT:
          strncpy( checkpoint_file[0] , parm, sizeof(checkpoint_file)/2 -1 );
          break;
        case CONF_CHECKPOINT2:
          strncpy( checkpoint_file[1] , parm, sizeof(checkpoint_file)/2 -1 );
          break;
        case CONF_PREFERREDBLOCKSIZE:
          preferred_blocksize = atoi(parm);
          if (preferred_blocksize < 28) preferred_blocksize = 28;
          if (preferred_blocksize > 31) preferred_blocksize = 31;
          break;
        case CONF_PREFERREDCONTEST:
          preferred_contest_id = atoi(parm) - 1;
          if ((preferred_contest_id < 0) || (preferred_contest_id > 1))
             preferred_contest_id = 1;
          break;
        default:
          break;
      }
    }
  }
}

//----------------------------------------------------------------------------

s32 Client::ReadConfig(void)
{
  IniSection ini;
  s32 inierror, tempconfig;
  char *p, buffer[64];

  inierror = ini.ReadIniFile( inifilename );
  if ( inierror )
  {
    LogScreen( "Error reading ini file - Using defaults\n" );
  }

#define INIGETKEY(key) (ini.getkey(OPTION_SECTION, options[key][0], options[key][2])[0])

  INIGETKEY(CONF_ID).copyto(id, sizeof(id));
  INIGETKEY(CONF_THRESHOLD).copyto(buffer, sizeof(buffer));
  p = strchr( buffer, ':' );
  if (p == NULL) {
    outthreshold[0]=inthreshold[0]=atoi(buffer);
  } else {
    outthreshold[0]=atoi(p+1);
    *p=0;
    inthreshold[0]=atoi(buffer);
  }
  INIGETKEY(CONF_THRESHOLD2).copyto(buffer, sizeof(buffer));
  p = strchr( buffer, ':' );
  if (p == NULL) {
    outthreshold[1]=inthreshold[1]=atoi(buffer);
  } else {
    outthreshold[1]=atoi(p+1);
    *p=0;
    inthreshold[1]=atoi(buffer);
  }
  blockcount = INIGETKEY(CONF_COUNT);
  INIGETKEY(CONF_HOURS).copyto(hours, sizeof(hours));
  minutes = (s32) (atol(hours) * 60.);
  timeslice = INIGETKEY(CONF_TIMESLICE);
  niceness = INIGETKEY(CONF_NICENESS);
  INIGETKEY(CONF_LOGNAME).copyto(logname, sizeof(logname));
  firemode = INIGETKEY(CONF_FIREMODE);
  INIGETKEY(CONF_KEYPROXY).copyto(keyproxy, sizeof(keyproxy));
  keyport = INIGETKEY(CONF_KEYPORT);
  INIGETKEY(CONF_HTTPPROXY).copyto(httpproxy, sizeof(httpproxy));
  httpport = INIGETKEY(CONF_HTTPPORT);
  uuehttpmode = INIGETKEY(CONF_UUEHTTPMODE);
  INIGETKEY(CONF_HTTPID).copyto(httpid, sizeof(httpid));
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_STRONGARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
  cputype = INIGETKEY(CONF_CPUTYPE);
#endif
  messagelen = INIGETKEY(CONF_MESSAGELEN);
  smtpport = INIGETKEY(CONF_SMTPPORT);
  INIGETKEY(CONF_SMTPSRVR).copyto(smtpsrvr, sizeof(smtpsrvr));
  INIGETKEY(CONF_SMTPFROM).copyto(smtpfrom, sizeof(smtpfrom));
  INIGETKEY(CONF_SMTPDEST).copyto(smtpdest, sizeof(smtpdest));
#if defined(MULTITHREAD)
  numcpu = INIGETKEY(CONF_NUMCPU);
#endif
  INIGETKEY(CONF_CHECKPOINT).copyto(checkpoint_file[0], sizeof(checkpoint_file)/2);
  INIGETKEY(CONF_CHECKPOINT2).copyto(checkpoint_file[1], sizeof(checkpoint_file)/2);
  randomprefix = INIGETKEY(CONF_RANDOMPREFIX);
  preferred_contest_id = INIGETKEY(CONF_PREFERREDCONTEST) - 1;
  preferred_blocksize = INIGETKEY(CONF_PREFERREDBLOCKSIZE);

  tempconfig=ini.getkey(OPTION_SECTION, "runbuffers", "0")[0];
  if (tempconfig) {
    offlinemode=2;
  } else {
    tempconfig=ini.getkey(OPTION_SECTION, "runoffline", "0")[0];
    if (tempconfig) offlinemode=1;
  }
  ini.getkey(OPTION_SECTION,"in",in_buffer_file[0])[0].copyto(in_buffer_file[0],sizeof(in_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"out",out_buffer_file[0])[0].copyto(out_buffer_file[0],sizeof(out_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"in2",in_buffer_file[1])[0].copyto(in_buffer_file[1],sizeof(in_buffer_file)/2);
  ini.getkey(OPTION_SECTION,"out2",out_buffer_file[1])[0].copyto(out_buffer_file[1],sizeof(out_buffer_file)/2);
  tempconfig=ini.getkey(OPTION_SECTION, "percentoff", "0")[0];
  if (tempconfig) percentprintingoff=1;
  tempconfig=ini.getkey(OPTION_SECTION, "frequent", "0")[0];
  if (tempconfig) connectoften=1;
  tempconfig=ini.getkey(OPTION_SECTION, "nodisk", "0")[0];
  if (tempconfig) nodiskbuffers=1;
  tempconfig=ini.getkey(OPTION_SECTION, "quiet", "0")[0];
  if (tempconfig) quietmode=1;
  tempconfig=ini.getkey(OPTION_SECTION, "nofallback", "0")[0];
  if (tempconfig) nofallback=1;
  tempconfig=ini.getkey(OPTION_SECTION, "cktime", "0")[0];
  if (tempconfig) checkpoint_min=max(2,tempconfig);
  tempconfig=ini.getkey(OPTION_SECTION, "nettimeout", "60")[0];
  if (tempconfig) nettimeout=min(300,max(30,nettimeout));
  tempconfig=ini.getkey(OPTION_SECTION, "noexitfilecheck", "0")[0];
  if (tempconfig) noexitfilecheck=1;
  tempconfig=ini.getkey(OPTION_SECTION, "exitfilechecktime", "30")[0];
  if (tempconfig) exitfilechecktime=max(tempconfig,1);
#if (CLIENT_OS == OS_WIN32)
#if (!defined(WINNTSERVICE))
  tempconfig=ini.getkey(OPTION_SECTION, "win95hidden", "0")[0];
  if (tempconfig) win95hidden=1;
#endif
  tempconfig=ini.getkey(OPTION_SECTION, "lurk", "0")[0];
  if (tempconfig) lurk=1;
  tempconfig=ini.getkey(OPTION_SECTION, "lurkonly", "0")[0];
  if (tempconfig) {lurk=2; connectoften=0;}
#endif
  ini.getkey(OPTION_SECTION,"pausefile",pausefile)[0].copyto(pausefile,sizeof(pausefile));
  tempconfig=ini.getkey(OPTION_SECTION, "contestdone", "0")[0];
  if (tempconfig) contestdone[0]=1;
  tempconfig=ini.getkey(OPTION_SECTION, "contestdone2", "0")[0];
  if (tempconfig) contestdone[1]=1;

#undef INIGETKEY
#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) && defined(NOMAIN)
  InternalReadConfig(ini);
#endif

  ValidateConfig();

  return( inierror ? -1 : 0 );
}

// --------------------------------------------------------------------------

void Client::ValidateConfig( void )
{
#if (CLIENT_OS == OS_WIN32)
  SYSTEM_INFO systeminfo;
  static bool did_detect_message = false;
#elif (CLIENT_OS == OS_BEOS)
  system_info the_info;
  static bool did_detect_message = false;
#endif
  if ( inthreshold[0] < 1   ) inthreshold[0] = 1;
  if ( inthreshold[0] > 1000 ) inthreshold[0] = 1000;
  if ( outthreshold[0] < 1   ) outthreshold[0] = 1;
  if ( outthreshold[0] > 1000 ) outthreshold[0] = 1000;
  if ( inthreshold[1] < 1   ) inthreshold[1] = 1;
  if ( inthreshold[1] > 1000 ) inthreshold[1] = 1000;
  if ( outthreshold[1] < 1   ) outthreshold[1] = 1;
  if ( outthreshold[1] > 1000 ) outthreshold[1] = 1000;
  if ( blockcount < 0 ) blockcount = 0;
  if ( timeslice < 1 ) timeslice = 0x10000;
  if ( niceness < 0 || niceness > 2 ) niceness = 0;
  if ( firemode < 1 || firemode > 5 ) firemode = 1;
  if ( uuehttpmode < 0 || uuehttpmode > 5 ) uuehttpmode = 0;
#if (CLIENT_CPU == CPU_X86)
  if ( cputype < -1 || cputype > 5) cputype = -1;
#elif ((CLIENT_CPU == CPU_STRONGARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
  if ( cputype < -1 || cputype > 1) cputype = -1;
#endif
  if ( messagelen < 0) messagelen = 0;
  if ( messagelen > MAXMAILSIZE) messagelen = MAXMAILSIZE;
  if ( randomprefix <0  ) randomprefix=100;
  if ( randomprefix >255) randomprefix=100;
  if (( preferred_contest_id < 0 ) || ( preferred_contest_id > 1 )) preferred_contest_id = 1;
  if (preferred_blocksize < 28) preferred_blocksize = 28;
  if (preferred_blocksize > 31) preferred_blocksize = 31;
  if ( minutes < 0 ) minutes=0;
  if ( blockcount < 0 ) blockcount=0;

  if (strlen(checkpoint_file[0])==0) strcpy(checkpoint_file[0],"none");
  if (strlen(checkpoint_file[1])==0) strcpy(checkpoint_file[1],"none");

  switch ( firemode )
  {
    case 1:
      strcpy( keyproxy, "us.v27.distributed.net" );
      uuehttpmode = 0;
      break;
    case 2:
      strcpy( keyproxy, "us23.v27.distributed.net" );
      uuehttpmode = 0;
      break;
    case 3:
      strcpy( keyproxy, "us23.v27.distributed.net" );
      uuehttpmode = 1;
      break;
    case 4:
      uuehttpmode = 3;
      break;
  }

  CheckForcedKeyport();

  strcpy(mailmessage.destid,smtpdest);
  strcpy(mailmessage.fromid,smtpfrom);
  strcpy(mailmessage.smtp,smtpsrvr);
  strcpy(mailmessage.rc5id,id);
  mailmessage.messagelen=messagelen;
  mailmessage.port=smtpport;

#if (CLIENT_OS == OS_BEOS)
  if (numcpu == -1)
  {
    get_system_info(&the_info);
    numcputemp = the_info.cpu_count;
    if (numcputemp < 1) {
      numcputemp = 1;
    }
    else if (numcputemp > MAXCPUS) {
      numcputemp = MAXCPUS;
    }
    if (!did_detect_message) {
      if (numcputemp == the_info.cpu_count)
      {
        LogScreenf("Detected %d cpu(s)\n", numcputemp);
      }
      else
      {
        LogScreenf("Detected %d cpu(s); using %d cpu(s)\n",
          the_info.cpu_count, numcputemp);
      }
      did_detect_message = true;
    }
  } else if (numcpu < 1) {
    numcputemp = numcpu = 1;
  } else if (numcpu > MAXCPUS) {
    numcputemp = numcpu = MAXCPUS;
  } else {
    numcputemp = numcpu;
  }
#elif ((CLIENT_OS == OS_NETWARE) && (CLIENT_CPU == CPU_X86))
  numcputemp = numcpu = CliValidateProcessorCount( numcpu );
#else
  numcputemp = numcpu;
  #if (CLIENT_OS == OS_WIN32)
    if (numcpu == -1) {
      GetSystemInfo(&systeminfo);
      numcputemp=systeminfo.dwNumberOfProcessors;
      if (!did_detect_message)
      {
        LogScreenf("Detected %d cpu(s)\n",numcputemp);
        did_detect_message = true;
      }
    } else {
      numcputemp=numcpu;
    }
  #elif (CLIENT_OS == OS2)
    if (numcpu == -1)
    {
      int rc = DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                &numcputemp, sizeof(numcputemp));
      // check if call is valid if not, default to one
      if(rc!=0 || numcputemp < 1 || numcputemp > MAXCPUS)
        numcputemp = numcpu;
      if (!did_detect_message)
      {
        LogScreenf("Detected %d cpu(s)\n", numcputemp);
        did_detect_message = true;
      }
    }
  #endif
  if ( numcputemp < 1)
     numcputemp = 1;
  if ( numcputemp > MAXCPUS)
     numcputemp = MAXCPUS;

  #if ((CLIENT_CPU != CPU_X86) && (CLIENT_CPU != CPU_88K) && (CLIENT_CPU != CPU_SPARC) && (CLIENT_CPU != CPU_POWERPC))
    if ( numcpu > 1 )
    {
      numcpu = numcputemp = 1;
      LogScreen("Core routines not yet updated for thread safe operation.  Using 1 cpu.\n");
    }
  #endif
    
  #if !defined(MULTITHREAD)
  if ( numcpu > 1) {
    numcpu = numcputemp = 1;
  }
  #endif
  //numcputemp=numcpu;
#endif
#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) & defined(NOMAIN)
  InternalValidateConfig();
#elif (CLIENT_OS == OS_NETWARE)
  {
    //    (destbuff, destsize, defaultvalue, changetoNONEifempty, source)

    CliValidateSinglePath( inifilename, sizeof(inifilename), 
                             "rc5des.ini", 0, inifilename );
    if (!nodiskbuffers)
    {
      CliValidateSinglePath( in_buffer_file[0], sizeof(in_buffer_file[0]), 
                                       "buff-in.rc5", 0, in_buffer_file[0] );
      CliValidateSinglePath( out_buffer_file[0], sizeof(out_buffer_file[0]), 
                                       "buff-out.rc5", 0, out_buffer_file[0] );
      CliValidateSinglePath( in_buffer_file[1], sizeof(in_buffer_file[1]), 
                                       "buff-out.des", 0, in_buffer_file[1] );
      CliValidateSinglePath( out_buffer_file[1], sizeof(out_buffer_file[1]), 
                                       "buff-out.des", 0, out_buffer_file[1] );
    }
    if (strcmp(exit_flag_file,"none")!=0)
      CliValidateSinglePath( exit_flag_file, sizeof(exit_flag_file), 
                                     "exitrc5.now", 1, exit_flag_file);
    if (strcmp(pausefile,"none")!=0)
      CliValidateSinglePath( pausefile, sizeof(pausefile), 
                                     "none", 1, pausefile);
    if (strcmp(checkpoint_file[0],"none")!=0)
      CliValidateSinglePath( checkpoint_file[0], sizeof(checkpoint_file[0]), 
                                       "ckpoint.rc5", 1, checkpoint_file[0]);
    if (strcmp(checkpoint_file[1],"none")!=0)
      CliValidateSinglePath( checkpoint_file[1], sizeof(checkpoint_file[1]), 
                                       "ckpoint.des", 1, checkpoint_file[1]);
    if (strlen(logname)!=0)
      CliValidateSinglePath( logname, sizeof(logname), "", 0, logname);
  }
#endif
  InitRandom2( id );

  if ( contestdone[0] && contestdone[1])
  {
    Log( "[%s] Both contests are marked as over.  Correct the ini file and restart\n", Time() );
    Log( "[%s] This may mean the contests are over.  Check at http://www.distributed.net/rc5/\n", Time() );
    exit(-1);
  }
}

// --------------------------------------------------------------------------

s32 Client::WriteConfig(void)
{
  IniSection ini;
  char buffer[64];

  ini.ReadIniFile(inifilename);

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, options[key][0], IniString(value))

  INISETKEY( CONF_ID, id );
  sprintf(buffer,"%d:%d",(int)inthreshold[0],(int)outthreshold[0]);
  INISETKEY( CONF_THRESHOLD, buffer );
  sprintf(buffer,"%d:%d",(int)inthreshold[1],(int)outthreshold[1]);
  INISETKEY( CONF_THRESHOLD2, buffer );
  INISETKEY( CONF_COUNT, blockcount );
  sprintf(hours,"%u.%02u", (unsigned)(minutes/60), 
    (unsigned)(minutes%60)); //1.000000 hours looks silly
  INISETKEY( CONF_HOURS, hours );
  INISETKEY( CONF_TIMESLICE, timeslice );
  INISETKEY( CONF_NICENESS, niceness );
  INISETKEY( CONF_LOGNAME, logname );
  INISETKEY( CONF_FIREMODE, firemode );
  INISETKEY( CONF_KEYPROXY, keyproxy );
  INISETKEY( CONF_KEYPORT, keyport );
  INISETKEY( CONF_HTTPPROXY, httpproxy );
  INISETKEY( CONF_HTTPPORT, httpport );
  INISETKEY( CONF_UUEHTTPMODE, uuehttpmode );
  INISETKEY( CONF_HTTPID, httpid);
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_STRONGARM) || ((CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS == OS_LINUX || CLIENT_OS == OS_AIX)) )
  INISETKEY( CONF_CPUTYPE, cputype );
#endif
  INISETKEY( CONF_MESSAGELEN, messagelen );
  INISETKEY( CONF_SMTPSRVR, smtpsrvr );
  INISETKEY( CONF_SMTPPORT, smtpport );
  INISETKEY( CONF_SMTPFROM, smtpfrom );
  INISETKEY( CONF_SMTPDEST, smtpdest );
#if defined(MULTITHREAD)
  INISETKEY( CONF_NUMCPU, numcpu );
#endif
  INISETKEY( CONF_CHECKPOINT, checkpoint_file[0] );
  INISETKEY( CONF_CHECKPOINT2, checkpoint_file[1] );
  INISETKEY( CONF_RANDOMPREFIX, randomprefix );
  INISETKEY( CONF_PREFERREDCONTEST, (s32)(preferred_contest_id + 1) );
  INISETKEY( CONF_PREFERREDBLOCKSIZE, preferred_blocksize );

#undef INISETKEY

  ini.setrecord(OPTION_SECTION, "contestdone",  IniString(contestdone[0]));
  ini.setrecord(OPTION_SECTION, "contestdone2", IniString(contestdone[1]));

#if (CLIENT_OS == OS_WIN32)
  ini.setrecord(OPTION_SECTION, "lurk",  IniString(lurk));
#endif

#define INIFIND(key) ini.findfirst(OPTION_SECTION, options[key][0])

  if (uuehttpmode <= 1)
  {
    // wipe out httpproxy and httpport & httpid
    IniRecord *ptr;
    ptr = INIFIND( CONF_HTTPPROXY );
    if (ptr) ptr->values.Erase();
    ptr = INIFIND( CONF_HTTPPORT );
    if (ptr) ptr->values.Erase();
    ptr = INIFIND( CONF_HTTPID );
    if (ptr) ptr->values.Erase();
  }

  if (firemode < 4)
  {
    // wipe keyproxy and keyport
    IniRecord *ptr;
    ptr = INIFIND( CONF_KEYPROXY );
    if (ptr) ptr->values.Erase();
    ptr = INIFIND( CONF_KEYPORT );
    if (ptr) ptr->values.Erase();
  }

  if (firemode < 5)
  {
    // wipe uuehttpmode
    IniRecord *ptr;
    ptr = INIFIND( CONF_UUEHTTPMODE );
    if (ptr) ptr->values.Erase();
  }

#if ((CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)) & defined(NOMAIN)
  InternalWriteConfig(ini);
#endif

#undef INIFIND

  return( ini.WriteIniFile(inifilename) ? -1 : 0 );
}

//----------------------------------------------------------------------------


#if defined(WINNTSERVICE)
static SERVICE_STATUS_HANDLE serviceStatusHandle;

void __stdcall ServiceCtrlHandler(DWORD controlCode)
{
  // update our status to stopped
  SERVICE_STATUS serviceStatus;
  serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = 0;
  serviceStatus.dwCheckPoint = 0;
  if (controlCode == SERVICE_CONTROL_SHUTDOWN ||
      controlCode == SERVICE_CONTROL_STOP)
  {
    serviceStatus.dwCurrentState = SERVICE_STOP_PENDING;
    serviceStatus.dwControlsAccepted = 0;
    serviceStatus.dwWaitHint = 10000;
    SignalTriggered = UserBreakTriggered = 1;
  } else {
    // SERVICE_CONTROL_INTERROGATE
    serviceStatus.dwCurrentState = SERVICE_RUNNING;
    serviceStatus.dwWaitHint = 0;
  }
  SetServiceStatus(serviceStatusHandle, &serviceStatus);
}
#endif

// ---------------------------------------------------------------------------

#if defined(WINNTSERVICE)

static Client *mainclient;

#pragma argsused
void ServiceMain(DWORD argc, LPTSTR *argv)
{
  SERVICE_STATUS serviceStatus;
  serviceStatusHandle = RegisterServiceCtrlHandler(WINNTSERVICE,
      ServiceCtrlHandler);

  // update our status to running
  serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  serviceStatus.dwCurrentState = SERVICE_RUNNING;
  serviceStatus.dwControlsAccepted = (SERVICE_ACCEPT_SHUTDOWN | SERVICE_ACCEPT_STOP);
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = 0;
  serviceStatus.dwCheckPoint = 0;
  serviceStatus.dwWaitHint = 0;
  SetServiceStatus(serviceStatusHandle, &serviceStatus);

  // start working
  NetworkInitialize();
  mainclient->ValidateConfig();
  mainclient->Run();
  mainclient->mailmessage.quietmode = mainclient->quietmode;
  mainclient->mailmessage.checktosend(1);
  NetworkDeinitialize();

  // update our status to stopped
  serviceStatus.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
  serviceStatus.dwCurrentState = SERVICE_STOPPED;
  serviceStatus.dwControlsAccepted = 0;
  serviceStatus.dwWin32ExitCode = NO_ERROR;
  serviceStatus.dwServiceSpecificExitCode = 0;
  serviceStatus.dwCheckPoint = 0;
  serviceStatus.dwWaitHint = 0;
  SetServiceStatus(serviceStatusHandle, &serviceStatus);
}
#endif

// ---------------------------------------------------------------------------

s32 Client::Install()
{
#if (!defined(WINNTSERVICE)) && (CLIENT_OS == OS_WIN32)
  HKEY srvkey=NULL;
  DWORD dwDisp=NULL;
  char mypath[200];
  GetModuleFileName(NULL, mypath, sizeof(mypath));

  strcat( mypath, " -hide" );

  // register a Win95 "RunService" item
  if (RegCreateKeyEx(HKEY_LOCAL_MACHINE,
      "Software\\Microsoft\\Windows\\CurrentVersion\\RunServices",0,"",
            REG_OPTION_NON_VOLATILE,KEY_ALL_ACCESS,NULL,
            &srvkey,&dwDisp) == ERROR_SUCCESS)
  {
    RegSetValueEx(srvkey, "bovwin32", 0, REG_SZ, (unsigned const char *)mypath, strlen(mypath) + 1);
    RegCloseKey(srvkey);
  }

  // register a Win95 "Run" item
  if (RegCreateKeyEx(HKEY_LOCAL_MACHINE,
            "Software\\Microsoft\\Windows\\CurrentVersion\\Run",0,"",
            REG_OPTION_NON_VOLATILE,KEY_ALL_ACCESS,NULL,
            &srvkey,&dwDisp) == ERROR_SUCCESS)
  {
    RegSetValueEx(srvkey, "bovwin32", 0, REG_SZ, (unsigned const char *)mypath, strlen(mypath) + 1);
    RegCloseKey(srvkey);
  }
  LogScreen("Win95 Service installation complete.\n");
#elif defined(WINNTSERVICE)
  char mypath[200];
  GetModuleFileName(NULL, mypath, sizeof(mypath));
  SC_HANDLE myService, scm;
  scm = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
  if (scm)
  {
    myService = CreateService(scm, WINNTSERVICE,
        "Distributed.Net RC5/DES Service Client",
        SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS,
        SERVICE_AUTO_START, SERVICE_ERROR_NORMAL,
        mypath, 0, 0, 0, 0, 0);
    if (myService)
    {
      LogScreen("Windows NT Service installation complete.\n"
          "Click on the 'Services' icon in 'Control Panel' and mark the\n"
          "Bovine RC5/DES service to startup automatically.\n");
      CloseServiceHandle(myService);
    } else {
      LogScreen("Error creating service entry.\n");
    }
    CloseServiceHandle(scm);
  } else {
    LogScreen("Error opening service control manager.\n");
  }
#elif (CLIENT_OS == OS_OS2)
  int rc;
  const int len = 4068;

  char   pszClassName[] = "WPProgram";
  char   pszTitle[] = "RC5-DES Cracking Client";
  char   pszLocation[] = "<WP_START>";    // Startup Folder
  ULONG ulFlags = 0;

  char   pszSetupString[len] =
            "OBJECTID=<RC5DES-CLI>;"
            "MINIMIZED=YES;"
            "PROGTYPE=WINDOWABLEVIO;";

  // Add full path of the program
  strncat(pszSetupString, "EXENAME=",len);

  if(os2hidden == 1)   // Run detached
  {
    strncat(pszSetupString, "CMD.EXE;", len);     // command processor
    strncat(pszSetupString, "PARAMETERS=/c detach ", len);   // detach
  }

  // Add exepath and exename
  strncat(pszSetupString, exepath, len);
  strncat(pszSetupString, exename, len);
  strncat(pszSetupString, ";", len);

  // Add on Working Directory
  strncat(pszSetupString, "STARTUPDIR=", len);
  strncat(pszSetupString, exepath, len);
  strncat(pszSetupString, ";", len);

  rc = WinCreateObject(pszClassName, pszTitle, pszSetupString,
              pszLocation, ulFlags);
  if(rc == NULLHANDLE)
    LogScreen("ERROR: RC5-DES Program object could not be added "
            "into your Startup Folder\n"
            "RC5-DES is probably already installed\n");
  else
    LogScreen("RC5-DES Program object has been added into your Startup Folder\n");
#endif
  return 0;
}

// ---------------------------------------------------------------------------

s32 Client::Uninstall(void)
{
#if (!defined(WINNTSERVICE)) && (CLIENT_OS == OS_WIN32)
  HKEY srvkey;

  // unregister a Win95 "RunService" item
  if (RegOpenKey(HKEY_LOCAL_MACHINE,
      "Software\\Microsoft\\Windows\\CurrentVersion\\RunServices",
      &srvkey) == ERROR_SUCCESS)
  {
    RegDeleteValue(srvkey, "bovwin32");
    RegCloseKey(srvkey);
  }

  // unregister a Win95 "Run" item
  if (RegOpenKey(HKEY_LOCAL_MACHINE,
      "Software\\Microsoft\\Windows\\CurrentVersion\\Run",
      &srvkey) == ERROR_SUCCESS)
  {
    RegDeleteValue(srvkey, "bovwin32");
    RegCloseKey(srvkey);
  }
  LogScreen("Win95 Service uninstallation complete.\n");
#elif defined(WINNTSERVICE)
  SC_HANDLE myService, scm;
  SERVICE_STATUS status;
  scm = OpenSCManager(0, 0, SC_MANAGER_CREATE_SERVICE);
  if (scm)
  {
    myService = OpenService(scm, WINNTSERVICE,
        SERVICE_ALL_ACCESS | DELETE);
    if (myService)
    {
      if (QueryServiceStatus(myService, &status) &&
        status.dwCurrentState != SERVICE_STOPPED)
      {
        LogScreen("Service currently active.  Stopping service...\n");
        if (!ControlService(myService, SERVICE_CONTROL_STOP, &status))
          LogScreen("Failed to stop service!\n");
      }
      if (DeleteService(myService))
      {
        LogScreen("Windows NT Service uninstallation complete.\n");
      } else {
        LogScreen("Error deleting service entry.\n");
      }
      CloseServiceHandle(myService);
    }
    CloseServiceHandle(scm);
  } else {
    LogScreen("Error opening service control manager.\n");
  }
#elif (CLIENT_OS == OS_OS2)
  int rc;
  const int len = 4068;
  char *cwd;

  char pObjectID[len];
  HOBJECT hObject;

  hObject = WinQueryObject("<RC5DES-CLI>");

  if(hObject == NULLHANDLE)
    LogScreen("ERROR: RC5-DES Client object was not found\n"
          "No RC5-DES client installed in the Startup folder\n");
  else
  {
    LogScreen("RC5-DES Client object found in Startup Folder... ");

    rc = WinDestroyObject(hObject);
    if(rc == TRUE)
      LogScreen("Object removed\n");
    else
      LogScreen("Object NOT removed\n");
  }
#endif
  return 0;
}

// ---------------------------------------------------------------------------

s32 Client::RunStartup(void)
{
#if (CLIENT_OS == OS_WIN32)
  LPVOID lpMsgBuf;
  OSVERSIONINFO osver;

  if (lurk && (!rasenumconnections || !rasgetconnectstatus))
  {
    HINSTANCE hinstance;
    hinstance=LoadLibrary("RASAPI32.dll");
    if (hinstance == NULL)
    {
      LogScreen("Couldn't load rasapi32.dll\n");
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      return -1;
    }
    rasenumconnections = (rasenumconnectionsT) GetProcAddress(hinstance,"RasEnumConnectionsA");
    if (rasenumconnections==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
    rasgetconnectstatus = (rasgetconnectstatusT) GetProcAddress(hinstance,"RasGetConnectStatusA");
    if (rasgetconnectstatus==NULL)
    {
      FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
        NULL,GetLastError(),MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), // Default language
        (LPTSTR) &lpMsgBuf,0,NULL);
      LogScreenf("%s\n",lpMsgBuf);
      LogScreen("Dial-up must be installed for -lurk/-lurkonly\n");
      LocalFree( lpMsgBuf );
      return -1;
    }
  }
#endif
#if ((!defined(WINNTSERVICE)) && (CLIENT_OS == OS_WIN32))
  // register ourself as a Win95 service
  if (win95hidden)
  {
    HMODULE kernl = GetModuleHandle("KERNEL32");
    if (kernl)
    {
      typedef DWORD (CALLBACK *ULPRET)(DWORD,DWORD);
      ULPRET func = (ULPRET) GetProcAddress(kernl, "RegisterServiceProcess");
      if (func) (*func)(0, 1);
    }

    // free the console window
    osver.dwOSVersionInfoSize=sizeof(OSVERSIONINFO);
    GetVersionEx(&osver);
    if (VER_PLATFORM_WIN32_NT == osver.dwPlatformId)
    {
      LogScreen("\n This is not recommended under NT.  Please use the NT Service client"
            "\n (There have been cases of this conflicting with system process csrss.exe)\n"
            "Continuing...\n");
      sleep(2);
    }
    FreeConsole();

    // only allow one running instance
    CreateMutex(NULL, TRUE, "Bovine RC5/DES Win32 Client");
    if (GetLastError()) return -1;
  }
  return 0;
#elif defined(WINNTSERVICE)
  LogScreen("Attempting to start up NT service.\n");
  mainclient = this;
  SERVICE_TABLE_ENTRY serviceTable[] = {
    {WINNTSERVICE, (LPSERVICE_MAIN_FUNCTION) ServiceMain},
    {NULL, NULL}};
  if (!StartServiceCtrlDispatcher(serviceTable))
  {
    LogScreen("Error starting up NT service.  Please remember that this\n"
          "client cannot be invoked directly.  If you wish to install it\n"
          "as a service, use the -install option\n");
    return -1;
  }
  return -1;
#else
  return 0;
#endif
}

// ---------------------------------------------------------------------------

s32 Client::SelectCore(void)
{
#if (CLIENT_OS == OS_AMIGA)
  if (!(SysBase->AttnFlags & AFF_68020))
  {
    LogScreen("\nIncompatible CPU type.  Sorry.\n");
    return -1;
  }
#elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS == OS_BEOS)
  // Be OS isn't supported on 601 machines
  LogScreen( "| PowerPC assembly by Dan Oetting at USGS\n");
  double fasttime = 0;
  whichcrunch = 1;
#elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_WIN32)
  const int benchsize = 500000;
  double fasttime = 0;
  LogScreen( "| RC5 PowerPC assembly by Dan Oetting at USGS\n");
  int fastcore = cputype;
  if (fastcore == -1)
  {
    for (whichcrunch = 0; whichcrunch < 2; whichcrunch++)
    {
      Problem problem;
      ContestWork contestwork;
      contestwork.key.lo = contestwork.key.hi = htonl( 0 );
      contestwork.iv.lo = contestwork.iv.hi = htonl( 0 );
      contestwork.plain.lo = contestwork.plain.hi = htonl( 0 );
      contestwork.cypher.lo = contestwork.cypher.hi = htonl( 0 );
      contestwork.keysdone.lo = contestwork.keysdone.hi = htonl( 0 );
      contestwork.iterations.lo = htonl( benchsize );
      contestwork.iterations.hi = htonl( 0 );
      problem.LoadState( &contestwork , 0 ); // RC5 core selection

      LogScreenf( "| Benchmarking version %d: ", whichcrunch );

      fflush( stdout );

      #ifndef NEW_STATS_AND_LOGMSG_STUFF
      struct timeval start, stop;
      struct timezone dummy;
      gettimeofday( &start, &dummy );
      #endif

      problem.Run( benchsize , 0 );

      #ifdef NEW_STATS_AND_LOGMSG_STUFF
        double elapsed = CliGetKeyrateForProblem( &problem );
        LogScreenf( "%.1f kkeys/sec\n", (elapsed / 1000.0) );
      #else
      gettimeofday( &stop, &dummy );

      double elapsed = max(.001,(stop.tv_sec - start.tv_sec) +
                  (((double)stop.tv_usec - (double)start.tv_usec)/1000000.0));
        LogScreenf( "%.1f kkeys/sec\n", (benchsize / 1000.0) / elapsed );
      #endif

      if (fastcore < 0 || elapsed < fasttime)
          {fastcore = whichcrunch; fasttime = elapsed;}
    }
  }
  whichcrunch = fastcore;
  LogScreenf( "| Using v%d.\n\n", whichcrunch );
  /*
  switch (whichcrunch)
  {
    case 0:
      Log("Using the 601 core.\n\n");
      break;
    case 1:
      Log("Using the 603/604/750 core.\n\n");
      break;
  }
  */
#elif (CLIENT_CPU == CPU_X86)
  // benchmark all cores
  int fastcore = cputype;
  if (fastcore == -1) fastcore = x86id(); // Will return -1 if unable to identify.
  if (fastcore == -1)
  {
    const int benchsize = 50000;
    double fasttime = 0;

    LogScreen("Automatically selecting fastest core...\n");
    LogScreen("This is just a guess based on a small test of each core.  If you know what CPU\n");
    LogScreenf("this machine has, then run 'rc5des -config', select option %d, and set it\n",CONF_CPUTYPE+1);
    fflush(stdout);
//    for (int i = 0; i < 6; i++)
    for (int i = 0; i < 5; i++)
    {
      Problem problem;
      ContestWork contestwork;
      contestwork.key.lo = contestwork.key.hi = htonl( 0 );
      contestwork.iv.lo = contestwork.iv.hi = htonl( 0 );
      contestwork.plain.lo = contestwork.plain.hi = htonl( 0 );
      contestwork.cypher.lo = contestwork.cypher.hi = htonl( 0 );
      contestwork.keysdone.lo = contestwork.keysdone.hi = htonl( 0 );
      contestwork.iterations.lo = htonl( benchsize );
      contestwork.iterations.hi = htonl( 0 );
      problem.LoadState( &contestwork , 0 ); // RC5 core selection

      // select the correct core engine
      switch(i)
      {
        case 1: rc5_unit_func = rc5_unit_func_486; break;
        case 2: rc5_unit_func = rc5_unit_func_p6; break;
        case 3: rc5_unit_func = rc5_unit_func_6x86; break;
        case 4: rc5_unit_func = rc5_unit_func_k5; break;
        case 5: rc5_unit_func = rc5_unit_func_k6; break;
        default: rc5_unit_func = rc5_unit_func_p5; break;
      }

      #ifndef NEW_STATS_AND_LOGMSG_STUFF
      struct timeval start, stop;
      struct timezone dummy;
      gettimeofday( &start, &dummy );
      #endif
      
      problem.Run( benchsize / PIPELINE_COUNT , 0 );
      
      #ifdef NEW_STATS_AND_LOGMSG_STUFF
        double elapsed = CliGetKeyrateForProblem( &problem );
      #else
      gettimeofday( &stop, &dummy );
      double elapsed = (stop.tv_sec - start.tv_sec) +
                       (((double)stop.tv_usec - (double)start.tv_usec)/1000000.0);
      #endif
//printf("Core %d: %f\n",i,elapsed);

      if (fastcore < 0 || elapsed < fasttime)
        {fastcore = i; fasttime = elapsed;}
    }
  }

  // select the correct core engine
  switch(fastcore)
  {
    case 1:
      LogScreen("Selecting Intel 80386, Intel 80486 code\n");
      rc5_unit_func = rc5_unit_func_486;
      break;
    case 2:
      LogScreen("Selecting Intel Pentium Pro, Intel Pentium II code\n");
      rc5_unit_func = rc5_unit_func_p6;
      break;
    case 3:
      LogScreen("Selecting AMD 486, Cyrix 6x86/6x86MX/M2 code\n");
      rc5_unit_func = rc5_unit_func_6x86;
      break;
    case 4:
      LogScreen("Selecting AMD K5 optimized code\n");
      rc5_unit_func = rc5_unit_func_k5;
      break;
    case 5:
      LogScreen("Selecting AMD K6 optimized code\n");
      rc5_unit_func = rc5_unit_func_k6;
      break;
    default:
      LogScreen("Selecting Intel Pentium, Intel Pentium MMX, Cyrix 5x86 code\n");
      rc5_unit_func = rc5_unit_func_p5;
      break;
  }
#elif (CLIENT_CPU == CPU_STRONGARM)

  int fastcore = cputype;
  if (fastcore == -1)
  {
#if (CLIENT_OS == OS_RISCOS)

    LogScreen("Determining CPU type...\n");
    // determine CPU type from coprocessor
    fastcore = find_core();
#else

    const int benchsize = 50000;
    double fasttime = 0;

    LogScreen("Automatically selecting fastest core...\n");
    LogScreen("This is just a guess based on a small test of each core.  If you know what CPU\n");
    LogScreenf("this machine has, then run 'rc5des -config', select option %d, and set it\n",CONF_CPUTYPE+1);
    fflush(stdout);
//    for (int i = 0; i < 6; i++)
    for (int i = 0; i < 2; i++)
    {
      Problem problem;
      ContestWork contestwork;
      contestwork.key.lo = contestwork.key.hi = htonl( 0 );
      contestwork.iv.lo = contestwork.iv.hi = htonl( 0 );
      contestwork.plain.lo = contestwork.plain.hi = htonl( 0 );
      contestwork.cypher.lo = contestwork.cypher.hi = htonl( 0 );
      contestwork.keysdone.lo = contestwork.keysdone.hi = htonl( 0 );
      contestwork.iterations.lo = htonl( benchsize );
      contestwork.iterations.hi = htonl( 0 );
      problem.LoadState( &contestwork , 0 ); // RC5 core selection

      // select the correct core engine
      switch(i)
      {
        case 1: rc5_unit_func = rc5_unit_func_strongarm; break;
        default: rc5_unit_func = rc5_unit_func_arm; break;
      }

      struct timeval start, stop;
      struct timezone dummy;
      gettimeofday( &start, &dummy );
      problem.Run( benchsize / PIPELINE_COUNT , 0 );
      gettimeofday( &stop, &dummy );
      double elapsed = (stop.tv_sec - start.tv_sec) +
                       (((double)stop.tv_usec - (double)start.tv_usec)/1000000.0);
//printf("Core %d: %f\n",i,elapsed);

      if (fastcore < 0 || elapsed < fasttime)
        {fastcore = i; fasttime = elapsed;}
    }
#endif
  }
  // select the correct core engine
  switch(fastcore)
  {
    case 0:
      LogScreen("Selecting ARM code\n");
      rc5_unit_func = rc5_unit_func_arm;
      break;
    default:
      LogScreen("Selecting StrongARM code\n");
      rc5_unit_func = rc5_unit_func_strongarm;
      break;
  }

#endif
  return 0;
}

// ---------------------------------------------------------------------------

void Client::SetNiceness(void)
{
  // renice maximally
  switch ( niceness )
  {
    case 0:
#if (CLIENT_OS == OS_IRIX)
  schedctl( NDPRI, 0, 200 );
#elif (CLIENT_OS == OS_OS2)
  DosSetPriority( 2, PRTYC_IDLETIME, 0, 0 );
#elif (CLIENT_OS == OS_WIN32)
  SetPriorityClass( GetCurrentProcess(), IDLE_PRIORITY_CLASS );
  SetThreadPriority(GetCurrentThread() ,THREAD_PRIORITY_IDLE );
#elif (CLIENT_OS == OS_MACOS)
  // nothing
#elif (CLIENT_OS == OS_NETWARE)
  // nothing - set dynamically
#elif (CLIENT_OS == OS_VMS)
  nice( 4 );  // Assumes base priority of 4, which is the default
              //   configuration, thus nice to 0 priority (lowest)
              // GO-VMS.COM also sets the priority, so this is only
              //   really needed when GO-VMS.COM isn't used.
#elif (CLIENT_OS == OS_AMIGA)
  SetTaskPri(FindTask(NULL), -20);
#elif (CLIENT_OS == OS_DOSWIN16) || (CLIENT_OS == OS_WIN16)
  // nothing
#elif (CLIENT_OS == OS_BEOS)
  // Main control thread runs at normal priority, since it does very little;
  // priority of crunching threads is set when they are created.
#elif (CLIENT_OS == OS_RISCOS)
  // nothing
#elif (CLIENT_OS == OS_QNX)
  setprio( 0, getprio(0)-1 );
#else
  nice( 19 );
#endif
    break;
    case 1:
#if (CLIENT_OS == OS_OS2)
  DosSetPriority( 2, PRTYC_IDLETIME, 31, 0 );
#elif (CLIENT_OS == OS_WIN32)
  SetPriorityClass( GetCurrentProcess(), IDLE_PRIORITY_CLASS );
#elif (CLIENT_OS == OS_MACOS)
  // nothing
#elif (CLIENT_OS == OS_NETWARE)
  // nothing - set dynamically
#elif (CLIENT_OS == OS_DOSWIN16) || (CLIENT_OS == OS_WIN16)
  // nothing
#elif (CLIENT_OS == OS_VMS)
  nice( 2 );
#elif (CLIENT_OS == OS_AMIGA)
  SetTaskPri(FindTask(NULL), -10);
#elif (CLIENT_OS == OS_BEOS)
  // nothing
#elif (CLIENT_OS == OS_RISCOS)
  // nothing
#elif (CLIENT_OS == OS_QNX)
  setprio( 0, getprio(0)+1 );
#else
  nice( 10 );
#endif
    break;
    case 2:
      // don't do anything
    break;
  }
}

// ---------------------------------------------------------------------------

#if (CLIENT_OS == OS_AMIGA)
/* Disable SAS/C CTRL-C handing */
extern "C" void __regargs __chkabort(void) { return ;}
#endif

// --------------------------------------------------------------------------

#if (CLIENT_OS == OS_WIN32)
bool CliSignalHandler(DWORD  dwCtrlType)
{
  if ( dwCtrlType == CTRL_C_EVENT || dwCtrlType == CTRL_BREAK_EVENT ||
       dwCtrlType == CTRL_CLOSE_EVENT || dwCtrlType == CTRL_SHUTDOWN_EVENT)
  {
    fprintf( stderr, "*Break*\n" );
    SignalTriggered = UserBreakTriggered = 1;
    return TRUE;
  }
  return FALSE;
}
#elif (CLIENT_OS == OS_NETWARE)
void CliSignalHandler( int sig )
{
  int itsAlive;
  unsigned int dieTime = CliGetCurrentTicks() + (30*18); /* 30 secs to die */

  if (sig == SIGABRT )
    ConsolePrintf("RC5DES: Caught an ABORT signal. Please try again after loading MATHLIB.NLM\r\n");

  SignalTriggered = UserBreakTriggered = 1;
  ConsolePrintf("RC5DES: Client is shutting down...\r\n"); //prints appname as well

  while ((itsAlive=CliIsClientRunning())!=0 && CliGetCurrentTicks()<dieTime )
    CliThreadSwitchWithDelay();

  if (itsAlive) /* timed out. If we got here, we're still alive anyway */
    {
    CliActivateConsoleScreen();
    ConsolePrintf("RC5DES: Failed to shutdown gracefully. Forcing exit.\r\n");
    CliForceClientShutdown(); //CliExitClient();  /* kill everything */    
    }
  else
    ConsolePrintf("RC5DES: Client has shut down.\r\n");
  return;
}
#elif (CLIENT_OS != OS_AMIGA) && (CLIENT_OS != OS_WIN16)
  #if (CLIENT_OS == OS_OS390)
    extern "C" void CliSignalHandler( int )
  #else
    void CliSignalHandler( int )
  #endif
{
  #if (CLIENT_OS != OS_DOSWIN16)
  fprintf(stderr, "*Break*\n");
  #endif
  SignalTriggered = UserBreakTriggered = 1;

  #if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOSWIN16)
    signal( SIGINT, CliSignalHandler );
    signal( SIGTERM, CliSignalHandler );
  #elif (CLIENT_OS == OS_RISCOS)
    // nothing
  #elif (CLIENT_OS == OS_NETWARE)
    /* see above. allow default handling otherwise may have infinite loop */
  #elif (CLIENT_OS == OS_BEOS)
    // nothing.  don't need to reregister signal handler
  #elif (CLIENT_OS != OS_MACOS)
    signal( SIGHUP, CliSignalHandler );
    signal( SIGQUIT, CliSignalHandler );
    signal( SIGTERM, CliSignalHandler );
    signal( SIGINT, CliSignalHandler );
    #if (CLIENT_OS != OS_VMS)
      signal( SIGSTOP, CliSignalHandler );
    #endif
  #endif
}
#endif

// --------------------------------------------------------------------------

void CliSetupSignals( void )
{
  SignalTriggered = 0;

  #if (CLIENT_OS == OS_AMIGA) || (CLIENT_OS == OS_WIN16)
    // nothing
  #elif (CLIENT_OS == OS_WIN32)
    SetConsoleCtrlHandler( (PHANDLER_ROUTINE) CliSignalHandler, TRUE );
  #elif (CLIENT_OS == OS_RISCOS)
    signal( SIGINT, CliSignalHandler );
  #elif (CLIENT_OS == OS_NETWARE)
    signal( SIGABRT, CliSignalHandler ); //abort on floating point [...]printf
    signal( SIGINT, CliSignalHandler );  //       and mathlib.nlm isn't loaded
    signal( SIGTERM, CliSignalHandler );
  #elif (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOSWIN16)
    signal( SIGINT, CliSignalHandler );
    signal( SIGTERM, CliSignalHandler );
  #elif (CLIENT_OS != OS_MACOS)
    signal( SIGHUP, CliSignalHandler );
    signal( SIGQUIT, CliSignalHandler );
    signal( SIGTERM, CliSignalHandler );
    signal( SIGINT, CliSignalHandler );
    #if (CLIENT_OS != OS_VMS)
      signal( SIGSTOP, CliSignalHandler );
    #endif
  #endif
}

// --------------------------------------------------------------------------
#ifndef NEW_STATS_AND_LOGMSG_STUFF

#if (CLIENT_OS == OS_MACOS)
//#include <Timer.h>
int gettimeofday(struct timeval *tv, struct timezone *)
{
  unsigned long long t;
  Microseconds( (UnsignedWide *)&t );
  tv->tv_sec = t / 1000000U;
  tv->tv_usec = t % 1000000U;
  return 0;
}
#elif (CLIENT_OS == OS_SCO) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_DOSWIN16) || ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
int gettimeofday(struct timeval *tv, struct timezone *)
{
  struct timeb tb;
  ftime(&tb);
  tv->tv_sec = tb.time;
  tv->tv_usec = tb.millitm*1000;
  return 0;
}
#elif (CLIENT_OS == OS_NETWARE)
unsigned int rawclock(void);
#pragma aux rawclock = "xor eax,eax" "out 0x43,al" "in al,0x40" \
     "mov ah,al" "in al,0x40" "xchg ah,al" "not ax" modify [eax];
#define PCLOCKS_PER_SEC (1193180) //often defined as UCLOCKS_PER_SEC
int gettimeofday(struct timeval *tv, struct timezone *)
{
/*static unsigned int timebase=0;
  unsigned int picsnow = rawclock();
  unsigned int ticksnow = GetCurrentTime();
  unsigned int secs, hsecs, ticks;

  TicksToSeconds( ticksnow, &secs, &hsecs );
  if (timebase==0) timebase = ((unsigned int)(time(NULL))) - secs;
  SecondsToTicks( secs, 0, &ticks); // ticks = (secs*18.207)
  ticksnow-=ticks;

  tv->tv_sec = (time_t)(timebase + secs);
  tv->tv_usec = ((picsnow + (ticksnow << 16))*1000)/(PCLOCKS_PER_SEC/1000);
  if (tv->tv_usec > 1000000)
  {
    tv->tv_sec += (tv->tv_usec/1000000);
    tv->tv_usec = (tv->tv_usec%1000000);
  }
  return 0;
*/
  unsigned int picsnow = rawclock();
  unsigned int ticksnow = CliGetCurrentTicks();
  double dblsecs = ((((double)(ticksnow))*65536.0)+(double)(picsnow))/((double)(PCLOCKS_PER_SEC));
  unsigned int timenow = ((unsigned int)(dblsecs));
  tv->tv_usec = (unsigned int)((dblsecs-(double)(timenow))*(double)(1000000));
  tv->tv_sec  = (time_t)(timenow);
  return 0;

}
#elif (CLIENT_OS == OS_AMIGA)
int gettimeofday(struct timeval *tv, struct timezone *)
{
  return timer((unsigned int *)tv);
}
#endif

// --------------------------------------------------------------------------

const char * Client::Time( void )
{
  time_t timenow;
  struct tm * gmt;
  static char timestring[64];

  // this may need to be tweaked on other platforms
  timenow = time( NULL );
#if (CLIENT_OS == OS_OS2)
  gmt = gmtime( (const time_t *) &timenow);
#else
  gmt = gmtime(&timenow );
#endif
  if (gmt)
  {
    if (gmt->tm_year > 99) {
    sprintf( timestring, "%02d/%02d/%02d %02d:%02d:%02d GMT", gmt->tm_mon + 1,
        gmt->tm_mday, gmt->tm_year - 100, gmt->tm_hour, gmt->tm_min, gmt->tm_sec );
    } else {
    sprintf( timestring, "%02d/%02d/%02d %02d:%02d:%02d GMT", gmt->tm_mon + 1,
        gmt->tm_mday, gmt->tm_year, gmt->tm_hour, gmt->tm_min, gmt->tm_sec );
    }
  } else {
    gmt = localtime(&timenow);
    if (gmt) {
      if (gmt->tm_year > 99) {
      sprintf( timestring, "%02d/%02d/%02d %02d:%02d:%02d local", gmt->tm_mon + 1,
          gmt->tm_mday, gmt->tm_year - 100, gmt->tm_hour, gmt->tm_min, gmt->tm_sec );
      } else {
      sprintf( timestring, "%02d/%02d/%02d %02d:%02d:%02d local", gmt->tm_mon + 1,
          gmt->tm_mday, gmt->tm_year, gmt->tm_hour, gmt->tm_min, gmt->tm_sec );
      }
    } else {
      timestring[0] = 0;
    }
  }
  return( timestring );
}
#endif //#ifndef NEW_STATS_AND_LOGMSG_STUFF

// --------------------------------------------------------------------------

void Client::ParseCommandlineOptions(int argc, char *argv[], s32 &inimissing)
{
  for (int i=1;i<argc;i++)
  {
    if ( strcmp(argv[i], "-percentoff" ) == 0) // This should be checked here, in case it
    {
      percentprintingoff = 1;                 // follows a -benchmark
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-nofallback" ) == 0 ) // Don't try rc5proxy.distributed.net
    {                                                 // After multiple errors
      nofallback=1;
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-quiet" ) == 0 ) // No messages
    {
      quietmode=1;
      argv[i][0] = 0;
    }
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
#if (!defined(WINNTSERVICE))
    else if ( strcmp( argv[i], "-hide" ) == 0 ) // Hide the client
    {
      quietmode=1;
#if (CLIENT_OS == OS_OS2)
      os2hidden=1;
#else
      win95hidden=1;
#endif
      argv[i][0] = 0;
    }
#endif
#endif

#if (CLIENT_OS == OS_WIN32)
    else if ( strcmp( argv[i], "-lurk" ) == 0 ) // Detect modem connections
    {
      lurk=1;
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-lurkonly" ) == 0 ) // Only connect when modem connects
    {
      lurk=2;
      argv[i][0] = 0;
    }
#endif
    else if ( strcmp( argv[i], "-noexitfilecheck" ) == 0 ) // Change network timeout
    {
      noexitfilecheck=1;
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-runoffline" ) == 0 ) // Run offline
    {
      offlinemode=1;
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-runbuffers" ) == 0 ) // Run offline & exit when buffer empty
    {
      offlinemode=2;
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-run" ) == 0 ) // Run online
    {
      offlinemode=0;
      argv[i][0] = 0;
    }
    else if ( strcmp( argv[i], "-nodisk" ) == 0 ) // No disk buff-*.rc5 files.
    {
      nodiskbuffers=1;
      strcpy(checkpoint_file[0],"none");
      strcpy(checkpoint_file[1],"none");
      argv[i][0] = 0;
    }
    else if ( strcmp(argv[i], "-frequent" ) == 0)
    {
      LogScreenf("Setting connections to frequent\n");
      connectoften=1;
      argv[i][0] = 0;
    }
    else if ((i+1) < argc) {
      if ( strcmp( argv[i], "-b" ) == 0 ) // Buffer threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 buffer size to %s\n",argv[i+1]);
        outthreshold[0] = inthreshold[0]  = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-b2" ) == 0 ) // Buffer threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des buffer size to %s\n",argv[i+1]);
        outthreshold[1] = inthreshold[1]  = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-bin" ) == 0 ) // Buffer input threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 input buffer size to %s\n",argv[i+1]);
        inthreshold[0]  = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-bin2" ) == 0 ) // Buffer input threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des input buffer size to %s\n",argv[i+1]);
        inthreshold[1]  = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-bout" ) == 0 ) // Buffer output threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 output buffer size to %s\n",argv[i+1]);
        outthreshold[0]  = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-bout2" ) == 0 ) // Buffer output threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des output buffer size to %s\n",argv[i+1]);
        outthreshold[1]  = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-u" ) == 0 ) // UUE/HTTP Mode
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting uue/http mode to %s\n",argv[i+1]);
        uuehttpmode = (s32) atoi( argv[i+1] );
        firemode = 5;
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-in" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 buffer input file to %s\n",argv[i+1]);
        strcpy(in_buffer_file[0], argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-in2" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des buffer input file to %s\n",argv[i+1]);
        strcpy(in_buffer_file[1], argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-out" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 buffer output file to %s\n",argv[i+1]);
        strcpy(out_buffer_file[0], argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-out2" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des buffer output file to %s\n",argv[i+1]);
        strcpy(out_buffer_file[1], argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-a" ) == 0 ) // Override the keyserver name
      {
        LogScreenf("Setting keyserver to %s\n",argv[i+1]);
        strcpy( keyproxy, argv[i+1] );
        firemode=5;
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-p" ) == 0 ) // Override the keyserver port
      {
        LogScreenf("Setting keyserver port to %s\n",argv[i+1]);
        keyport = (s32) atoi(argv[i+1]);
        firemode=5;
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-ha" ) == 0 ) // Override the http proxy name
      {
        LogScreenf("Setting http proxy to %s\n",argv[i+1]);
        strcpy( httpproxy, argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-hp" ) == 0 ) // Override the http proxy port
      {
        LogScreenf("Setting http proxy port to %s\n",argv[i+1]);
        httpport = (s32) atoi(argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-l" ) == 0 ) // Override the log file name
      {
        LogScreenf("Setting log file to %s\n",argv[i+1]);
        strcpy( logname, argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-smtplen" ) == 0 ) // Override the mail message length
      {
        LogScreenf("Setting Mail message length to %s\n",argv[i+1]);
        messagelen = (s32) atoi(argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-smtpport" ) == 0 ) // Override the smtp port for mailing
      {
        LogScreenf("Setting smtp port to %s\n",argv[i+1]);
        smtpport = (s32) atoi(argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-smtpsrvr" ) == 0 ) // Override the smtp server name
      {
        LogScreenf("Setting smtp server to %s\n",argv[i+1]);
        strcpy(smtpsrvr, argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-smtpfrom" ) == 0 ) // Override the smtp source id
      {
        LogScreenf("Setting smtp 'from' address to %s\n",argv[i+1]);
        strcpy(smtpfrom, argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-smtpdest" ) == 0 ) // Override the smtp destination id
      {
        LogScreenf("Setting smtp 'To' address to %s\n",argv[i+1]);
        strcpy(smtpdest, argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-nettimeout" ) == 0 ) // Change network timeout
      {
        LogScreenf("Setting network timeout to %s\n",argv[i+1]);
        nettimeout = (s32) min(300,max(30,atoi(argv[i+1])));
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-exitfilechecktime" ) == 0 ) // Change network timeout
      {
        exitfilechecktime=max(1,atoi(argv[i+1]));
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-c" ) == 0)      // set cpu type
      {
        cputype = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-e" ) == 0 ) // Override the email id
      {
        LogScreenf("Setting email for notifications to %s\n",argv[i+1]);
        strcpy( id, argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-nice" ) == 0 ) // Nice level
      {
        LogScreenf("Setting nice option to %s\n",argv[i+1]);
        niceness = (s32) atoi( argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-h" ) == 0 ) // Hours to run
      {
        LogScreenf("Setting time limit to %s hours\n",argv[i+1]);
        minutes = (s32) (60. * atol( argv[i+1] ));
        strncpy(hours,argv[i+1],sizeof(hours));
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-n" ) == 0 ) // Blocks to complete in a run
      {
        blockcount = min(1, (s32) atoi( argv[i+1] ));
        LogScreenf("Setting block completion limit to %s\n",blockcount);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( argv[i], "-until" ) == 0 ) // Exit time
      {
        time_t timenow = time( NULL );
        struct tm *gmt = localtime(&timenow );
        minutes = atoi( argv[i+1] );
        minutes = (int)( ( ((int)(minutes/100))*60 + (minutes%100) ) - ((60. * gmt->tm_hour) + gmt->tm_min));
        if (minutes<0) minutes += 24*60;
        if (minutes<0) minutes = 0;
        LogScreenf("Setting time limit to %d minutes\n",minutes);
  #if (CLIENT_OS == OS_NETWARE)
        sprintf(hours,"%u.%02u",minutes/60, minutes%60);
  #else
        sprintf(hours,"%f",minutes/60.);
  #endif
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
  #if defined(MULTITHREAD)
      else if ( strcmp( argv[i], "-numcpu" ) == 0 ) // Override the number of cpus
      {
        LogScreenf("Configuring for %s CPUs\n",argv[i+1]);
        numcpu = (s32) atoi(argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
  #endif
      else if ( strcmp(argv[i], "-ckpoint" ) == 0)
      {
        LogScreenf("Setting rc5 checkpoint file to %s\n",argv[i+1]);
        strcpy(checkpoint_file[0], argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-ckpoint2" ) == 0)
      {
        LogScreenf("Setting des checkpoint file to %s\n",argv[i+1]);
        strcpy(checkpoint_file[1], argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-cktime" ) == 0)
      {
        LogScreenf("Setting checkpointing to %s minutes\n",argv[i+1]);
        checkpoint_min=(s32) atoi(argv[i+1]);
        checkpoint_min=max(2, checkpoint_min);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-pausefile" ) == 0)
      {
        LogScreenf("Setting pause file to %s\n",argv[i+1]);
        strcpy(pausefile, argv[i+1]);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-blsize" ) == 0)
      {
        preferred_blocksize = (s32) atoi(argv[i+1]);
        if (preferred_blocksize < 28) preferred_blocksize = 28;
        if (preferred_blocksize > 31) preferred_blocksize = 31;
        LogScreenf("Setting preferred blocksize to 2^%d\n",preferred_blocksize);
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(argv[i], "-prefer" ) == 0)
      {
        preferred_contest_id = (s32) atoi(argv[i+1]) - 1;
        if (preferred_contest_id == 0) {
          LogScreen("Setting preferred contest to RC5\n");
        } else {
          LogScreen("Setting preferred contest to DES\n");
          preferred_contest_id = 1;
        }
        argv[i][0] = argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
    }
  }
}

// --------------------------------------------------------------------------

void Client::PrintBanner(char * clname)
{
  LogScreenf( "\nRC5DES v2.%d.%d client - a project of distributed.net\n"
          "Copyright distributed.net 1997\n"
#if (CLIENT_CPU == CPU_X86)
          "DES Search routines Copyright Svend Olaf Mikkelsen\n"
#endif
#if defined(KWAN)
              "DES Search routines Copyright Matthew Kwan\n"
#endif
              "Visit http://www.distributed.net/ for more information."
#if (CLIENT_OS == OS_NETWARE)
             "\n\n",CLIENT_CONTEST*100 + CLIENT_BUILD, CLIENT_BUILD_FRAC);
#else
              "  '%s HELP' for usage\n\n",
              CLIENT_CONTEST*100 + CLIENT_BUILD, CLIENT_BUILD_FRAC,clname);
#endif
}

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_X86)
int Client::x86id()
#if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_DOSWIN16)
{
  u32 detectedvalue; //valye x86ident returns, must be interpreted
  int coretouse; // the core the client should use of the 5(6?)

  LogScreen("Beginning CPU identification...\n");
  detectedvalue = x86ident();
  LogScreen("Completed CPU identification. ");
  if ((detectedvalue >> 16)== 0x7943) // Cyrix CPU
  {
    detectedvalue &= 0xfff0; //strip last 4 bits, don't need stepping info
    if (detectedvalue == 0x40)
    {
      LogScreen("Detected a Cyrix 486\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0490)
    {
      LogScreen("Detected a Cyrix 5x86\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0440)
    {
      LogScreen("Detected a Cyrix MediaGX\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0520)
    {
      LogScreen("Detected a Cyrix 6x86\n");
      coretouse=3;
    }
    else if (detectedvalue == 0x0540)
    {
      LogScreen("Detected a Cyrix GXm\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0600)
    {
      LogScreen("Detected a Cyrix 6x86MX\n");
      coretouse=3;
    }
    else
    {
      LogScreen("Detected an Unknown Cyrix Processor\n");
      coretouse=3;
    }
  }
  else if ((detectedvalue >> 16) == 0x6543) //centaur/IDT cpu
  {
    detectedvalue &= 0xfff0; //strip last 4 bits, don't need stepping info
    if (detectedvalue == 0x0540)
    {
    coretouse=0;
    LogScreen("Detected a Centaur C6\n");
    }
  }
  else if ((detectedvalue >> 16) == 0x7541) // AMD CPU
  {
    detectedvalue &= 0xfff0; //strip last 4 bits, don't need stepping info
    if (detectedvalue == 0x40)
    {
      LogScreen("Detected an AMD 486\n");
      coretouse=3;
    }
    else if (detectedvalue == 0x0430)
    {
      coretouse = 3; // 486
      LogScreen("Detected an AMD 486DX2\n");
    }
    else if (detectedvalue == 0x0470)
    {
      coretouse = 3;
      LogScreen("Detected an AMD 486DX2WB\n");
    }
    else if (detectedvalue == 0x0480)
    {
      coretouse = 3;
      LogScreen("Detected an AMD 486DX4\n");
    }
    else if (detectedvalue == 0x0490)
    {
      coretouse = 3;
      LogScreen("Detected an AMD 486DX4WB\n");
    }
    else if (detectedvalue == 0x04E0)
    {
      coretouse = 3;
      LogScreen("Detected an AMD 5x86\n");
    }
    else if (detectedvalue == 0x04F0)
    {
      coretouse = 3;
      LogScreen("Detected an AMD 5x86WB\n");
    }
    else if (detectedvalue == 0x0500)
    {
      coretouse = 4; // K5
      LogScreen("Detected an AMD K5 PR75, PR90, or PR100\n");
    }
    else if (detectedvalue == 0x0510)
    {
      coretouse = 4;
      LogScreen("Detected an AMD K5 PR120 or PR133\n");
    }
    else if (detectedvalue == 0x0520)
    {
      coretouse = 4;
      LogScreen("Detected an AMD K5 PR166");
    }
    else if (detectedvalue == 0x0530)
    {
      coretouse = 4;
      LogScreen("Detected an AMD K5 PR200");
    }
    else if (detectedvalue == 0x0560)
    {
      coretouse = 5; // K6
      LogScreen("Detected an AMD K6\n");
    }
    else if (detectedvalue == 0x0570)
    {
      coretouse = 5;
      LogScreen("Detected an AMD K6\n");
    }
    else if (detectedvalue == 0x0580)
    {
      coretouse = 5;
      LogScreen("Detected an AMD K6-3D\n");
    }
    else if (detectedvalue == 0x0590)
    {
      coretouse = 5;
      LogScreen("Detected an AMD K6-3D+\n");
    }
    else
    {
      coretouse = 5;                    // for the future
      LogScreen("Detected an unknown AMD processor\n");
    }
  }
  else if ((detectedvalue >> 16) == 0x6849 || (detectedvalue >> 16) == 0x6547) // Intel CPU
  {
    detectedvalue &= 0xfff0; //strip last 4 bits, don't need stepping info
    if (detectedvalue == 0x30)
    {
      LogScreen("Detected an 80386\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x40)
    {
      LogScreen("Detected an Intel 486\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0400)
    {
      LogScreen("Detected an Intel 486DX 25 or 33\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0410)
    {
      LogScreen("Detected an Intel 486DX 50\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0420)
    {
      LogScreen("Detected an Intel 486SX\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0430)
    {
      LogScreen("Detected an Intel 486DX2\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0440)
    {
      LogScreen("Detected an Intel 486SL\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0450)
    {
      LogScreen("Detected an Intel 486SX2\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0470)
    {
      LogScreen("Detected an Intel 486DX2WB\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0480)
    {
      LogScreen("Detected an Intel 486DX4\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0490)
    {
      LogScreen("Detected an Intel 486DX4WB\n");
      coretouse=1;
    }
    else if (detectedvalue == 0x0500)
    {
      LogScreen("Detected an Intel Pentium\n"); // stepping A
      coretouse=0;
    }
    else if (detectedvalue == 0x0510)
    {
      LogScreen("Detected an Intel Pentium\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0520)
    {
      LogScreen("Detected an Intel Pentium\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0530)
    {
      LogScreen("Detected an Intel Pentium Overdrive\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0540)
    {
      LogScreen("Detected an Intel Pentium MMX\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0570)
    {
      LogScreen("Detected an Intel Pentium\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0580)
    {
      LogScreen("Detected an Intel Pentium MMX\n");
      coretouse=0;
    }
    else if (detectedvalue == 0x0600)
    {
      LogScreen("Detected an Intel Pentium Pro\n");
      coretouse=2;
    }
    else if (detectedvalue == 0x0610)
    {
      LogScreen("Detected an Intel Pentium Pro\n");
      coretouse=2;
    }
    else if (detectedvalue == 0x0630)
    {
      LogScreen("Detected an Intel Pentium II\n");
      coretouse=2;
    }
    else if (detectedvalue == 0x0650)
    {
      LogScreen("Detected an Intel Pentium II\n");
      coretouse=2;
    }
    else
    {
      coretouse = 2;  //PPro and PII
      LogScreen("Detected an unknown Intel Processor\n");
    }
  }
  else
  {
    LogScreen("Detected an unknown processor from an unknown manufacturer\n");
    coretouse=-1;
  }
  return coretouse;
}
#else
{ return -1; }
#endif
#endif

// --------------------------------------------------------------------------

bool Client::CheckForcedKeyport(void)
{
  bool Forced = false;
  char *dot = strchr(keyproxy, '.');
  if (dot && (strcmpi(dot, ".v27.distributed.net") != 0 ||
      strcmpi(dot, ".distributed.net") != 0))
  {
    int foundport = 2064;
    for (char *p = keyproxy; p < dot; p++)
      if (isdigit(*p)) { foundport = atoi(p); break; }
    if (foundport == 2064 || foundport == 23 || foundport == 80)
    {
      if (keyport != 3064 && keyport != foundport)
      {
        keyport = foundport;
        Forced = true;
      }
    }
  }
  return Forced;
}

// --------------------------------------------------------------------------

