// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "client.h"


// --------------------------------------------------------------------------

#define OPTION_COUNT    43
#define MAXMENUENTRIES  15

char *menutable[5]=
  {
  "Required Options",
  "Logging Options",
  "Communication Options",
  "Performance Options",
  "Miscellaneous Options"
  };

#if (CLIENT_CPU == CPU_X86)
char cputypetable[7][60]=
  {
  "Autodetect",
  "Intel Pentium, Intel Pentium MMX, Cyrix 486/5x86/MediaGX",
  "Intel 80386, Intel 80486",
  "Intel Pentium Pro, Intel Pentium II",
  "AMD 486, Cyrix 6x86/6x86MX/M2",
  "AMD K5",
  "AMD K6",
  };
#elif (CLIENT_CPU == CPU_ARM)
char cputypetable[3][60]=
  {
  "Autodetect",
  "ARM",
  "StrongARM",
  };
#elif (CLIENT_CPU == CPU_POWERPC && (CLIENT_OS == OS_LINUX || CLIENT_OS == OS_AIX))
char cputypetable[3][60]=
  {
  "Autodetect",
  "PowerPC 601",
  "PowerPC 603/604/750",
  };
#endif

char uuehttptable[6][60]=
  {
  "No special encoding",
  "UUE encoding (telnet proxies)",
  "HTTP encoding",
  "HTTP+UUE encoding",
  "SOCKS4 proxy",
  "SOCKS5 proxy"
  };

char contesttable[3][60]=
  {
  "",//need a null placeholder since this is 1/2 based
  "RC5",
  "DES"
  };

char offlinemodetable[3][60]=
  {
  "Normal Operation",
  "Offline Always (no communication)",
  "Finish Buffers and exit"
  };

char lurkmodetable[3][60]=
  {
  "Normal mode",
  "Dial-up detection mode",
  "Dial-up detection ONLY mode"
  };

struct optionstruct
  {
  char *name;//name of the option in the .ini file
  char *description;//description of the option
  char *defaultsetting;//default setting
  char *comments;//additional comments
  s32 optionscreen;//screen to appear on
  s32 type;//type: 0=other, 1=string, 2=integer, 3=boolean (yes/no)
  s32 menuposition;//number on that menu to appear as
  void *thevariable;//pointer to the variable
  char *choicelist;//pointer to the char* array of choices
                   //(used for numeric responses)
  s32 choicemin;//minimum choice number
  s32 choicemax;//maximum choice number
  };

optionstruct options[OPTION_COUNT]=
{
//0
{ "id", "Your E-mail address", "rc5@distributed.net", "(64 characters max)",1,1,1,NULL},
//1
{ "threshold", "RC5 Blocks to Buffer", "10", "(max 1000)",1,2,2,NULL},
//2
{ "threshold", "RC5 block flush threshold", "10",
    "\nSet this equal to RC5 Blocks to Buffer except in rare cases.",5,2,3,NULL},
//3
{ "threshold2", "DES Blocks to Buffer", "10", "(max 1000)",1,2,3,NULL},
//4
{ "threshold2", "DES block flush threshold", "10",
    "\nSet this equal to DES Blocks to Buffer except in rare cases.",5,2,4,NULL},
//5
{ "count", "Complete this many blocks, then exit", "0", "(0 = no limit)",5,2,1,NULL},
//6
{ "hours", "Run for this many hours, then exit", "0.00", "(0 = no limit)",5,1,2,NULL},
//7
{ "timeslice", "Keys per timeslice - for Macs, Win16, RISC OS, etc",
#if (CLIENT_OS == OS_WIN16)
    "200",
#elif (CLIENT_OS == OS_RISCOS)
    "2048",
#else
    "65536",
#endif
    "(0 = default timeslicing)\n"
    "DO NOT TOUCH this unless you know what you're doing!!!",4,2,4,NULL},
//8
{ "niceness", "Level of niceness to run at", "0",
     "\n  mode 0) (recommended) Very nice, should not interfere with any other process\n"
     "  mode 1) Nice, runs with slightly higher priority than idle processes\n"
     "          Same as mode 0 in OS/2 and Win32\n"
     "  mode 2) Normal, runs with same priority as normal user processes\n",4,2,1,NULL},
//9
{ "logname", "File to log to", "", "(128 characters max, blank = no log)\n",2,1,1,NULL},
//10
{ "uuehttpmode", "Firewall Communications mode (UUE/HTTP/SOCKS)", "0",
  "",3,2,1,NULL,&uuehttptable[0][0],0,5},
//11
{ "keyproxy", "Preferred KeyServer Proxy", "us.v27.distributed.net",
   "\nThis specifies the DNS or IP address of the keyserver your client will\n"
   "communicate with. Unless you have a special configuration, use the setting\n"
   "automatically set by the client.",3,1,2,NULL},
//12
{ "keyport", "Preferred KeyServer Port", "2064", "(TCP/IP port on preferred proxy)",3,2,3,NULL},
//13
{ "httpproxy", "Local HTTP/SOCKS proxy address",
       "wwwproxy.corporate.com", "(DNS or IP address)\n",3,1,4,NULL},
//14
{ "httpport", "Local HTTP/SOCKS proxy port", "80", "(TCP/IP port on HTTP proxy)",3,2,5,NULL},
//15
{ "httpid", "HTTP/SOCKS proxy userid/password", "", "(Enter userid (. to reset it to empty) )",3,1,6,NULL},
#if (CLIENT_CPU == CPU_X86)
//16
{ "cputype", "Optimize performance for CPU type", "-1",
      "\n",4,2,3,NULL,&cputypetable[1][0],-1,5},
#elif (CLIENT_CPU == CPU_ARM)
{ "cputype", "Optimize performance for CPU type", "-1",
      "\n",4,2,3,NULL,&cputypetable[1][0],-1,1},
#elif (CLIENT_CPU == CPU_POWERPC && (CLIENT_OS == OS_LINUX || CLIENT_OS == OS_AIX))
//16
{ "cputype", "Optimize performance for CPU type", "-1",
      "\n",4,2,3,NULL,&cputypetable[1][0],-1,1},
#else
//16
{ "cputype", "CPU type...not applicable in this client", "-1", "(default -1)",4,2,8,
  NULL,NULL,0,0},
#endif
//17
{ "messagelen", "Message Mailing (bytes)", "0", "(0=no messages mailed.  10000 recommended.  125000 max.)\n",2,2,2,NULL},
//18
{ "smtpsrvr", "SMTP Server to use", "your.smtp.server", "(128 characters max)",2,1,3,NULL},
//19
{ "smtpport", "SMTP Port", "25", "(SMTP port on mail server -- default 25)",2,2,4,NULL},
//20
{ "smtpfrom", "E-mail address that logs will be mailed from", "RC5notify", "\n(Some servers require this to be a real address)\n",2,1,5,NULL},
//21
{ "smtpdest", "E-mail address to send logs to", "you@your.site", "\n(Full name and site eg: you@your.site.  Comma delimited list permitted)\n",2,1,6,NULL},
//22
#if ((CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_BEOS))
  { "numcpu", "Number of CPUs in this machine", "-1 (autodetect)", "\n"
#else
  { "numcpu", "Number of CPUs in this machine", "1", "\n"
#endif
,4,2,2,NULL},
//23
{ "checkpointfile", "RC5 Checkpoint filename","none","\n(Non-shared file required.  "
#if (CLIENT_OS == OS_RISCOS)
  "ckpoint/rc5"
#else
  "ckpoint.rc5"
#endif
  " recommended.  'none' to disable)\n",1,1,4,NULL},
//24
{ "checkpointfile2", "DES Checkpoint filename","none","\n(Non-shared file required.  "
#if (CLIENT_OS == OS_RISCOS)
  "ckpoint/des"
#else
  "ckpoint.des"
#endif
  " recommended.  'none' to disable)\n",1,1,5,NULL},
//25
{ "randomprefix", "High order byte of random blocks","100","Do not change this",0,2,0,NULL},
//26
{ "preferredblocksize", "Preferred Block Size","30",
  "(2^28 -> 2^31)",5,2,5,NULL},
//27
{ "preferredcontest", "Preferred Contest","2","- DES strongly recommended",5,2,6,
  NULL,&contesttable[0][0],1,2},
//28
{ "quiet", "Disable all screen output? (quiet mode)","no","",5,3,7,NULL},
//29
{ "noexitfilecheck", "Disable exit file checking?","no","",5,3,8,NULL},
//30
{ "percentoff", "Disable block percent completion indicators?","no","",5,3,9,NULL},
//31
{ "frequent", "Attempt keyserver connections frequently?","no","",3,3,6,NULL},
//32
{ "nodisk", "Buffer blocks in RAM only? (no disk I/O)","no",
    "\nNote: This option will cause all buffered, unflushable blocks to be lost\n"
    "during client shutdown!",5,3,10,NULL},
//33
{ "nofallback", "Disable fallback to US Round-Robin?","no",
  "\nIf your specified proxy is down, the client normally falls back\n"
  "to the US Round-Robin (us.v27.distributed.net) - this option causes\n"
  "the client to NEVER attempt a fallback if the local proxy is down.",
  3,3,7,NULL},
//34
{ "cktime", "Interval between saving of checkpoints (minutes):","5",
  "",5,2,11,NULL},
//35
{ "nettimeout", "Network Timeout (seconds)", "60"," ",3,2,8,NULL},
//36
{ "exitfilechecktime", "Exit file check time (seconds)","30","",5,2,12,NULL},
//37
{ "runbuffers", "Offline operation mode","0",
  "\nNormal Operation: The client will connect to a keyserver as needed,\n"
  "        and use random blocks if a keyserver connection cannot be made.\n"
  "Offline Always: The client will never connect to a keyserver, and will\n"
  "        generate random blocks if the block buffers empty.)\n"
  "Finish Buffers and exit: The client will never connect\n"
  "        to a keyserver, and when the block buffers empty, it will\n"
  "        terminate.\n",3,2,9,NULL,&offlinemodetable[0][0],0,2},
//38
{ "lurk", "Modem detection options","0",
  "\nNormal mode: the client will send/receive blocks only when it\n"
  "        empties the in buffer, hits the flush threshold, or the user\n"
  "        specifically requests a flush/fetch.\n"
  "Dial-up detection mode: This acts like mode 0, with the addition\n"
  "        that the client will automatically send/receive blocks when a\n"
  "        dial-up networking connection is established. Modem users\n"
  "        will probably wish to use this option so that their client\n"
  "        never runs out of blocks.\n"
  "Dial-up detection ONLY mode: Like the previous mode, this will cause\n"
  "        the client to automatically send/receive blocks when\n"
  "        connected. HOWEVER, if the client runs out of blocks,\n"
  "        it will NOT trigger auto-dial, and will instead work\n"
  "        on random blocks until a connection is detected.\n",
  3,2,10,NULL,&lurkmodetable[0][0],0,2},
{ "in",  "RC5 In-Buffer Path/Name", "[Current Path]\\buff-in.rc5","",0,1,13,NULL},
{ "out", "RC5 Out-Buffer Path/Name", "[Current Path]\\buff-out.rc5","",0,1,14,NULL},
{ "in2", "DES In-Buffer Path/Name", "[Current Path]\\buff-in.des","",0,1,15,NULL},
{ "out2","DES Out-Buffer Path/Name","[Current Path]\\buff-out.des","",0,1,16,NULL}
};

#define CONF_ID 0
#define CONF_THRESHOLDI 1
#define CONF_THRESHOLDO 2
#define CONF_THRESHOLDI2 3
#define CONF_THRESHOLDO2 4
#define CONF_COUNT 5
#define CONF_HOURS 6
#define CONF_TIMESLICE 7
#define CONF_NICENESS 8
#define CONF_LOGNAME 9
#define CONF_UUEHTTPMODE 10
#define CONF_KEYPROXY 11
#define CONF_KEYPORT 12
#define CONF_HTTPPROXY 13
#define CONF_HTTPPORT 14
#define CONF_HTTPID 15
#define CONF_CPUTYPE 16
#define CONF_MESSAGELEN 17
#define CONF_SMTPSRVR 18
#define CONF_SMTPPORT 19
#define CONF_SMTPFROM 20
#define CONF_SMTPDEST 21
#define CONF_NUMCPU 22
#define CONF_CHECKPOINT 23
#define CONF_CHECKPOINT2 24
#define CONF_RANDOMPREFIX 25
#define CONF_PREFERREDBLOCKSIZE 26
#define CONF_PREFERREDCONTEST 27
#define CONF_QUIETMODE 28
#define CONF_NOEXITFILECHECK 29
#define CONF_PERCENTOFF 30
#define CONF_FREQUENT 31
#define CONF_NODISK 32
#define CONF_NOFALLBACK 33
#define CONF_CKTIME 34
#define CONF_NETTIMEOUT 35
#define CONF_EXITFILECHECKTIME 36
#define CONF_OFFLINEMODE 37
#define CONF_LURKMODE 38
#define CONF_RC5IN 39
#define CONF_RC5OUT 40
#define CONF_DESIN 41
#define CONF_DESOUT 42

// --------------------------------------------------------------------------

s32 Client::ConfigureGeneral( s32 currentmenu )
{
  char parm[128],parm2[128];
  s32 choice=1;
  s32 temp;
  s32 temp2;
  s32 contestidtemp;//since it's 0/1 based now, we need a temp for
                    //screen I/O
  char str[3];
  char *p;
#if (CLIENT_OS == OS_WIN32) && defined(MULTITHREAD)
  SYSTEM_INFO systeminfo;
#elif (CLIENT_OS == OS_BEOS)
  system_info the_info;
#endif

  while ( 1 )
  {
setupoptions();
options[CONF_PREFERREDCONTEST].thevariable=&contestidtemp;
contestidtemp=preferred_contest_id+1;

    // display menu

clearscreen();
printf("Distributed.Net RC5/DES Client build v2.70%i.%i config menu\n",CLIENT_BUILD,CLIENT_BUILD_FRAC);
printf("%s\n",menutable[currentmenu-1]);
printf("------------------------------------------------------------\n\n");

for ( temp2=1; temp2 < MAXMENUENTRIES; temp2++ )
    {
      choice=findmenuoption(currentmenu,temp2);
      if (choice >= 0)
          {
          printf("%2d) %s ==> ",
                  options[choice].menuposition, options[choice].description);

          if (options[choice].type==1)
             {
             if (options[choice].thevariable != NULL)
               printf("%s\n",(char *)options[choice].thevariable);
             }
          else if (options[choice].type==2)
             if (options[choice].choicelist == NULL)
               printf("%li\n",(long)*(s32 *)options[choice].thevariable);
             else printf("%s\n",options[choice].choicelist+
             ((long)*(s32 *)options[choice].thevariable*60));
          else if (options[choice].type==3)
             {
             sprintf(str, "%s", *(s32 *)options[choice].thevariable?"yes":"no");
             printf("%s\n",str);
             };
          };
    }
    printf("\n 0) Return to main menu\n");


    // get choice from user
    while(1)
    {
      printf("\nChoice --> ");
      fflush( stdout );
      fgets(parm, 128, stdin);
      choice = atoi(parm);

      if (choice == 0) return 1;

      choice=findmenuoption(currentmenu,choice);

      if (choice >= 0)
        break;
    }



    // prompt for new value
    if (options[choice].type==1)
      printf("\n%s %s\nDefault Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
              options[choice].description, options[choice].comments,
              options[choice].defaultsetting,(char *)options[choice].thevariable);
    else if (options[choice].type==2)
      if (options[choice].choicelist == NULL)
        printf("\n%s %s\nDefault Setting: %s\nCurrent Setting: %li\nNew Setting --> ",
                options[choice].description, options[choice].comments,
                options[choice].defaultsetting, (long)*(s32 *)options[choice].thevariable);
      else {
        printf("\n%s %s\n",options[choice].description,options[choice].comments);
           for ( temp = options[choice].choicemin;
                 temp < options[choice].choicemax+1; temp++)
             {
             printf("  %2d) %s\n",temp,options[choice].choicelist+temp*60);
             }
           printf("\nDefault Setting: %s\nCurrent Setting: %s\nNew Setting --> ",
                  options[choice].choicelist+atoi(options[choice].defaultsetting)*60,
                  options[choice].choicelist+
                  ((long)*(s32 *)options[choice].thevariable*60));
           }
    else if (options[choice].type==3)
      {
      printf("\n%s %s\nDefault Setting: %s\nCurrent Setting: ",
              options[choice].description, options[choice].comments,
              options[choice].defaultsetting);
      sprintf(str, "%s", *(s32 *)options[choice].thevariable?"yes":"no");
      printf("%s\nNew Setting --> ",str);
      };

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
          char *whitespacekiller;
          strncpy( id, parm, sizeof(id) - 1 );
          while (strchr(id, ' ') != NULL)
            {
            whitespacekiller=strchr(id, ' ');
            strncpy(whitespacekiller, whitespacekiller+1,
                    sizeof(id)-(whitespacekiller+1-id));
            };
          break;
        case CONF_THRESHOLDI:
          inthreshold[0]=atoi(parm);
          if ( inthreshold[0] < 1   ) inthreshold[0] = 1;
          if ( inthreshold[0] > 1000 ) inthreshold[0] = 1000;
          outthreshold[0]=inthreshold[0];
          break;
        case CONF_THRESHOLDO:
          outthreshold[0]=atoi(parm);
          if ( outthreshold[0] < 1   ) outthreshold[0] = 1;
          if ( outthreshold[0] > 1000 ) outthreshold[0] = 1000;
          if ( outthreshold[0] > inthreshold[0] )
             outthreshold[0]=inthreshold[0];
          break;
        case CONF_THRESHOLDI2:
          inthreshold[1]=atoi(parm);
          if ( inthreshold[1] < 1   ) inthreshold[1] = 1;
          if ( inthreshold[1] > 1000 ) inthreshold[1] = 1000;
          outthreshold[1]=inthreshold[1];
          break;
        case CONF_THRESHOLDO2:
          outthreshold[1]=atoi(parm);
          if ( outthreshold[1] < 1   ) outthreshold[1] = 1;
          if ( outthreshold[1] > 1000 ) outthreshold[1] = 1000;
          if ( outthreshold[1] > inthreshold[1] )
             outthreshold[1]=inthreshold[1];
          break;
        case CONF_COUNT:
          blockcount = atoi(parm);
          if (blockcount < 0)
            blockcount = 0;
          break;
        case CONF_HOURS:
          minutes = (s32) (60. * atol(parm));
          if ( minutes < 0 ) minutes = 0;
          sprintf(hours,"%u.%02u", (unsigned)(minutes/60),
          (unsigned)(minutes%60)); //1.000000 hours looks silly          sprintf( hours, "%d", minutes/60);
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
        case CONF_KEYPROXY:
          strncpy( keyproxy, parm, sizeof(keyproxy) - 1 );
          CheckForcedKeyport();
          break;
        case CONF_KEYPORT:
          keyport = atoi(parm); CheckForcedKeyport();
          break;
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
          switch (uuehttpmode)
            {
            case 0:strcpy( keyproxy, "us.v27.distributed.net");//normal communications
                   keyport=2064;
                   break;
            case 1:strcpy( keyproxy, "us23.v27.distributed.net");//UUE mode (telnet)
                   keyport=23;
                   break;
            case 2:strcpy( keyproxy, "us80.v27.distributed.net");//HTTP mode
                   keyport=80;
                   break;
            case 3:strcpy( keyproxy, "us80.v27.distributed.net");//HTTP+UUE mode
                   keyport=80;
                   break;
            case 4:strcpy( keyproxy, "us.v27.distributed.net");//SOCKS4
                   keyport=2064;
                   break;
            case 5:strcpy( keyproxy, "us.v27.distributed.net");//SOCKS5
                   keyport=2064;
                   break;
            };
          if (uuehttpmode > 1)
            {
            options[CONF_HTTPPROXY].optionscreen=3;
            options[CONF_HTTPPORT].optionscreen=3;
            options[CONF_HTTPID].optionscreen=3;
            }
            else
            {
            options[CONF_HTTPPROXY].optionscreen=0;
            options[CONF_HTTPPORT].optionscreen=0;
            options[CONF_HTTPID].optionscreen=0;
            };
          break;
#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_ARM)
        case CONF_CPUTYPE:
          cputype = atoi(parm);
          if (cputype < -1 ||
              cputype > options[CONF_CPUTYPE].choicemax)
            cputype = -1;
          break;
#elif (CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))
        case CONF_CPUTYPE:
          cputype = atoi(parm);
          if (cputype < -1 ||
              cputype > options[CONF_CPUTYPE].choicemax)
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
          if (messagelen != 0)
            {
            options[CONF_SMTPSRVR].optionscreen=2;
            options[CONF_SMTPPORT].optionscreen=2;
            options[CONF_SMTPDEST].optionscreen=2;
            options[CONF_SMTPFROM].optionscreen=2;
            }
          else
            {
            options[CONF_SMTPSRVR].optionscreen=0;
            options[CONF_SMTPPORT].optionscreen=0;
            options[CONF_SMTPDEST].optionscreen=0;
            options[CONF_SMTPFROM].optionscreen=0;
            };
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
        case CONF_QUIETMODE:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_QUIETMODE].thevariable=choice;
          break;
        case CONF_NOEXITFILECHECK:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_NOEXITFILECHECK].thevariable=choice;
          break;
        case CONF_PERCENTOFF:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_PERCENTOFF].thevariable=choice;
          break;
        case CONF_FREQUENT:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_FREQUENT].thevariable=choice;
          break;
        case CONF_NODISK:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_NODISK].thevariable=choice;
          break;
        case CONF_NOFALLBACK:
          choice=yesno(parm);
          if (choice >= 0) *(s32 *)options[CONF_NOFALLBACK].thevariable=choice;
          break;
        case CONF_CKTIME:
          choice=atoi(parm);
          choice=max(2,choice);
          *(s32 *)options[CONF_CKTIME].thevariable=choice;
          break;
        case CONF_NETTIMEOUT:
          choice=atoi(parm);
          choice=min(300,max(30,choice));
          *(s32 *)options[CONF_NETTIMEOUT].thevariable=choice;
          break;
        case CONF_EXITFILECHECKTIME:
          choice=atoi(parm);
          choice=max(choice,1);
          *(s32 *)options[CONF_EXITFILECHECKTIME].thevariable=choice;
          break;
        case CONF_OFFLINEMODE:
          choice=atoi(parm);
          if (choice < 0) choice=0;
          if (choice > 2) choice=2;
          *(s32 *)options[CONF_OFFLINEMODE].thevariable=choice;
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_OS2)
        case CONF_LURKMODE:
          choice=atoi(parm);
          if (choice < 0) choice=0;
          if (choice > 2) choice=0;
          if (choice==0)
            {
            choice=0;lurk=0;connectoften=0;
            }
          else if (choice==1) lurk=1;
          else if (choice==2)
            {
            lurk=2;
            connectoften=0;
            };
#endif
        case CONF_RC5IN:
          strncpy( in_buffer_file[0] , parm, sizeof(in_buffer_file)/2 -1 );
          break;
        case CONF_RC5OUT:
          strncpy( out_buffer_file[0] , parm, sizeof(out_buffer_file)/2 -1 );
          break;
        case CONF_DESIN:
          strncpy( in_buffer_file[1] , parm, sizeof(in_buffer_file)/2 -1 );
          break;
        case CONF_DESOUT:
          strncpy( out_buffer_file[1] , parm, sizeof(out_buffer_file)/2 -1 );
          break;

        default:
          break;
      }
    }
  }
}

//----------------------------------------------------------------------------

s32 Client::Configure( void )
//A return of 1 indicates to save the changed configuration
//A return of -1 indicates to NOT save the changed configuration
{

  s32 choice;
  char parm[128];
  s32 returnvalue=0;

while (returnvalue == 0)
   {
   clearscreen();
   printf("Distributed.Net RC5/DES Client build v2.70%i.%i config menu\n",CLIENT_BUILD,CLIENT_BUILD_FRAC);
   printf("------------------------------------------------------------\n\n");
   printf(" 1) %s\n",menutable[0]);
   printf(" 2) %s\n",menutable[1]);
   printf(" 3) %s\n",menutable[2]);
   printf(" 4) %s\n",menutable[3]);
   printf(" 5) %s\n\n",menutable[4]);
   printf(" 9) Discard settings and exit\n");
   printf(" 0) Save settings and exit\n\n");
   if (strcmpi(id,"rc5@distributed.net")==0)
     printf("*Note: You have not yet configured your e-mail address.\n"
            "       Please go to %s and configure it.\n",menutable[0]);
   printf("Choice --> ");

   fflush( stdout );
   fgets(parm, 128, stdin);
   choice = atoi(parm);

   switch (choice)
      {
      case 1: ConfigureGeneral(1);break;
      case 2: ConfigureGeneral(2);break;
      case 3: ConfigureGeneral(3);break;
      case 4: ConfigureGeneral(4);break;
      case 5: ConfigureGeneral(5);break;
      case 0: returnvalue=1;break; //Breaks and tells it to save
      case 9: returnvalue=-1;break; //Breaks and tells it NOT to save
      };

   }

  return returnvalue;
}

//----------------------------------------------------------------------------

s32 Client::yesno(char *str)
// checks for user to type yes or no.
// Returns 1=yes, 0=no, -1=unknown

{
  s32 returnvalue;

  returnvalue=-1;
  if (strcmpi(str, "yes")==0) returnvalue=1;
  if (strcmpi(str, "no")==0) returnvalue=0;
  fflush( stdin );
  return returnvalue;
}

//----------------------------------------------------------------------------

s32 Client::findmenuoption( s32 menu, s32 option)
    // Returns the id of the option that matches the menu and option
    // requested. Will return -1 if not found.
{
s32 returnvalue=-1;
s32 temp;

for (temp=0; temp < OPTION_COUNT; temp++)
  {
  if ((options[temp].optionscreen==menu) &&
      (options[temp].menuposition==option))

     returnvalue=temp;
  };

return returnvalue;
}

//----------------------------------------------------------------------------

void Client::setupoptions( void )
// Sets all the pointers/etc for optionstruct options
{

options[CONF_ID].thevariable=&id;
options[CONF_THRESHOLDI].thevariable=&inthreshold[0];
options[CONF_THRESHOLDO].thevariable=&outthreshold[0];
options[CONF_THRESHOLDI2].thevariable=&inthreshold[1];
options[CONF_THRESHOLDO2].thevariable=&outthreshold[1];
options[CONF_COUNT].thevariable=&blockcount;
options[CONF_HOURS].thevariable=&hours;
#if !((CLIENT_OS==OS_MACOS) || (CLIENT_OS==OS_RISCOS) || (CLIENT_OS==OS_WIN16))
options[CONF_TIMESLICE].optionscreen=0;
#endif
options[CONF_TIMESLICE].thevariable=&timeslice;
options[CONF_NICENESS].thevariable=&niceness;
options[CONF_LOGNAME].thevariable=&logname;
options[CONF_UUEHTTPMODE].thevariable=&uuehttpmode;
options[CONF_KEYPROXY].thevariable=&keyproxy;
options[CONF_KEYPORT].thevariable=&keyport;
options[CONF_HTTPPROXY].thevariable=&httpproxy;
options[CONF_HTTPPORT].thevariable=&httpport;
options[CONF_HTTPID].thevariable=&httpid;
#if !((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_ARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
options[CONF_CPUTYPE].optionscreen=0;
#endif
options[CONF_CPUTYPE].thevariable=&cputype;
options[CONF_MESSAGELEN].thevariable=&messagelen;
options[CONF_SMTPSRVR].thevariable=&smtpsrvr;
options[CONF_SMTPPORT].thevariable=&smtpport;
options[CONF_SMTPFROM].thevariable=&smtpfrom;
options[CONF_SMTPDEST].thevariable=&smtpdest;
#if !defined(MULTITHREAD)
options[CONF_NUMCPU].optionscreen=0;
#endif
options[CONF_NUMCPU].thevariable=&numcpu;
options[CONF_CHECKPOINT].thevariable=&checkpoint_file[0];
options[CONF_CHECKPOINT2].thevariable=&checkpoint_file[1];
options[CONF_RANDOMPREFIX].thevariable=&randomprefix;
options[CONF_PREFERREDBLOCKSIZE].thevariable=&preferred_blocksize;
options[CONF_QUIETMODE].thevariable=&quietmode;
options[CONF_NOEXITFILECHECK].thevariable=&noexitfilecheck;
options[CONF_PERCENTOFF].thevariable=&percentprintingoff;
#if !defined(MULTITHREAD)
options[CONF_FREQUENT].optionscreen=0;
#endif
options[CONF_FREQUENT].thevariable=&connectoften;
options[CONF_NODISK].thevariable=&nodiskbuffers;
options[CONF_NOFALLBACK].thevariable=&nofallback;
options[CONF_CKTIME].thevariable=&checkpoint_min;
options[CONF_NETTIMEOUT].thevariable=&nettimeout;
options[CONF_EXITFILECHECKTIME].thevariable=&exitfilechecktime;
options[CONF_OFFLINEMODE].thevariable=&offlinemode;

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
options[CONF_LURKMODE].thevariable=&lurk;
#else
options[CONF_LURKMODE].optionscreen=0;
#endif
options[CONF_RC5IN].thevariable=&in_buffer_file[0];
options[CONF_RC5OUT].thevariable=&out_buffer_file[0];
options[CONF_DESIN].thevariable=&in_buffer_file[1];
options[CONF_DESOUT].thevariable=&out_buffer_file[1];

if (messagelen != 0)
  {
  options[CONF_SMTPSRVR].optionscreen=2;
  options[CONF_SMTPPORT].optionscreen=2;
  options[CONF_SMTPDEST].optionscreen=2;
  options[CONF_SMTPFROM].optionscreen=2;
  }
  else
  {
  options[CONF_SMTPSRVR].optionscreen=0;
  options[CONF_SMTPPORT].optionscreen=0;
  options[CONF_SMTPDEST].optionscreen=0;
  options[CONF_SMTPFROM].optionscreen=0;
  };

if (uuehttpmode > 1)
  {
  options[CONF_HTTPPROXY].optionscreen=3;
  options[CONF_HTTPPORT].optionscreen=3;
  options[CONF_HTTPID].optionscreen=3;
  }
  else
  {
  options[CONF_HTTPPROXY].optionscreen=0;
  options[CONF_HTTPPORT].optionscreen=0;
  options[CONF_HTTPID].optionscreen=0;
  };


}

//----------------------------------------------------------------------------


void Client::clearscreen( void )
// Clears the screen. (Platform specific ifdefs go inside of it.)

{
#if (CLIENT_OS == OS_WIN32)
  HANDLE hStdout;
  CONSOLE_SCREEN_BUFFER_INFO csbiInfo;
  DWORD nLength;
  COORD topleft = {0,0};

  hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hStdout == INVALID_HANDLE_VALUE) return;
  if (! GetConsoleScreenBufferInfo(hStdout, &csbiInfo)) return;
  nLength = csbiInfo.dwSize.X * csbiInfo.dwSize.Y;

  FillConsoleOutputCharacter(hStdout, ' ', nLength, topleft, NULL);
  FillConsoleOutputAttribute(hStdout, csbiInfo.wAttributes, nLength, topleft, NULL);
  SetConsoleCursorPosition(hStdout, topleft);
#elif (CLIENT_OS == OS_OS2)
  BYTE space[] = " ";
  VioScrollUp(0, 0, -1, -1, -1, space, 0);
#endif

}

//----------------------------------------------------------------------------

s32 Client::ReadConfig(void)
{
  IniSection ini;
  s32 inierror, tempconfig;
  char *p, buffer[64];
  char *whitespacekiller;

  inierror = ini.ReadIniFile( inifilename );
  if ( inierror )
  {
    LogScreen( "Error reading ini file - Using defaults\n" );
  }

#define INIGETKEY(key) (ini.getkey(OPTION_SECTION, options[key].name, options[key].defaultsetting)[0])

  INIGETKEY(CONF_ID).copyto(id, sizeof(id));
  while (strchr(id, ' ') != NULL)
    {
    whitespacekiller=strchr(id, ' ');
    strncpy(whitespacekiller, whitespacekiller+1,
    sizeof(id)-(whitespacekiller+1-id));
    };// removes whitespace in the address

  INIGETKEY(CONF_THRESHOLDI).copyto(buffer, sizeof(buffer));
  p = strchr( buffer, ':' );
  if (p == NULL) {
    outthreshold[0]=inthreshold[0]=atoi(buffer);
  } else {
    outthreshold[0]=atoi(p+1);
    *p=0;
    inthreshold[0]=atoi(buffer);
  }
  INIGETKEY(CONF_THRESHOLDI2).copyto(buffer, sizeof(buffer));
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
  INIGETKEY(CONF_KEYPROXY).copyto(keyproxy, sizeof(keyproxy));
  keyport = INIGETKEY(CONF_KEYPORT);
  INIGETKEY(CONF_HTTPPROXY).copyto(httpproxy, sizeof(httpproxy));
  httpport = INIGETKEY(CONF_HTTPPORT);
  uuehttpmode = INIGETKEY(CONF_UUEHTTPMODE);
  INIGETKEY(CONF_HTTPID).copyto(httpid, sizeof(httpid));
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_ARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
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
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_OS2)
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
#if defined(NEEDVIRTUALMETHODS)
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
  if ( timeslice < PIPELINE_COUNT ) timeslice=PIPELINE_COUNT;
  if ( niceness < 0 || niceness > 2 ) niceness = 0;
  if ( uuehttpmode < 0 || uuehttpmode > 5 ) uuehttpmode = 0;
#if (CLIENT_CPU == CPU_X86)
  if ( cputype < -1 || cputype > 5) cputype = -1;
#elif ((CLIENT_CPU == CPU_ARM) || ((CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_AIX))) )
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
#if defined(NEEDVIRTUALMETHODS)
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

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, options[key].name, IniString(value))

  INISETKEY( CONF_ID, id );
  sprintf(buffer,"%d:%d",(int)inthreshold[0],(int)outthreshold[0]);
  INISETKEY( CONF_THRESHOLDI, buffer );
  sprintf(buffer,"%d:%d",(int)inthreshold[1],(int)outthreshold[1]);
  INISETKEY( CONF_THRESHOLDI2, buffer );
  INISETKEY( CONF_COUNT, blockcount );
  sprintf(hours,"%u.%02u", (unsigned)(minutes/60),
    (unsigned)(minutes%60)); //1.000000 hours looks silly
  INISETKEY( CONF_HOURS, hours );
  INISETKEY( CONF_TIMESLICE, timeslice );
  INISETKEY( CONF_NICENESS, niceness );
  INISETKEY( CONF_LOGNAME, logname );
  INISETKEY( CONF_KEYPROXY, keyproxy );
  INISETKEY( CONF_KEYPORT, keyport );
  INISETKEY( CONF_HTTPPROXY, httpproxy );
  INISETKEY( CONF_HTTPPORT, httpport );
  INISETKEY( CONF_UUEHTTPMODE, uuehttpmode );
  INISETKEY( CONF_HTTPID, httpid);
#if ((CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_ARM) || ((CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS == OS_LINUX || CLIENT_OS == OS_AIX)) )
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
  INISETKEY( CONF_PREFERREDBLOCKSIZE, preferred_blocksize );
  INISETKEY( CONF_PREFERREDCONTEST, (s32)(preferred_contest_id + 1) );
  INISETKEY( CONF_QUIETMODE, quietmode );
  INISETKEY( CONF_NOEXITFILECHECK, noexitfilecheck );
  INISETKEY( CONF_PERCENTOFF, percentprintingoff );
  INISETKEY( CONF_FREQUENT, connectoften );
  INISETKEY( CONF_NODISK, nodiskbuffers );
  INISETKEY( CONF_NOFALLBACK, nofallback );
  INISETKEY( CONF_CKTIME, checkpoint_min );
  INISETKEY( CONF_NETTIMEOUT, nettimeout );
  INISETKEY( CONF_EXITFILECHECKTIME, exitfilechecktime );
//  INISETKEY( CONF_RC5IN, in_buffer_file[0]);
//  INISETKEY( CONF_RC5OUT, out_buffer_file[0]);
//  INISETKEY( CONF_DESIN, in_buffer_file[1]);
//  INISETKEY( CONF_DESOUT, out_buffer_file[1]);

  if (offlinemode == 0)
    {
    IniRecord *tempptr;
    tempptr = ini.findfirst(OPTION_SECTION, "runbuffers");
    if (tempptr) tempptr->values.Erase();
    tempptr=NULL;
    tempptr = ini.findfirst(OPTION_SECTION, "runoffline");
    if (tempptr) tempptr->values.Erase();
    }
  else if (offlinemode == 1)
    {
    IniRecord *tempptr;
    s32 tempvalue;
    tempptr = ini.findfirst(OPTION_SECTION, "runbuffers");
    if (tempptr) tempptr->values.Erase();
    tempvalue=1;
    ini.setrecord(OPTION_SECTION, "runoffline", IniString(tempvalue));
    }
  else if (offlinemode == 2)
    {
    IniRecord *tempptr;
    s32 tempvalue;
    tempptr = ini.findfirst(OPTION_SECTION, "runoffline");
    if (tempptr) tempptr->values.Erase();
    tempvalue=1;
    ini.setrecord(OPTION_SECTION, "runbuffers", IniString(tempvalue));
    };

#undef INISETKEY

  ini.setrecord(OPTION_SECTION, "contestdone",  IniString(contestdone[0]));
  ini.setrecord(OPTION_SECTION, "contestdone2", IniString(contestdone[1]));

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS==OS_OS2)

  if (lurk==0)
    {
    IniRecord *tempptr;
    tempptr = ini.findfirst(OPTION_SECTION, "lurk");
    if (tempptr) tempptr->values.Erase();
    tempptr = ini.findfirst(OPTION_SECTION, "lurkonly");
    if (tempptr) tempptr->values.Erase();
    }
  else if (lurk==1)
    {
    IniRecord *tempptr;
    s32 tempvalue=1;
    tempptr = ini.findfirst(OPTION_SECTION, "lurkonly");
    if (tempptr) tempptr->values.Erase();
    ini.setrecord(OPTION_SECTION, "lurk",  IniString(tempvalue));
    }
  else if (lurk==2)
    {
    IniRecord *tempptr;
    s32 tempvalue=1;
    tempptr = ini.findfirst(OPTION_SECTION, "lurk");
    if (tempptr) tempptr->values.Erase();
    ini.setrecord(OPTION_SECTION, "lurkonly",  IniString(tempvalue));
    };
#endif

#define INIFIND(key) ini.findfirst(OPTION_SECTION, options[key].name)

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

#if defined(NEEDVIRTUALMETHODS)
  InternalWriteConfig(ini);
#endif

#undef INIFIND

  return( ini.WriteIniFile(inifilename) ? -1 : 0 );
}

// --------------------------------------------------------------------------

s32 Client::WriteContestandPrefixConfig(void)
    // returns -1 on error, 0 otherwise
    // only writes contestdone and randomprefix .ini entries
{
  IniSection ini;
  char buffer[64];

  ini.ReadIniFile(inifilename);

#define INISETKEY(key, value) ini.setrecord(OPTION_SECTION, options[key].name, IniString(value))

  INISETKEY( CONF_RANDOMPREFIX, randomprefix );

#undef INISETKEY

  ini.setrecord(OPTION_SECTION, "contestdone",  IniString(contestdone[0]));
  ini.setrecord(OPTION_SECTION, "contestdone2", IniString(contestdone[1]));

#define INIFIND(key) ini.findfirst(OPTION_SECTION, options[key].name)

#if defined(NEEDVIRTUALMETHODS)
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
void ServiceMain(DWORD Argc, LPTSTR *Argv)
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
#if ((CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU != CPU_POWERPC))
  if (!(SysBase->AttnFlags & AFF_68020))
  {
    LogScreen("\nIncompatible CPU type.  Sorry.\n");
    return -1;
  }
#elif (CLIENT_CPU == CPU_POWERPC) && ((CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_AMIGAOS))
  // Be OS isn't supported on 601 machines
  // There is no 601 PPC board for the Amiga
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
        if (fastcore < 0 || elapsed > fasttime)
          {fastcore = i; fasttime = elapsed;}
      #else
      gettimeofday( &stop, &dummy );
      double elapsed = (stop.tv_sec - start.tv_sec) +
                       (((double)stop.tv_usec - (double)start.tv_usec)/1000000.0);

      if (fastcore < 0 || elapsed < fasttime)
        {fastcore = i; fasttime = elapsed;}
      #endif
//printf("Core %d: %f\n",i,elapsed);
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
#elif (CLIENT_CPU == CPU_ARM)
  int fastcore = cputype;
#if (CLIENT_OS == OS_RISCOS)
  if (fastcore == -1) fastcore = ARMid(); // will return -1 if unable to identify
#endif
  if (fastcore == -1)
  {
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
#elif (CLIENT_OS == OS_AMIGAOS)
  SetTaskPri(FindTask(NULL), -20);
#elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16)
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
#elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_WIN16)
  // nothing
#elif (CLIENT_OS == OS_VMS)
  nice( 2 );
#elif (CLIENT_OS == OS_AMIGAOS)
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

#if (CLIENT_OS == OS_AMIGAOS)
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
#elif (CLIENT_OS != OS_AMIGAOS) && (CLIENT_OS != OS_WIN16)
  #if (CLIENT_OS == OS_OS390)
    extern "C" void CliSignalHandler( int )
  #else
    void CliSignalHandler( int )
  #endif
{
  #if (CLIENT_OS != OS_DOS)
  #if (CLIENT_OS == OS_RISCOS)
  if (!guiriscos)
  #endif
    fprintf(stderr, "*Break*\n");
  #endif
  SignalTriggered = UserBreakTriggered = 1;

  #if (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOS)
    signal( SIGINT, CliSignalHandler );
    signal( SIGTERM, CliSignalHandler );
  #elif (CLIENT_OS == OS_RISCOS)
    _kernel_escape_seen(); // clear escape flag for polling check in Problem::Run
    signal( SIGINT, CliSignalHandler );
  #elif (CLIENT_OS == OS_NETWARE)
    /* see above. allow default handling otherwise may have infinite loop */
  #elif (CLIENT_OS == OS_BEOS)
    // nothing.  don't need to reregister signal handler
  #elif (CLIENT_OS == OS_IRIX) && defined(__GNUC__)
    signal( SIGHUP, (void(*)(...)) CliSignalHandler );
    signal( SIGQUIT, (void(*)(...)) CliSignalHandler );
    signal( SIGTERM, (void(*)(...)) CliSignalHandler );
    signal( SIGINT, (void(*)(...)) CliSignalHandler );
    signal( SIGSTOP, (void(*)(...)) CliSignalHandler );
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

  #if (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_WIN16)
    // nothing
  #elif (CLIENT_OS == OS_WIN32)
    SetConsoleCtrlHandler( (PHANDLER_ROUTINE) CliSignalHandler, TRUE );
  #elif (CLIENT_OS == OS_RISCOS)
    signal( SIGINT, CliSignalHandler );
  #elif (CLIENT_OS == OS_NETWARE)
    signal( SIGABRT, CliSignalHandler ); //abort on floating point [...]printf
    signal( SIGINT, CliSignalHandler );  //       and mathlib.nlm isn't loaded
    signal( SIGTERM, CliSignalHandler );
  #elif (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_DOS)
    signal( SIGINT, CliSignalHandler );
    signal( SIGTERM, CliSignalHandler );
  #elif (CLIENT_OS == OS_IRIX) && defined(__GNUC__)
    signal( SIGHUP, (void(*)(...)) CliSignalHandler );
    signal( SIGQUIT, (void(*)(...)) CliSignalHandler );
    signal( SIGTERM, (void(*)(...)) CliSignalHandler );
    signal( SIGINT, (void(*)(...)) CliSignalHandler );
    signal( SIGSTOP, (void(*)(...)) CliSignalHandler );
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
#elif (CLIENT_OS == OS_SCO) || (CLIENT_OS == OS_OS2) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_DOS) || ((CLIENT_OS == OS_VMS) && !defined(MULTINET))
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
#elif (CLIENT_OS == OS_AMIGAOS)
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

void Client::ParseCommandlineOptions(int Argc, char *Argv[], s32 &inimissing)
{
  for (int i=1;i<Argc;i++)
  {
    if ( strcmp(Argv[i], "-percentoff" ) == 0) // This should be checked here, in case it
    {
      percentprintingoff = 1;                 // follows a -benchmark
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-nofallback" ) == 0 ) // Don't try rc5proxy.distributed.net
    {                                                 // After multiple errors
      nofallback=1;
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-quiet" ) == 0 ) // No messages
    {
      quietmode=1;
      Argv[i][0] = 0;
    }
#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
#if (!defined(WINNTSERVICE))
    else if ( strcmp( Argv[i], "-hide" ) == 0 ) // Hide the client
    {
      quietmode=1;
#if (CLIENT_OS == OS_OS2)
      os2hidden=1;
#else
      win95hidden=1;
#endif
      Argv[i][0] = 0;
    }
#endif
#endif

#if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_OS2)
    else if ( strcmp( Argv[i], "-lurk" ) == 0 ) // Detect modem connections
    {
      lurk=1;
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-lurkonly" ) == 0 ) // Only connect when modem connects
    {
      lurk=2;
      Argv[i][0] = 0;
    }
#endif
    else if ( strcmp( Argv[i], "-noexitfilecheck" ) == 0 ) // Change network timeout
    {
      noexitfilecheck=1;
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-runoffline" ) == 0 ) // Run offline
    {
      offlinemode=1;
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-runbuffers" ) == 0 ) // Run offline & exit when buffer empty
    {
      offlinemode=2;
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-run" ) == 0 ) // Run online
    {
      offlinemode=0;
      Argv[i][0] = 0;
    }
    else if ( strcmp( Argv[i], "-nodisk" ) == 0 ) // No disk buff-*.rc5 files.
    {
      nodiskbuffers=1;
      strcpy(checkpoint_file[0],"none");
      strcpy(checkpoint_file[1],"none");
      Argv[i][0] = 0;
    }
    else if ( strcmp(Argv[i], "-frequent" ) == 0)
    {
      LogScreenf("Setting connections to frequent\n");
      connectoften=1;
      Argv[i][0] = 0;
    }
    else if ((i+1) < Argc) {
      if ( strcmp( Argv[i], "-b" ) == 0 ) // Buffer threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 buffer size to %s\n",Argv[i+1]);
        outthreshold[0] = inthreshold[0]  = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-b2" ) == 0 ) // Buffer threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des buffer size to %s\n",Argv[i+1]);
        outthreshold[1] = inthreshold[1]  = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-bin" ) == 0 ) // Buffer input threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 input buffer size to %s\n",Argv[i+1]);
        inthreshold[0]  = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-bin2" ) == 0 ) // Buffer input threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des input buffer size to %s\n",Argv[i+1]);
        inthreshold[1]  = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-bout" ) == 0 ) // Buffer output threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 output buffer size to %s\n",Argv[i+1]);
        outthreshold[0]  = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-bout2" ) == 0 ) // Buffer output threshold size
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des output buffer size to %s\n",Argv[i+1]);
        outthreshold[1]  = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-u" ) == 0 ) // UUE/HTTP Mode
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting uue/http mode to %s\n",Argv[i+1]);
        uuehttpmode = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-in" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 buffer input file to %s\n",Argv[i+1]);
        strcpy(in_buffer_file[0], Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-in2" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des buffer input file to %s\n",Argv[i+1]);
        strcpy(in_buffer_file[1], Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-out" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting rc5 buffer output file to %s\n",Argv[i+1]);
        strcpy(out_buffer_file[0], Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-out2" ) == 0)
      {                                           // Here in case its with a fetch/flush/update
        LogScreenf("Setting des buffer output file to %s\n",Argv[i+1]);
        strcpy(out_buffer_file[1], Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-a" ) == 0 ) // Override the keyserver name
      {
        LogScreenf("Setting keyserver to %s\n",Argv[i+1]);
        strcpy( keyproxy, Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-p" ) == 0 ) // Override the keyserver port
      {
        LogScreenf("Setting keyserver port to %s\n",Argv[i+1]);
        keyport = (s32) atoi(Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-ha" ) == 0 ) // Override the http proxy name
      {
        LogScreenf("Setting http proxy to %s\n",Argv[i+1]);
        strcpy( httpproxy, Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-hp" ) == 0 ) // Override the http proxy port
      {
        LogScreenf("Setting http proxy port to %s\n",Argv[i+1]);
        httpport = (s32) atoi(Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-l" ) == 0 ) // Override the log file name
      {
        LogScreenf("Setting log file to %s\n",Argv[i+1]);
        strcpy( logname, Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-smtplen" ) == 0 ) // Override the mail message length
      {
        LogScreenf("Setting Mail message length to %s\n",Argv[i+1]);
        messagelen = (s32) atoi(Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-smtpport" ) == 0 ) // Override the smtp port for mailing
      {
        LogScreenf("Setting smtp port to %s\n",Argv[i+1]);
        smtpport = (s32) atoi(Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-smtpsrvr" ) == 0 ) // Override the smtp server name
      {
        LogScreenf("Setting smtp server to %s\n",Argv[i+1]);
        strcpy(smtpsrvr, Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-smtpfrom" ) == 0 ) // Override the smtp source id
      {
        LogScreenf("Setting smtp 'from' address to %s\n",Argv[i+1]);
        strcpy(smtpfrom, Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-smtpdest" ) == 0 ) // Override the smtp destination id
      {
        LogScreenf("Setting smtp 'To' address to %s\n",Argv[i+1]);
        strcpy(smtpdest, Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-nettimeout" ) == 0 ) // Change network timeout
      {
        LogScreenf("Setting network timeout to %s\n",Argv[i+1]);
        nettimeout = (s32) min(300,max(30,atoi(Argv[i+1])));
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-exitfilechecktime" ) == 0 ) // Change network timeout
      {
        exitfilechecktime=max(1,atoi(Argv[i+1]));
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-c" ) == 0)      // set cpu type
      {
        cputype = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-e" ) == 0 ) // Override the email id
      {
        LogScreenf("Setting email for notifications to %s\n",Argv[i+1]);
        strcpy( id, Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-nice" ) == 0 ) // Nice level
      {
        LogScreenf("Setting nice option to %s\n",Argv[i+1]);
        niceness = (s32) atoi( Argv[i+1] );
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-h" ) == 0 ) // Hours to run
      {
        LogScreenf("Setting time limit to %s hours\n",Argv[i+1]);
        minutes = (s32) (60. * atol( Argv[i+1] ));
        strncpy(hours,Argv[i+1],sizeof(hours));
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-n" ) == 0 ) // Blocks to complete in a run
      {
        blockcount = max(1, (s32) atoi( Argv[i+1] ));
        LogScreenf("Setting block completion limit to %d\n",blockcount);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp( Argv[i], "-until" ) == 0 ) // Exit time
      {
        time_t timenow = time( NULL );
        struct tm *gmt = localtime(&timenow );
        minutes = atoi( Argv[i+1] );
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
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
  #if defined(MULTITHREAD)
      else if ( strcmp( Argv[i], "-numcpu" ) == 0 ) // Override the number of cpus
      {
        LogScreenf("Configuring for %s CPUs\n",Argv[i+1]);
        numcpu = (s32) atoi(Argv[i+1]);
        inimissing=0; // Don't complain if the inifile is missing
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
  #endif
      else if ( strcmp(Argv[i], "-ckpoint" ) == 0)
      {
        LogScreenf("Setting rc5 checkpoint file to %s\n",Argv[i+1]);
        strcpy(checkpoint_file[0], Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-ckpoint2" ) == 0)
      {
        LogScreenf("Setting des checkpoint file to %s\n",Argv[i+1]);
        strcpy(checkpoint_file[1], Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-cktime" ) == 0)
      {
        LogScreenf("Setting checkpointing to %s minutes\n",Argv[i+1]);
        checkpoint_min=(s32) atoi(Argv[i+1]);
        checkpoint_min=max(2, checkpoint_min);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-pausefile" ) == 0)
      {
        LogScreenf("Setting pause file to %s\n",Argv[i+1]);
        strcpy(pausefile, Argv[i+1]);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-blsize" ) == 0)
      {
        preferred_blocksize = (s32) atoi(Argv[i+1]);
        if (preferred_blocksize < 28) preferred_blocksize = 28;
        if (preferred_blocksize > 31) preferred_blocksize = 31;
        LogScreenf("Setting preferred blocksize to 2^%d\n",preferred_blocksize);
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
      else if ( strcmp(Argv[i], "-prefer" ) == 0)
      {
        preferred_contest_id = (s32) atoi(Argv[i+1]) - 1;
        if (preferred_contest_id == 0) {
          LogScreen("Setting preferred contest to RC5\n");
        } else {
          LogScreen("Setting preferred contest to DES\n");
          preferred_contest_id = 1;
        }
        Argv[i][0] = Argv[i+1][0] = 0;
        i++; // Don't try and parse the next argument
      }
    }
  }
}

// --------------------------------------------------------------------------

void Client::PrintBanner(char * clname)
{
#if (CLIENT_OS == OS_RISCOS)
  if (guiriscos)
  {
    if (guirestart)
      return;
    else
      clname="";
  }
#endif
  LogScreenf( "\nRC5DES v2.%d.%d client - a project of distributed.net\n"
          "Copyright distributed.net 1997-1998\n"
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
//#if (CLIENT_OS == OS_NETWARE) || (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_OS2)
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
//#else
//{ return -1; }
//#endif
#endif

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_ARM)
#if (CLIENT_OS == OS_RISCOS)
#include <setjmp.h>

static jmp_buf ARMident_jmpbuf;

static void ARMident_catcher(int)
{
  longjmp(ARMident_jmpbuf, 1);
}
#endif

int Client::ARMid()
#if (CLIENT_OS == OS_RISCOS)
{
  u32 detectedvalue; // value ARMident returns, must be interpreted
  int coretouse; // the core the client should use

  LogScreen("Beginning CPU identification...\n");

  // ARMident() will throw SIGILL on an ARM 2 or ARM 250, because
  // they don't have the system control coprocessor. (We ignore the
  // ARM 1 because I'm not aware of any existing C++ compiler that
  // targets it...)

  signal(SIGILL, ARMident_catcher);

  if (setjmp(ARMident_jmpbuf))
  {
    detectedvalue = 0x2000;
  }
  else
  {
    detectedvalue = ARMident();
  }

  signal(SIGILL, SIG_DFL);

  detectedvalue = (detectedvalue >> 4) & 0xfff; // extract part number field

  if ((detectedvalue & 0xf00) == 0)
  {
    // an old-style ID (ARM 3 or prototype ARM 600) - shift it into the new form
    detectedvalue <<= 4;
  }

  if (detectedvalue == 0x300)
  {
    detectedvalue = 3;
  }
  else if (detectedvalue == 0x710)
  {
    // the ARM 7500 returns an ARM 710 ID - need to look at its
    // integral IOMD unit to spot the difference
    u32 detectediomd = IOMDident();
    detectediomd &= 0xff00; // just want most significant byte

    if (detectediomd == 0x5b00)
      detectedvalue = 0x7500;
    else if (detectediomd == 0xaa00)
      detectedvalue = 0x7500FE;
  }

  LogScreen("Completed CPU identification. ");

  switch (detectedvalue)
  {
    case 0x200:
      LogScreen("Detected an ARM 2 or ARM 250\n");
      coretouse=0;
      break;
    case 0x3:
    case 0x600:
    case 0x610:
    case 0x700:
    case 0x710:
    case 0x7500:
    case 0x7500FE:
      LogScreenf("Detected an ARM %X\n", detectedvalue);
      coretouse=0;
      break;
    case 0x810:
      LogScreenf("Detected an ARM %X\n", detectedvalue);
      coretouse=1;
      break;
    case 0xA10:
      LogScreen("Detected a StrongARM 110\n");
      coretouse=1;
      break;
    default:
      LogScreen("Detected an unknown processor\n");
      coretouse=-1;
      break;
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
  if (dot && (strcmpi(dot, ".v27.distributed.net") == 0 ||
      strcmpi(dot, ".distributed.net") == 0))
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

