/*
 * init/exit and other misc stuff that gets called from client/common code.
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Functions in this module:
 *   const char *nwCliGetNLMBaseName(void);
 *   int nwCliInitClient( int argc, char **argv);
 *   int nwCliExitClient( void );
 *   void nwCliMillisecSleep(unsigned long millisecs);
 *
 * $Id: nwcmisc.c,v 1.1.2.1 2001/01/21 15:10:29 cyp Exp $
 *
*/

#ifdef __cplusplus
extern "C"
{
#endif
  #include <string.h>   /* strlen(), strncpy() */
  #include <unistd.h>   /* chdir() */
  #include <nwcntask.h> /* SetCurrentConnection() */
  #include <nwfile.h>   /* SetTargetNameSpace(), SetCurrentNameSpace() */
  #include <nwadv.h>    /* AllocateResourceTag(), RegisterForEvent() */
#ifdef __cplusplus
}
#endif
extern const char *utilGetAppName(void); /* for nwCliGetNLMBaseName() */

#include "nwlemu.h"  /* kernel goodies */
#include "nwcconf.h" /* nwCliLoadSettings() */
#if defined(HAVE_POLLPROC_SUPPORT)
#include "nwcpprun.h" /* nwCli[Uni|I]nitPollingProcedures() */
#endif
#include "nwcmisc.h" /* ourselves */

/* =================================================================== */

static int  __DOWN_handler_control( int mode ); /* forward reference */
static void __DOWN_handler_event(LONG) {__DOWN_handler_control(0);}
static int  __DOWN_handler_control( int mode )
{
  static int seen_exit = -1;
  static int in_handler = 0;
  static int event_handle = -1;
  static int threadGroupID = -1;

  if (mode > 0) /* install */
  {
    int tgid = GetThreadGroupID();
    if (event_handle != -1)
      return 0;
    if (tgid == 0 || tgid == -1)
      return -1;
    event_handle = RegisterForEvent( EVENT_DOWN_SERVER, 
                                     __DOWN_handler_event, NULL );
    if (event_handle == -1)
      return -1;
    in_handler = 0;
    threadGroupID = tgid;
    seen_exit = 0;
  }
  else if (mode < 0) /* uninstall */
  {
    seen_exit = +1;
    if (event_handle != -1 /* && !in_handler */)
    {
      if (UnregisterForEvent( event_handle ) != 0)
        return -1;
      event_handle = -1;
    }
  }
  else /* the event itself */ 
  {
    in_handler = 1;
    if (seen_exit == 0 && threadGroupID != -1)
    {
      unsigned long lasttime = 0;
      unsigned int ticks = ((20*182)/10); /* 20 seconds */
      int tgid = SetThreadGroupID(threadGroupID);
      RaiseExitRequestTrigger();
      ConsolePrintf("%s: waiting for threads to end...\r\n", 
                                       nwCliGetNLMBaseName());
      SetThreadGroupID(tgid);
      while (ticks && seen_exit == 0)
      {
        unsigned long elapsed, currtime = GetCurrentTime();
        if (lasttime == 0) 
          lasttime = currtime;
        elapsed = currtime - lasttime; 
        if (lasttime > currtime) /* useless, but oh well */
          elapsed = 1+currtime-((0xfffffffful)-lasttime);
        lasttime = currtime;
        if (elapsed > ticks)
          elapsed = ticks;
        ticks -= elapsed;
        CYieldWithDelay(); /* sleep allowed! */
      }
    }
    in_handler = 0;
  }
  return 0;
}

/* ==================================================================== */

static unsigned long __nwCliGetPollingProcedureResourceTag(void)
{
  static unsigned long plrpResourceTag = 0xfffffffful;
  if (plrpResourceTag == 0xfffffffful)
  {
    unsigned int thrgrid = GetThreadGroupID();
    if (thrgrid != -1 && thrgrid != 0)
    {
      char *cadd ="AddPollingProcedureRTag";
      char *crem = "RemovePollingProcedure";
      unsigned int nlmHandle = GetNLMHandle();
      void *add = ImportSymbol( nlmHandle, cadd );
      void *rem = ImportSymbol( nlmHandle, crem );

      if (!add || !rem)
        plrpResourceTag = 0;
      else
      {
        char *symname = "AllocateResourceTag";
        unsigned long (*allocrt)(unsigned long,unsigned char *,unsigned long);
        allocrt = (unsigned long (*)(unsigned long,unsigned char *,
              unsigned long))ImportSymbol(nlmHandle, symname );
        if (allocrt)
        {
          unsigned char *rtname = (unsigned char *)("Polling Procedure");
          plrpResourceTag = (*allocrt)(nlmHandle, rtname, 'RPLP' ); 
          UnimportSymbol( nlmHandle, symname );
        }  
      }
      if (add)
        UnimportSymbol( nlmHandle, cadd );
      if (rem)
        UnimportSymbol( nlmHandle, crem );
    }
  }
  if (plrpResourceTag != 0xfffffffful && plrpResourceTag != 0)
    return plrpResourceTag; 
  return 0;
}

/* ==================================================================== */

static const char *__nwCliSetNLMBaseName(const char *argv0)
{
  static int initialized = -1;
  static char basename[32];

  if (initialized < 0)
  {
    int pos = GetThreadGroupID();
    if (pos != -1 && pos != 0)
    {
      char buff[128+1];
      if (GetNLMNameFromNLMID( GetNLMID(), basename, buff ) == 0)
      {
        pos = 0;
        while (basename[pos] && basename[pos]!='.')
        {
          if (basename[pos] >= 'a' && basename[pos] <= 'z')
            basename[pos] -= ('a'-'A');
          pos++;
        }
        basename[pos] = '\0';
        if (pos)
          initialized = 1;
      }
    }
    if (initialized < 0 && argv0)
    {
      pos = strlen(argv0);
      while (pos > 0)
      {
        pos--;
        if (argv0[pos]=='/' || argv0[pos]=='\\' || argv0[pos]==':')
        {
          argv0+=pos+1;
          break;
        }
      }
      pos = 0;
      while (pos < (sizeof(basename)-1))
      {
        basename[pos] = *argv0++;
        if (!basename[pos] || basename[pos]!='.')
          break;
        if (basename[pos] >= 'a' && basename[pos] <= 'z')
          basename[pos] -= ('a'-'A');
        pos++;
      }
      basename[pos] = '\0';
      if (pos)
        initialized = 1;
    }
    if (initialized < 0) /* shouldn't get here */
    {
      initialized = 0; /* recursion protection */
      argv0 = utilGetAppName();
      pos = 0;
      while (pos < (sizeof(basename)-1))
      {
        basename[pos] = *argv0++;
        if (basename[pos] >= 'a' && basename[pos] <= 'z')
          basename[pos] -= ('a'-'A');
        pos++;
      }  
      initialized = 1;
    }
  }
  if (initialized > 0)
    return (const char *)&basename[0];
  return "";
}

const char *nwCliGetNLMBaseName(void)
{
  return __nwCliSetNLMBaseName(NULL);
}

extern "C" unsigned long GetNLMFileHandleFromPrelude(void);

static int __do_path_stuff(int argc, char **argv) 
{                                                 
  /* a) chdir() [if possible] */
  /* b) set namespace [if possible] */
  /* c) set ini basename (always) */

  char *p;
  int i, cwd_changed = 0;
  char inibasename[64];
  int cwd_base_arg = -1;

  inibasename[0] = '\0';
  if (argc >= 1 && argv)
  {
    p = argv[0];
    while (!*p || *p == ' ' || *p == '\t')
      argv = NULL;
  }      
  if (argc < 1 || !argv)
  {
    ConsolePrintf("\r%s: Fatal error: No appname or command line.\r\n", nwCliGetNLMBaseName());
    return -1;
  }
  
  cwd_base_arg = -1;

  p = argv[0];
  i = strlen(p);
  while (i > 0 && p[i-1] != '\\' && p[i-1] != '/' && p[i-1] != ':')
    i--;
  if (i>0) /* argv[0] has a directory component (launched from NetWare partn) */
  {        /* no directory component means got launched from dos partition */
    cwd_base_arg = 0;
  }
  if (strlen(&p[i]) < sizeof(inibasename))
  {
    strcpy(inibasename,&p[i]);
    p = &inibasename[0];
    while (*p && *p != '.')
      p++;
    if (*p == '.')
      strcpy(p, ".ini");
    else
      inibasename[0] = '\0';
  }

  for (i=1; i<argc; i++)
  {
    if (*argv[i] == '-')
    {
      if (strcmp(argv[i]+1,"ini")==0 || strcmp(argv[i]+1,"-ini")==0)
      {
        if ((i+1) >= argc)
        {
          ConsolePrintf("%s: -ini option specified without matching filename.\r\n",
                            nwCliGetNLMBaseName() );
          return -1;
        }
        else
        {
          p = argv[i+1];
          while (*p == ' ' || *p == '\t')
          {
            p++;
          }
          argv[i+1] = p;
          p += strlen(p);
          while (p > argv[i+1])
          {
            p--;
            if (*p != ' ' && *p != '\t')
            {
              p++;
              break;
            }
          }
          *p = '\0';
          while (p > argv[i+1])
          {
            p--;
            if (*p == '\\' || *p == '/' || *p == ':')
            {
              p++;                /* -ini option has a directory component */
              break;
            }
          }
          if (p == argv[i+1]) /* no dir spec */
          {
            if (cwd_base_arg < 0) /* no dir spec in argv[0] either */
            {
              ConsolePrintf("\r%s: -ini option must be a canonical filename\r\n"
                            "       (must include a volume and directory specifier) when\r\n"
                            "       when launching from a non-NetWare partition.\r\n",
                            nwCliGetNLMBaseName() );
              return -1;
            }
          }
          else if (!*p) /* terminated by slash or ':' */
          {
            ConsolePrintf("\r%s: Invalid -ini option (must be a filename)\r\n",
                               nwCliGetNLMBaseName() );
            return -1;
          }
          else if (strlen(p) >= sizeof(inibasename))
          {
            ConsolePrintf("\r%s: Invalid -ini option (name too long)\r\n",
                           nwCliGetNLMBaseName() );
            return -1; 
          }
          else
          {
            strcpy(inibasename,p);
            cwd_base_arg = i+1;
          } 
        } /* if ((i+1) < argc) */
        break; 
      } /* if (strcmp(argv[i]+1,"ini")==0 || strcmp(argv[i]+1,"-ini")==0) */
    } /* if (argv[i] == '-') */
  } /* for (i=1; i<(argc-1); i++) */

  if (cwd_base_arg >= 0)
  {
    p = argv[cwd_base_arg];
    if ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z'))
    {
      if (p[1] == ':') /* DOS partition */
        cwd_base_arg = -1;
    }
  }

  if (cwd_base_arg >= 0)
  {
    int wd_maxlen = 128+strlen(argv[cwd_base_arg]);
    char *wd = (char *)malloc(wd_maxlen);

    if (!wd)
    {
      ConsolePrintf("\r%s: Fatal error: Insufficient memory.\r\n", nwCliGetNLMBaseName());
      return -1;
    }
    i = strlen(strcpy(wd,argv[cwd_base_arg]));
    while (i > 0 && wd[i-1]!='/' && wd[i-1]!='\\' && wd[i-1]!=':')
    {
      i--;
    }
    if (i == 0) /* should not happen */
      cwd_changed = +1; /* use "SYS:/" */
    else
    {
      int oldnamespc, namespc = -1;

      p = &wd[i-1];
      if (*p == ':')
        p++;
      *p = '\0';
      if (chdir(wd) == 0)
        cwd_changed = +1;
      else if (p == &wd[i-1])
      {
        p[0] = '/';
        p[1] = '\0';
        if (chdir(wd) == 0)
          cwd_changed = +1;
        *p = '\0'; 
      }
      if (cwd_changed == 0)
      {  
        oldnamespc = (unsigned char)SetCurrentNameSpace(0);
        if ((oldnamespc & 0xff) == 255)
          oldnamespc = 0; 

        for (namespc = 5; namespc >= 0; namespc--)
        {
          if ( namespc == 0 /* DOS */
            || namespc == 4 /* OS2 aka LONG */
            || namespc == 5 /* NT */ )
          {
            //ConsolePrintf("trying namespace %d\r\n", namespc );
            if (SetCurrentNameSpace( (unsigned char)namespc ) != 0xff &&
                SetTargetNameSpace( (unsigned char)namespc ) != 0xff )      
            {
              //ConsolePrintf("got1 namespace %d\r\n", namespc );
              if ( chdir( wd ) == 0 )
                cwd_changed = +1;
              else if (p == &wd[i-1])
              {
                p[0] = '/';
                p[1] = '\0';
                if (chdir(wd) == 0)
                  cwd_changed = +1;
                *p = '\0'; 
              }
              if (cwd_changed)
              {
                //ConsolePrintf("switched to namespace %d\r\n", namespc);
                break;
              }
            }
          } 
        }
        if (!cwd_changed)
        { 
          SetCurrentNameSpace((unsigned char)oldnamespc);
          SetTargetNameSpace((unsigned char)oldnamespc);

          #if 0
          if (cwd_base_arg > 0) /* used -ini */
          {
            if (wd[1] == ':' && 
              ((*wd >= 'a' && *wd <= 'z') || (*wd >= 'A' && *wd <= 'Z')))
            {                   /* let it pass */
              cwd_changed = +1; /* can't chdir() on a dos partition */
            }
          }
          #endif
        }
      }  /* not found in default namespace */

      if (cwd_changed && cwd_base_arg == 0)
      {
        oldnamespc = namespc;
        if ((oldnamespc & 0xff)== 0xff)
        {
          oldnamespc = (unsigned char)SetCurrentNameSpace(0);
          oldnamespc &= 0xff;
          SetCurrentNameSpace((unsigned char)oldnamespc);
        }
        namespc = FEGetOriginatingNameSpace(FEGetCWVnum(),FEGetCWDnum());
        //ConsolePrintf("\rorig namespace=%d\r\n", namespc);
        namespc &= 0xff;
        if (namespc != 255 && namespc != oldnamespc && 
            namespc != 1 /*not mac*/ && namespc != 3 /*not ftam*/)
        {
          static char newname[256];
          int switched = -1;
          SetCurrentNameSpace((unsigned char)namespc);
          SetTargetNameSpace((unsigned char)namespc);
          if (getcwd( wd, wd_maxlen ))
          {
            p = argv[0]; 
            i = strlen(p);
            while (i > 0 && p[i-1]!='\\' && p[i-1]!='/' && p[i-1]!=':')
              i--;
            p+=i;
            i = strlen(p);
            if ((strlen(wd)+2+i) < sizeof(newname))
            {
              i = strlen(wd);
              strcpy(newname,wd);
              strcat(strcat(newname,"/"),p);
              switched = sopen(newname,(0x0000|0x0200),0x40);/*rdonly|bin,denyno*/
              if (switched != -1)
              {
                close(switched);
                argv[0] = newname;
              }
            }
          }    
          if (switched == -1)
          {
            SetCurrentNameSpace((unsigned char)oldnamespc);
            SetTargetNameSpace((unsigned char)oldnamespc);
          }    
        } /* if ((namespc & 0xff) != 255 && (namespc & 0xff) != oldnamespc) */
      } /* if (cwd_changed && cwd_base_arg == 0) */
    } /* path has dir spec */
//ConsolePrintf("cwd: '%s', argv[0]='%s'\r\n", getcwd( wd, wd_maxlen ),argv[0]);
    free(wd);
  } /* if (cwd_base_arg >= 0) */

  if (inibasename[0] == '\0')
    strcat(strcpy(inibasename, nwCliGetNLMBaseName()),".ini");
  nwCliLoadSettings(inibasename);

  return cwd_changed;
}

#include "version.h"
void __fixup_nlm_ver(const char *apppath)
{
  int fd = sopen(apppath,(0x0002|0x0200),0x40);/*rdwr|bin,denyno*/
  if (fd != -1)
  {
    char *buffer = (char *)malloc(4096);
    if (buffer)
    {
      long len = read(fd,buffer,4096);
      if (len > 0)
      {
        const char *signature="VeRsIoN#";
        unsigned int siglen = strlen(signature);
        long pos, maxpos = (len-(siglen+8));
        for (pos = 0; pos < maxpos; pos++)
        {
          if (((const char)buffer[pos]) == *signature)
          {
            if (memcmp(&buffer[pos],signature,siglen) == 0)
            {
              unsigned long major = (((unsigned long)CLIENT_BUILD));
                                  /*+(((unsigned long)CLIENT_MAJOR_VER)*10000)*/
                                  /*+(((unsigned long)CLIENT_CONTEST)*100)*/
              unsigned long minor = CLIENT_BUILD_FRAC;
              if (major != *((unsigned long *)(&buffer[pos+siglen+0])) &&
                  minor != *((unsigned long *)(&buffer[pos+siglen+4])) )
              {
                memcpy( &buffer[pos+siglen+0], &major, 4);
                memcpy( &buffer[pos+siglen+4], &minor, 4);
                if (lseek( fd, 0, 0 ) == 0)
                {
                  write( fd, buffer, pos+siglen+8 );
                }
              }
              break;
            }
          }
        } /* for (;;) */
      } /* if (read) */
      free((void *)buffer);
    } /* if (buffer) */
    close(fd);
  } /* if (fd != -1) */
  return;
}

/* ==================================================================== */

int nwCliInitClient( int argc, char **argv)
{
  int i;
  ConsolePrintf("\r");     /* why can't *someone* fix console? */

  i = GetThreadGroupID(); /* better safe than sorry */
  if (i == 0 || i == -1) /* should not happen in this function */
  {
    ConsolePrintf("\r?????: Unable to obtain thread group ID\r\n");
    return -1;
  }

  ThreadSwitchLowPriority();
  SetCurrentConnection(0); /* this is said to be neccesary on NetWare 4.x */
  nwCliGetNLMBaseName(); /* self-initializing */

  if (__do_path_stuff(argc, argv) < 0)
    return -1;            /* error message will have been printed */

  __fixup_nlm_ver(argv[0]);

  __DOWN_handler_control( +1 /* initialize */ );
  __nwCliGetPollingProcedureResourceTag(); /* self-initializing */
  #if defined(HAVE_POLLPROC_SUPPORT)
  nwCliInitPollingProcedures();
  #endif

  #if 0
  if (GetSetableParameterValue( 0, (unsigned char *)
                  "SMP NetWare Kernel Mode", (void *)&pathbuffer[0] ) != 0)
      strcpy(pathbuffer,"*unknown*");
  ConsolePrintf("SMP NetWare Kernel Mode = \"%s\"\r\n", pathbuffer );
  #endif

  return 0;
}

int nwCliExitClient( void )
{
  __DOWN_handler_control( -1 /* deinitialize */ );
  #if defined(HAVE_POLLPROC_SUPPORT)
  nwCliUninitPollingProcedures(); /* this must be clib-call free */
  #endif
  return 0;
}  

/* ==================================================================== */

static void __doNothing(void){}
static void __reset_util_counters(void)
{
  if (GetFileServerMajorVersionNumber()<5)
  {
    static unsigned int (*_MaximumNumberOfPollingLoops) = ((unsigned int (*))0x01);
    static unsigned int (*_NumberOfPollingLoops) = (unsigned int (*))0x00;
    static unsigned int (*_CPU_Utilization) = (unsigned int (*))0x00;
    static unsigned int (*_CPU_Combined) = (unsigned int (*))0x0;
    int cpu_util = -1;
    
    /* 
       we can do this for NW411 SMP.NLM too, since the _CPU_Utilization
       stuff is almost purely informational: the only place its actually
       involved in a decision making process is in thread_switch(),
       and we have to squelch that anyway otherwise the crunchers
       bounce around way too much for mp to be good for anything.
    */
    if (_MaximumNumberOfPollingLoops == ((unsigned int (*))0x01) )
    {
      int nlmHandle = GetNLMHandle();
      _CPU_Combined = 
         (unsigned int *)ImportSymbol(nlmHandle,"CPU_Combined");
      _CPU_Utilization = 
         (unsigned int *)ImportSymbol(nlmHandle, "CPU_Utilization");
      _NumberOfPollingLoops =
         (unsigned int *)ImportSymbol(nlmHandle, "NumberOfPollingLoops" );
      _MaximumNumberOfPollingLoops =
         (unsigned int *)ImportSymbol(nlmHandle, "MaximumNumberOfPollingLoops" );
      //ConsolePrintf("_CPU_Combined=%08x, _CPU_Utilization=%08x\r\n",_CPU_Combined,_CPU_Utilization);
    }
    if (_CPU_Utilization && _CPU_Combined)
    {
      int curr, count = GetNumberOfRegisteredProcessors();
      unsigned int (*utilptr) = _CPU_Utilization;
      for (curr = 0; curr < count; curr++)
        *utilptr++ = 0;
      *_CPU_Combined = 0;
      cpu_util = 100; /* force fall through */
    }
    if (cpu_util == -1)
    {
      cpu_util = 100;
      if (_MaximumNumberOfPollingLoops && _NumberOfPollingLoops)
      {
        unsigned long m = *_MaximumNumberOfPollingLoops;
        unsigned long n = *_NumberOfPollingLoops;
        if (n > ((0xFFFFFFFFul)/150))
        {
          n>>=8;
          m>>=8;
        }
        cpu_util = (((m == 0) || (n > m)) ? (0) : (100-(((n*100)+(m>>1))/m)));
      }
    }
    if (cpu_util >= 75)
    {
      if (_MaximumNumberOfPollingLoops && _NumberOfPollingLoops)
      {
        *_NumberOfPollingLoops = *_MaximumNumberOfPollingLoops;
      }
      else
      {
        unsigned long plrpResourceTag;
    //  ConsolePrintf("adddel %u\r\n", GetCurrentTime());
        if ((plrpResourceTag = __nwCliGetPollingProcedureResourceTag()) != 0)
        {
          if (AddPollingProcedureRTag( __doNothing, plrpResourceTag) == 0)
          {
            RemovePollingProcedure( __doNothing );
          }
        }
      }
    }
  }
  return;
}  

void nwCliMillisecSleep(unsigned long millisecs)
{
  if (!nwCliGetUtilizationSupressionFlag())
  {
    delay(millisecs);
  }
  else
  {
    //ConsolePrintf("%u: getcpuutil: %d ms=%u\r\n", GetCurrentTime(),
    //                          GetProcessorUtilization(), millisecs);
    do
    {
      unsigned long ticks2, ticks = GetCurrentTicks();
      unsigned long elapsed = 54;
      unsigned int sleeptime;
      __reset_util_counters();
      sleeptime = 4; /* don't sleep more than 220ms in one loop */
      if (millisecs < 220UL) 
        sleeptime = (unsigned int)(millisecs / 55UL);
      delay( sleeptime * 55 ); /* delay(0) if less than 55ms */
      __reset_util_counters();
      ticks2 = GetCurrentTicks();
      if (ticks2 > ticks)
        elapsed = (ticks2 - ticks)*55;
      if (elapsed > millisecs)
        elapsed = millisecs;
      millisecs -= elapsed;
    } while (millisecs);
  }
  return;
}

/* ------------------------------------------------------------------- */

