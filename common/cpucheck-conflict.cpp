/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/
const char *cpucheck_cpp(void) {
return "@(#)$Id: cpucheck-conflict.cpp,v 1.79.2.5 1999/06/16 12:22:14 ivo Exp $"; }

/* ------------------------------------------------------------------------ */
/*
   Implementing long __GetRawProcessorID( const char **cpuname ):
   
   if identification failed:           return ID==-1L, and set cpuname
                                       to NULL 
   if identification is not supported: return ID==-2L, and set cpuname
                                       to NULL 
   if we have a name, but no ID:       return ID==0, set cpuname to the 
                                       raw name (eg "PCA56" )
   if we have an ID and a name:        return ID and fully formatted 
                                       name (eg "Alpha EV5.6 (21164PC)")
   if we have an ID but no name:       return ID, set cpuname to ""
                                                   -  cyp April/03/1999
*/

#include "cputypes.h"
#include "baseincs.h"  // for platform specific header files
#include "cpucheck.h"  //just to keep the prototypes in sync.
#include "logstuff.h"  //LogScreen()/LogScreenRaw()

#if (CLIENT_OS == OS_SOLARIS)
#  include <unistd.h>    // cramer - sysconf()
#elif (CLIENT_OS == OS_IRIX)
#  include <sys/prctl.h>
#elif (CLIENT_OS == OS_DEC_UNIX)
#  include <unistd.h>
#  include <sys/sysinfo.h>
#  include <machine/hal_sysinfo.h>
#  include <machine/cpuconf.h>
#elif (CLIENT_OS == OS_AIX)
#  include <sys/systemcfg.h>
/* if compiled on older versions of 4.x ... */
#  ifndef POWER_620
#    define POWER_620 0x0040
#  endif
#  ifndef POWER_630
#    define POWER_630 0x0080
#  endif
#  ifndef POWER_A35
#    define POWER_A35 0x0100
#  endif
#  ifndef POWER_RS64II
#    define POWER_RS64II    0x0200          /* RS64-II class CPU */
#  endif
#elif (CLIENT_OS == OS_FREEBSD)
#  include <sys/sysctl.h>
#endif

/* ------------------------------------------------------------------------ */

unsigned int GetNumberOfSupportedProcessors( void )
{
#if (CLIENT_OS == OS_RISCOS)
  return ( 2 ); /* not just some arbitrary number */
#elif (CLIENT_OS == OS_MACOS)
  if (haveMP)
    return( MAC_MAXCPUS );
  return(1);
#else
  return ( 128 ); /* just some arbitrary number */
#endif
}

/* ---------------------------------------------------------------------- */

int GetNumberOfDetectedProcessors( void )  //returns -1 if not supported
{
  static int cpucount = -2;

  if (cpucount == -2)
  {
    cpucount = -1;
    #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_NETBSD) || \
        (CLIENT_OS == OS_BSDOS) || (CLIENT_OS == OS_OPENBSD)
    { /* comment out if inappropriate for your *bsd - cyp (25/may/1999) */
      int ncpus; size_t len = sizeof(ncpus);
      //int mib[2]; mib[0] = CTL_HW; mib[1] = HW_NCPU;
      //if (sysctl( &mib[0], 2, &ncpus, &len, NULL, 0 ) == 0)
      if (sysctlbyname("hw.ncpu", &ncpus, &len, NULL, 0 ) == 0)
        cpucount = ncpus;
    }
    #elif (CLIENT_OS == OS_BEOS)
    {
      system_info the_info;
      get_system_info(&the_info);
      cpucount = the_info.cpu_count;
    }
    #elif (CLIENT_OS == OS_WIN32)
    {
      SYSTEM_INFO systeminfo;
      GetSystemInfo(&systeminfo);
      cpucount = systeminfo.dwNumberOfProcessors;
      if (cpucount < 1)
        cpucount = -1;
    }
    #elif (CLIENT_OS == OS_NETWARE)
    {
      cpucount = nwCliGetNumberOfProcessors();
    }
    #elif (CLIENT_OS == OS_OS2)
    {
      int rc = (int) DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                  &cpucount, sizeof(cpucount));
      if (rc != 0 || cpucount < 1)
        cpucount = -1;
    }
    #elif (CLIENT_OS == OS_LINUX)
    {
      FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
      cpucount = 0;
      if ( cpuinfo )
      {
        char buffer[256];
        while(fgets(buffer, sizeof(buffer), cpuinfo))
        {
          buffer[sizeof(buffer) - 1] = '\0';
          #if (CLIENT_CPU == CPU_X86 || CLIENT_CPU == CPU_POWERPC)
          if (strstr(buffer, "processor") == buffer)
            cpucount++;
          #elif (CLIENT_CPU == CPU_SPARC)
          // 2.1.x kernels (at least 2.1.125)
          if (strstr( buffer, "ncpus active\t: " ) == buffer)
            cpucount = atoi( buffer+15 );
          // 2.0.x non-smp kernels (at least 2.0.35)
          else if (strstr( buffer, "BogoMips\t: " ) == buffer)
            cpucount = 1;
          // 2.0.x smp kernels (at least 2.0.35)
          else if ( buffer == strstr( buffer, 
                    "        CPU0\t\tCPU1\t\tCPU2\t\tCPU3\n" ) )
          {
            fgets( buffer, 256, cpuinfo );
            for (char *p = strtok( buffer+7, "\t \n" ); p;
                    p = strtok( NULL, "\t \n" ))
            {
              if (strstr( p, "online" ) || strstr( p, "akp"))
                cpucount++;
            }
          }
          #elif (CLIENT_CPU == CPU_ALPHA)
          cpucount = 1; /* assume this until we know otherwise */
          /* SMP data (2.1+) - "CPUs probed %d active %d map 0x%x AKP %d\n" */
          if (memcmp(buffer, "CPUs probed", 11) == 0 && 
                    (buffer[11] == '\t' || buffer[11] == ' '))
          {
            char *p = strstr( buffer, "active" );
            if (p && (p[6] == '\t' || p[6] == ' '))
            {
              p += 6; while (*p && !isdigit(*p)) p++;
              cpucount = atoi(p);
              break;
            }
          }
          #else
          cpucount = -1;
          break;
          #endif
        }
        fclose(cpuinfo);
      }
    }
    #elif (CLIENT_OS == OS_IRIX)
    {
      cpucount = (int)prctl( PR_MAXPPROCS, 0, 0);
    }
    #elif (CLIENT_OS == OS_SOLARIS)
    {
      cpucount = sysconf(_SC_NPROCESSORS_ONLN);
    }
    #elif (CLIENT_OS == OS_AIX)
    {
      cpucount = sysconf(_SC_NPROCESSORS_ONLN);
//    cpucount = _system_configuration.ncpus; should work the same way but might go
    }
    #elif (CLIENT_OS == OS_RISCOS)
    {
      cpucount = riscos_count_cpus();
    }
    #elif (CLIENT_OS == OS_QNX)
    {
      cpucount = 1;
    }
    #elif (CLIENT_OS == OS_MACOS)
    {
      cpucount = 1;
      if (haveMP)
        cpucount = MPProcessors();
    }
    #elif ( (CLIENT_OS == OS_DEC_UNIX) && defined(OS_SUPPORTS_SMP))
    {
      // We really only want to do this for multithreaded clients.
      // Earlier versions of the Digital Unix don't support this.
      int status = 0;
      struct cpu_info buf;
      int st = 0;
      status = getsysinfo(GSI_CPU_INFO, (char *) &buf, sizeof(buf), st, NULL,NULL);
      if (status == -1)
        cpucount = -1;
      else
        cpucount = buf.cpus_in_box;
    }
    #endif
    if (cpucount < 1)  //not supported
      cpucount = -1;
  }

  return cpucount;
}

/* ---------------------------------------------------------------------- */

unsigned int ValidateProcessorCount( int numcpu, int quietly )
{
  static int detected_count = -2;

  //-------------------------------------------------------------
  // Validate processor/thread count.
  // --------------------
  // numcpu with zero implies "force-single-threaded"
  //
  // Whether a CLIENT_CPU (core) is thread safe or not, or whether the client
  // was built with thread support or not, is not the responsibility of
  // select cpu logic.
  //--------------------

  if (detected_count == -2)  // returns -1 if no hardware detection
    detected_count = GetNumberOfDetectedProcessors(); 

  if (numcpu < 0)                //numcpu == 0 implies force non-mt;
  {                              //numcpu < 0  implies autodetect
    if ( detected_count < 1 )
    {
      if (!quietly)
        LogScreen( CLIENT_OS_NAME " does not support SMP or\n"
                  "does not support processor count detection.\n"
                  /* "Automatic processor count detection failed.\n" */
                  "A single processor machine is assumed.\n");
      detected_count = 1;
    }
    else if (!quietly)
    {
      LogScreen("Automatic processor detection found %d processor%s.\n",
                detected_count, ((detected_count==1)?(""):("s")) );
    }
    numcpu = detected_count;
    if (numcpu < 0) //zero is legal (implies force-single-threaded)
      numcpu = 0;
  }

  if (((unsigned int)(numcpu)) > GetNumberOfSupportedProcessors())
    numcpu = (int)GetNumberOfSupportedProcessors();

  return (unsigned int)numcpu;
}

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_68K)
static long __GetRawProcessorID(const char **cpuname)
{
  static long detectedtype = -2; /* -1==failed, -2==not supported */
  static const char *detectedname = NULL;
  static char namebuf[20];
  static struct { const char *name; long rid; } cpuridtable[] = {
                { "68000",           68000L  },
                { "68010",           68010L  },
                { "68020",           68020L  },
                { "68030",           68030L  },
                { "68040",           68040L  },
                { "68060",           68060L  }
                };

  #if (CLIENT_OS == OS_AMIGAOS)
  if (detectedtype == -2)
  {
    long flags = (long)(SysBase->AttnFlags);

    if ((flags & AFF_68060) && (flags & AFF_68040) && (flags & AFF_68030))
      detectedtype = 68060L; // Phase5 060 boards at least report this...
    else if ((flags & AFF_68040) && (flags & AFF_68030))
      detectedtype = 68040L; // 68040
    else if ((flags & AFF_68030) && (flags & AFF_68020))
      detectedtype = 68030L; // 68030
    else if ((flags & AFF_68020) && (flags & AFF_68010))
      detectedtype = 68020L; // 68020
    else if (flags & AFF_68010)
      detectedtype = 68010L; // 68010
    else
      detectedtype = -1;
  }
  #elif (CLIENT_OS == OS_MACOS)
  if (detectedtype == -2)
  {
    long result;
    // Note: gestaltProcessorType is used so that if the 68K client is run on
    // on PPC machine for test purposes, it will get the type of the 68K
    // emulator. Also, gestaltNativeCPUType is not present in early versions of
    // System 7. (For more info, see Gestalt.h)
    detectedtype = -1L;
    if (Gestalt(gestaltProcessorType, &result) == noErr)
    {
      switch(result)
      {
        case gestalt68000: detectedtype = 68000L;  break;
        case gestalt68010: detectedtype = 68010L;  break;
        case gestalt68020: detectedtype = 68020L;  break;
        case gestalt68030: detectedtype = 68030L;  break;
        case gestalt68040: detectedtype = 68040L;  break;
        default:           detectedtype = -1L;     break;
      }
    }
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "CPU:\t\t";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
          {
            unsigned int l = strlen( cpuridtable[n].name );
            if ((!p[l] || isspace(p[l])) && memcmp(p,cpuridtable[n].name,l)==0)
            {
              detectedtype = cpuridtable[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ )
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif

  if (detectedtype > 0 && detectedname == NULL )
  {
    unsigned int n;
    detectedname = "";
    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (((long)(cpuridtable[n].rid)) == detectedtype )
      {
        strcpy( namebuf, "Motorola " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }

  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}

#endif /* (CLIENT_CPU == CPU_68K) */

/* ---------------------------------------------------------------------- */

#if (CLIENT_OS == OS_AIX)

#define ARCH_IS_POWER 0x20

static long __GetRawProcessorID( const char **cpuname )
{
  static long detectedtype = -2; /* -1==failed, -2==not supported */
  static const char *detectedname = NULL;

  if ( detectedtype == -2 ) {
    long arch_id;
// we treat the PPC as the default platform
    if ( _system_configuration.architecture == POWER_RS ) {
      arch_id=ARCH_IS_POWER;
    } else {
      arch_id=0;
    }

    switch (_system_configuration.implementation) {
    case POWER_601:
      detectedname="PowerPC 601";
      detectedtype= 0x01 | arch_id;
      break;
    case POWER_603:
      detectedname="PowerPC 603";
      detectedtype= 0x02 | arch_id;
      break;
    case POWER_604:
      detectedname="PowerPC 604";
      detectedtype= 0x03 | arch_id;
      break;
    case POWER_620:
      detectedname="PowerPC 620";
      detectedtype= (0x04 | arch_id);
      break;
    case POWER_630:
      detectedname="PowerPC 630";
      detectedtype= (0x05 | arch_id);
      break;
    case POWER_A35:
      detectedname="PowerPC A35"; // this should be an AS/400 !!!! (65-bit)
      detectedtype= (0x05 | arch_id);
      break;
    case POWER_RS64II:
      detectedname="PowerPC RS64II"; // nameing not correct but how
      detectedtype= (0x06 | arch_id);
      break;
    case POWER_RS1:
      detectedname="POWER RS";
      detectedtype= (0x10 | arch_id);
      break;
    case POWER_RSC:
      detectedname="POWER RS2 Superchip"; // nameing ??
      detectedtype= (0x11 | arch_id);
      break;
    case POWER_RS2:
      detectedname="POWER RS2";
      detectedtype= (0x12 | arch_id);
      break;
    default:
      detectedname=NULL;
      detectedtype= -1;
      break;
    }
  #ifndef _AIXALL
  #if (CLIENT_CPU == CPU_POWER)
    if ( arch_id != ARCH_IS_POWER )
        LogScreen("The CPU detected is not supported by this client.\n"
                "Please use this client only on POWER systems.\n");
  #else
    if ( arch_id == ARCH_IS_POWER )
        LogScreen("The CPU detected is not supported by this client.\n"
                "Please use this client only on PowerPC systems.\n");
  #endif
  #endif
  }
  *cpuname=detectedname;
  return(detectedtype);

}

#endif /* (CLIENT_OS == OS_AIX) */

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_POWERPC) && !defined(_AIXALL)
static long __GetRawProcessorID(const char **cpuname)
{
  /* ******* detected type reference is (PVR value >> 16) *********** */
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static char namebuf[30];
  static struct { int rid; const char *name; } cpuridtable[] = {
                {       1, "601"             },
                {       3, "603"             },
                {       4, "604"             },
                {       6, "603e"            },
                {       7, "603ev"           },
                {       8, "740/750/G3"      },
                {       9, "604e"            },
                {      10, "604ev"		     }
                };

  #if (CLIENT_OS == OS_MACOS)
  if (detectedtype == -2L)
  {
    // Note: need to use gestaltNativeCPUtype in order to get the correct
    // value for G3 upgrade cards in a 601 machine.
    long result;
    detectedtype = -1;
    if (Mac_PPC_prototype) 
      detectedtype = 1L;  // old developer machines - 601
    else if (Gestalt(gestaltNativeCPUtype, &result) == noErr)
      detectedtype = result - 0x100L;
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static struct 
           { const char *sig;  int rid; } sigs[] = {
           { "601",                  1  },
           { "603",                  3  },
           { "604",                  4  },
           { "603e",                 6  },
           { "603ev",                7  },
           { "750",         		 8  },
           { "604e",                 9  },
           { "604ev",      			10  }
           };
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(sigs)/sizeof(sigs[0])); n++ )
          {
            unsigned int l = strlen( sigs[n].sig );
            if ((!p[l] || isspace(p[l])) && memcmp( p, sigs[n].sig, l)==0)
            {
              detectedtype = (long)sigs[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\n' && *p!='\r' && n<(sizeof(namebuf)-1); n++ )
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif
  
  if (detectedtype > 0 && detectedname == NULL )
  {
    unsigned int n;
    detectedname = "";
    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (((long)(cpuridtable[n].rid)) == detectedtype )
      {
        strcpy( namebuf, "PowerPC " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif /* (CLIENT_CPU == CPU_POWERPC) */

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_X86)

#if (defined(__WATCOMC__) || (CLIENT_OS == OS_QNX)) 
  #define x86ident _x86ident
#endif
#if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
  extern "C" u32 x86ident( void ) asm ("x86ident");
#else
  extern "C" u32 x86ident( void );
#endif

long __GetRawProcessorID(const char **cpuname, int whattoret = 0 )
{
  static long detectedtype = -2L;
  static const char *detectedname = NULL;
  static int  kKeysPerMhz = 512; /* default rate if not found */
  static int  coretouse   = 0;   /* default core if not found */
  
  if ( detectedtype == -2L )
  {
    static char namebuf[30];
    const char *vendorname = NULL;
    struct cpuxref { int cpuid, kKeysPerMhz, coretouse; 
                     const char *cpuname; } *internalxref = NULL;
    u32 dettype     = x86ident();
    int cpuidbmask  = 0xfff0; /* mask with this to find it in the table */
    int cpuid       = (((int)dettype) & 0xffff);
    int vendorid    = (((int)(dettype >> 16)) & 0xffff);
  
    sprintf( namebuf, "%04X:%04X", vendorid, cpuid );
    detectedname = (const char *)&namebuf[0];
    detectedtype = 0; /* assume not found */
  
    if ( vendorid == 0x7943 ) // Cyrix CPU
    {
      static struct cpuxref cyrixxref[]={
          {    0x40, 0512,     0, "486"       }, // use Pentium core
          {  0x0440, 0512,     0, "MediaGX"   },
          {  0x0490, 1185,     0, "5x86"      },
          {  0x0520, 2090,     3, "6x86"      }, // "Cyrix 6x86/6x86MX/M2"
          {  0x0540, 1200,     0, "GXm"       }, // use Pentium core here too
          {  0x0600, 2115, 0x103, "6x86MX"    },
          {  0x0000, 2115,     3, NULL        } //default core == 6x86
          }; internalxref = &cyrixxref[0];
      vendorname = "Cyrix ";
      #if defined(SMC)            //self modifying core
      cyrixxref[0].coretouse = 1; // /bugs/ #99  pentium -> 486smc
      #endif                      // des is unaffected. both 0/1 use p5 core
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x6543 ) //centaur/IDT cpu
    {
      static struct cpuxref centaurxref[]={
          {  0x0540, 1200,0x100, "C6"          }, // use Pentium core
          {  0x0585, 1346,0x102, "WinChip 2"   }, // pentium Pro (I think)
          {  0x0000, 1346,    0, NULL          }  // default core == Pentium
          }; internalxref = &centaurxref[0];
      vendorname = "Centaur/IDT ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x7541 ) // AMD CPU
    {
      static struct cpuxref amdxref[]={
          {  0x0040, 0512,     0, "486"        },   // "Pentium, Pentium MMX, Cyrix 486/5x86/MediaGX, AMD 486",
          {  0x0430, 0512,     0, "486DX2"     },
          {  0x0470, 0512,     0, "486DX2WB" },
          {  0x0480, 0512,     0, "486DX4"     },
          {  0x0490, 0512,     0, "486DX4WB" },
          {  0x04E0, 1185,     0, "5x86"       },
          {  0x04F0, 1185,     0, "5x86WB"     },
          {  0x0500, 2353,     4, "K5 PR75, PR90, or PR100" }, // use K5 core
          {  0x0510, 2353,     4, "K5 PR120 or PR133" },
          {  0x0520, 2353,     4, "K5 PR166" },
          {  0x0530, 2353,     4, "K5 PR200" },
          {  0x0560, 1611, 0x105, "K6"         },
          {  0x0570, 1611, 0x105, "K6"         },
          {  0x0580, 1690, 0x105, "K6-2"       },
          {  0x0590, 1690, 0x105, "K6-3"       },
          {  0x0000, 1690,     5, NULL         }   // for the future - default core = K6
          }; internalxref = &amdxref[0];
      vendorname = "AMD ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x6E49 || vendorid == 0x6547 ) // Intel CPU
    {
      static struct cpuxref intelxref[]={
          {  0x0030, 0426,     1, "80386"      },   // generic 386/486 core
          {  0x0040, 0512,     1, "80486"      },
          {  0x0400, 0512,     1, "486DX 25 or 33" },
          {  0x0410, 0512,     1, "486DX 50" },
          {  0x0420, 0512,     1, "486SX" },
          {  0x0430, 0512,     1, "486DX2" },
          {  0x0440, 0512,     1, "486SL" },
          {  0x0450, 0512,     1, "486SX2" },
          {  0x0470, 0512,     1, "486DX2WB" },
          {  0x0480, 0512,     1, "486DX4" },
          {  0x0490, 0512,     1, "486DX4WB" },
          {  0x0500, 1416,     0, "Pentium" }, //stepping A
          {  0x0510, 1416,     0, "Pentium" },
          {  0x0520, 1416,     0, "Pentium" },
          {  0x0530, 1416,     0, "Pentium Overdrive" },
          {  0x0540, 1432, 0x106, "Pentium MMX" },
          {  0x0570, 1416,     0, "Pentium" },
          {  0x0580, 1432, 0x106, "Pentium MMX" },
          {  0x0600, 2785,     2, "Pentium Pro" },
          {  0x0610, 2785,     2, "Pentium Pro" },
          {  0x0630, 3092, 0x102, "Pentium II" },
          {  0x0650, 3092, 0x102, "Pentium II" },
          {  0x0660, 3092, 0x102, "Celeron-A" },
          {  0x0670, 4096, 0x102, "Pentium III" },
          {  0x0000, 4096,     2, NULL           }  // default core = PPro/PII
          }; internalxref = &intelxref[0];
      vendorname = "Intel "; 
      if ((cpuid == 0x30) || (cpuid == 0x40))
        vendorname = ""; //not for generic 386/486
      cpuidbmask = 0x0ff0; //strip type AND stepping bits.
    }
  
    if (internalxref != NULL) /* we know about this vendor */
    {
      unsigned int pos;
      int maskedid = ( cpuid & cpuidbmask );
  
      for (pos = 0; ; pos++ )
      {
        if ((internalxref[pos].cpuname == NULL) || /* bonk! hit bottom */
           (maskedid == (internalxref[pos].cpuid & cpuidbmask))) /* found it */
        {
          kKeysPerMhz  = internalxref[pos].kKeysPerMhz;
          coretouse    = internalxref[pos].coretouse;
          detectedtype = dettype;
          if ( internalxref[pos].cpuname )
          {
            strcpy( namebuf, vendorname );
            strcat( namebuf, internalxref[pos].cpuname );
            detectedname = (const char *)&namebuf[0];
          }
          break;
        }
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  if (whattoret == 'c')
    return ((long)coretouse);
  if (whattoret == 'k')
    return ((long)kKeysPerMhz);
  return detectedtype;
}
#endif /* X86 */

/* ---------------------------------------------------------------------- */

#if (CLIENT_OS == OS_RISCOS)
#include <setjmp.h>
static jmp_buf ARMident_jmpbuf;
static void ARMident_catcher(int)
{
  longjmp(ARMident_jmpbuf, 1);
}
#endif

#if (CLIENT_CPU == CPU_ARM)
static long __GetRawProcessorID(const char **cpuname )
{
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;

  #if (CLIENT_OS == OS_RISCOS)
  if ( detectedvalue == -2L )
  {
    static char namebuf[60];
    /*
       ARMident() will throw SIGILL on an ARM 2 or ARM 250, because they 
       don't have the system control coprocessor. (We ignore the ARM 1 
       because I'm not aware of any existing C++ compiler that targets it...)
     */
    signal(SIGILL, ARMident_catcher);
    if (setjmp(ARMident_jmpbuf))
      detectedvalue = 0x2000;
    else
      detectedvalue = ARMident();
    signal(SIGILL, SIG_DFL);
    detectedvalue = (detectedvalue >> 4) & 0xfff; // extract part number field

    if ((detectedvalue & 0xf00) == 0) //old-style ID (ARM 3 or prototype ARM 600)
      detectedvalue <<= 4;            // - shift it into the new form
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
        detectedvalue = 0x7500FEL;
    }
    detectedname = ((const char *)&(namebuf[0]));
    switch (detectedvalue)
    {
      case 0x200: strcpy( namebuf, "ARM 2 or 250" ); 
                  break;
      case 0xA10: strcpy( namebuf, "StrongARM 110" ); 
                  break;
      case 0x3:
      case 0x600:
      case 0x610:
      case 0x700:
      case 0x7500:
      case 0x7500FEL:
      case 0x710:
      case 0x810: sprintf( namebuf, "ARM %lX", detectedvalue );
                  break;
      default:    sprintf( namebuf, "%lX", detectedvalue );
                  detectedvalue = 0;
                  break;
    }
  }
  #endif /* RISCOS */

  if ( cpuname )
    *cpuname = detectedname;
  return detectedvalue;
}  
#endif

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_MIPS)
static long __GetRawProcessorID(const char **cpuname)
{
  static int detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static char namebuf[30];
  static struct { const char *name; int rid; } cpuridtable[] = {
                { "R2000"         ,       1  },
                { "R3000"         ,       2  },
                { "R3000A"        ,       3  },
                { "R3041"         ,       4  },
                { "R3051"         ,       5  },
                { "R3052"         ,       6  },
                { "R3081"         ,       7  },
                { "R3081E"        ,       8  },
                { "R4000PC"       ,       9  },
                { "R4000SC"       ,      10  },
                { "R4000MC"       ,      11  },
                { "R4200"         ,      12  },
                { "R4400PC"       ,      13  },
                { "R4400SC"       ,      14  },
                { "R4400MC"       ,      15  },
                { "R4600"         ,      16  },
                { "R6000"         ,      17  },
                { "R6000A"        ,      18  },
                { "R8000"         ,      19  },
                { "R10000"        ,      20  }
                };
  
  #if (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu model\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
          {
            unsigned int l = strlen( cpuridtable[n].sig );
            if ((!p[l] || isspace(p[l])) && memcmp( p, cpuridtable[n].sig, l)==0)
            {
              detectedtype = (long)cpuridtable[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ ) 
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif

  if (detectedtype > 0 && detectedname == NULL)
  {
    unsigned int n;
    detectedname = "";
    for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (detectedtype == cpuridtable[n].rid )
      {
        strcpy( namebuf, "MIPS " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_SPARC)
static long __GetRawProcessorID(const char **cpuname)
{
  /* detectedtype reference is (0x100 + ((get_psr() >> 24) & 0xff)) */
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  
  /* from linux/arch/sparc/kernel/cpu.c */
  static struct { int psr_impl, psr_vers; const char *name; } cpuridtable[] = {
  { 0, 0, "Fujitsu  MB86900/1A or LSI L64831 SparcKIT-40"}, /* Sun4/100, 4/200, SLC */
  { 0, 4, "Fujitsu  MB86904"}, /* born STP1012PGA */
  { 1, 0, "LSI Logic Corporation - L64811"}, /* SparcStation2, SparcServer 490 & 690 */
  { 1, 1, "Cypress/ROSS CY7C601"}, /* SparcStation2 */
  { 1, 3, "Cypress/ROSS CY7C611"}, /* Embedded controller */
  { 1, 0xf, "ROSS HyperSparc RT620"}, /* Ross Technologies HyperSparc */
  { 1, 0xe, "ROSS HyperSparc RT625"},
  /* ECL Implementation, CRAY S-MP Supercomputer... AIEEE! */
  /* Someone please write the code to support this beast! ;) */
  { 2, 0, "Bipolar Integrated Technology - B5010"}, /* Cray S-MP */
  { 3, 0, "LSI Logic Corporation - unknown-type"},
  { 4, 0, "Texas Instruments, Inc. - SuperSparc 50"},
  { 4, 1, "Texas Instruments, Inc. - MicroSparc"}, /* SparcClassic STP1010TAB-50*/
  { 4, 2, "Texas Instruments, Inc. - MicroSparc II"},
  { 4, 3, "Texas Instruments, Inc. - SuperSparc 51"},
  { 4, 4, "Texas Instruments, Inc. - SuperSparc 61"},
  { 4, 5, "Texas Instruments, Inc. - unknown"},
  { 5, 0, "Matsushita - MN10501"},
  { 6, 0, "Philips Corporation - unknown"},
  { 7, 0, "Harvest VLSI Design Center, Inc. - unknown"},
  { 8, 0, "Systems and Processes Engineering Corporation (SPEC)"}, /* Gallium arsenide 200MHz, BOOOOGOOOOMIPS!!! */
  { 9, 0, "Fujitsu #3"},
  };
  
  #if (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu model\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static char namebuf[128];
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
          {
            unsigned int l = strlen( cpuridtable[n].name );
            if ((!p[l] || isspace(p[l])) && memcmp( p, cpuridtable[n].name, l)==0)
            {
              detectedtype = 0x100 | ((cpuridtable[n].psr_impl & 0x0f)<<4) |
                             (cpuridtable[n].psr_vers & 0x0f);
              detectedname = cpuridtable[n].name;
              break;
            }
          }
          if (detectedtype == -1) /* found name but no ID */
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ )
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif
  
  if (detectedtype >= 0x100 && detectedname == NULL)
  {
    int psr_impl = ((detectedtype>>4)&0xf);
    int psr_vers = ((detectedtype   )&0xf);
    unsigned int n;
    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (( psr_impl == cpuridtable[n].psr_impl ) &&
          ( psr_vers == cpuridtable[n].psr_vers ))
      {
        detectedname = cpuridtable[n].name;
        break;
      }
    }
  }   
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_ALPHA)
static long __GetRawProcessorID(const char **cpuname)
{
  static char namebuf[30];
  static int detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static struct { int rid; const char *name;     } cpuridtable[] = {
                {       1, "EV3"                 },
                {       2, "EV4 (21064)"         },
                {       4, "LCA4 (21066/21068)"  },
                {       5, "EV5 (21164)"         },
                {       6, "EV4.5 (21064)"       },
                {       7, "EV5.6 (21164A)"      },
                {       8, "EV6 (21264)",        },
                {       9, "EV5.6 (21164PC)"     },
                {      10, "EV5.7"               }
                };
  
  #if (CLIENT_OS == OS_DEC_UNIX)
  if (detectedtype == -2)
  {
    long buf;
    int st = 0;
    detectedtype = -1;
    if (getsysinfo(GSI_PROC_TYPE,(char *)&buf,sizeof(buf),st,NULL,NULL)!=-1)
      detectedtype = (buf & 0xffff);
    if (detectedtype <= 0)
      detectedtype = -1;
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2L)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "cpu model\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static struct 
           { const char *sig;  int rid; } sigs[] = {
           { "EV3",            1      },
           { "EV4",            2      },
           { "Unknown 1",      3      }, /* 2.0.x kernel */
           { "Unknown",        3      }, /* 2.2.x kernel */
           { "LCA4",           4      },
           { "EV5",            5      },
           { "EV45",           6      },
           { "EV56",           7      },
           { "EV6",            8      },
           { "PCA56",          9      },
           { "PCA57",         10      }, /* 2.2.x kernel */
           };
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(sigs)/sizeof(sigs[0])); n++ )
          {
            unsigned int l = strlen( sigs[n].sig );
            if ((!p[l] || isspace(p[l])) && memcmp( p, sigs[n].sig, l)==0)
            {
              detectedtype = (long)sigs[n].rid;
              break;
            }
          }
          if (detectedtype == -1L)
          {
            for ( n = 0; *p && *p!='\r' && *p!='\n' && n<(sizeof(namebuf)-1); n++ )  
              namebuf[n] = *p++;
            namebuf[n] = '\0';
            detectedname = (const char *)&namebuf[0];
            detectedtype = 0;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #endif
  
  if (detectedtype > 0 && detectedname == NULL)
  {
    unsigned int n;
    detectedname = "";
    for ( n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++ )
    {
      if (detectedtype == cpuridtable[n].rid )
      {
        strcpy( namebuf, "Alpha " );
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  return detectedtype;
}
#endif

/* ---------------------------------------------------------------------- */

int GetProcessorType(int quietly)
{
  int coretouse = -1;
  const char *apd = "Automatic processor type detection ";
  #if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K) || \
      (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_X86) || \
      (CLIENT_CPU == CPU_ARM)     || (CLIENT_CPU == CPU_MIPS) || \
      (CLIENT_CPU == CPU_SPARC)   || (CLIENT_OS == OS_AIX)
  {
    const char *cpuname = NULL;
    long rawid = __GetRawProcessorID(&cpuname);
    if (!quietly)
    {
      if (rawid < 0)
        LogScreen("%s%s.\n", apd, ((rawid == -1L)?("failed"):("is not supported")));
      else if (rawid == 0)
        LogScreen("%sdid not\nrecognized the processor (tag: %s)\n", apd, (cpuname?cpuname:"???") );
      else if (cpuname == NULL || *cpuname == '\0')
        LogScreen("%sdid not\nrecognized the processor (id: %ld)\n", apd, rawid );
      else
        LogScreen("%sfound\na%s %s processor.\n",apd, 
           ((strchr("aeiou8", tolower(*cpuname)))?("n"):("")), cpuname);
    }
    #if (CLIENT_CPU == CPU_68K)
    if ((coretouse = ((rawid <= 0) ? (-1) : (((int)(rawid-68010L))/10))) == 6)
      coretouse = 5; /* remap 68060 to 68050 */
    #elif (CLIENT_CPU == CPU_ALPHA)
    coretouse = ((rawid <= 0) ? (-1) : ((int)rawid));
    #elif (CLIENT_OS == OS_AIX)
    if (rawid && ARCH_IS_POWER) // POWER
	coretouse=0;
    else if (rawid == 1) 
        coretouse=1;            // PowerPC 601
    else
        coretouse=2;            // PowerPC 604 and up

    #elif (CLIENT_CPU == CPU_POWERPC) && (CLIENT_OS != OS_AIX)
    coretouse = ((rawid < 0) ? (-1) : ((rawid==1L)?(0/*601*/):(1)));
    #elif (CLIENT_CPU == CPU_X86) /* way too many cpu<->core combinations */
    if (( rawid = __GetRawProcessorID(NULL,'c')) >= 0) coretouse = (int)rawid;
    #elif (CLIENT_CPU == CPU_ARM)
    if (rawid <= 0)                                coretouse =-1;
    else if (rawid == 0x3    || rawid == 0x600 ||
             rawid == 0x610  || rawid == 0x700 ||
             rawid == 0x7500 || rawid == 0x7500FE) coretouse = 0;
    else if (rawid == 0x810  || rawid == 0xA10)    coretouse = 1;
    else if (rawid == 0x200)                       coretouse = 2;
    else if (rawid == 0x710)                       coretouse = 3;
    #endif
  }
  #else
  {
    if (!quietly)
      LogScreen("%sis not supported.\n", apd );
    coretouse = -1;
  }
  #endif

  return (coretouse);
}

/* ---------------------------------------------------------------------- */

// GetTimesliceBaseline() returns a value that the ideal RC5 keyrate (kKeys
// per Mhz) would be IF a machine were running at peak efficiency. For
// non-preemptive systems, it is thus a good indicator of how low we can
// set the timeslice/rate-of-yield without losing efficiency. Or inversely,
// at what point OS responsiveness starts to suffer - which also applies to
// preemptive but non-mt systems handling of a break request.
//
// The function can also be used on non-mt systems to check for an excessive
// timeslice - on x86 systems an excessive timeslice is > 4*baseline

unsigned int GetTimesliceBaseline(void)
{
#if (CLIENT_CPU == CPU_X86)
  return __GetRawProcessorID(NULL, 'k');
#else
  return 0;
#endif
}

/* ---------------------------------------------------------------------- */

// Originally intended as tool to assist in managing the processor ID table
// for x86, I now (after writing it) realize that it could also get users on
// non-x86 platforms to send us *their* processor detection code. :) - Cyrus

void GetProcessorInformationStrings( const char ** scpuid, const char ** smaxscpus, const char ** sfoundcpus )
{
  const char *maxcpu_s, *foundcpu_s, *cpuid_s;

#if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K) || \
    (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_X86) || \
    (CLIENT_CPU == CPU_ARM)     || (CLIENT_CPU == CPU_MIPS) || \
    (CLIENT_CPU == CPU_SPARC)      
  long rawid = __GetRawProcessorID(&cpuid_s);
  if (rawid < 0)
    cpuid_s = ((rawid==-1)?("?\n\t(identification failed)"):
              ("none\n\t(client does not support identification)"));
  else
  {
    static char namebuf[60];
    if (cpuid_s == NULL) cpuid_s = "*unknown*";
    if (*cpuid_s =='\0') cpuid_s = "???";
  #if (CLIENT_CPU == CPU_ARM)
    namebuf[0] = '\0';
    if (rawid != 0) /* if rawid == 0, then cpuid_s == "%lX" */
      sprintf( namebuf, "%lX\n\tname: ", rawid );
    strcat( namebuf, cpuid_s ); /* always valid */
    #if (CLIENT_OS == OS_RISCOS)
    if (riscos_count_cpus() == 2)
      strcat(strcat(namebuf,"\n\t+ "),riscos_x86_determine_name());
    #endif
  #elif (CLIENT_CPU == CPU_X86)
    namebuf[0] = '\0';
    if (rawid != 0) /* if rawid == 0, then cpuid_s == "%04x:%04x" */
      sprintf( namebuf, "%04X:%04X\n\tname: ",(int)((rawid>>16)&0xffff),(int)(rawid&0xffff));
    strcat( namebuf, cpuid_s ); /* always valid */
  #else
    sprintf(namebuf, "%ld\n\tname: %s\n", rawid, cpuid_s );
  #endif
    cpuid_s = ((const char *)(&namebuf[0]));
  }    
#else    
  cpuid_s = "none\n\t(client does not support identification)";
#endif

  #if defined(CLIENT_SUPPORTS_SMP)
    static char maxcpu_b[80];
    sprintf( maxcpu_b, "%d", (int)GetNumberOfSupportedProcessors() );
    maxcpu_s = ((const char *)(&maxcpu_b[0]));
  #elif (!defined(CORES_SUPPORT_SMP))
    maxcpu_s = "1\n\t(cores are not thread-safe)";
  #elif (CLIENT_OS == OS_RISCOS)
    maxcpu_s = "2\n\t(with RiscPC x86 card)";
  #else
    maxcpu_s = "1\n\t(OS or client-build does not support threads)";
  #endif

  int cpucount = GetNumberOfDetectedProcessors();
  if (cpucount < 1)
    foundcpu_s = "1\n\t(OS does not support detection)";
  else
  {
    static char foundcpu_b[6];
    sprintf( foundcpu_b, "%d", cpucount );
    foundcpu_s = ((const char *)(&foundcpu_b[0]));
  }

  if ( scpuid ) *scpuid = cpuid_s;
  if ( smaxscpus ) *smaxscpus = maxcpu_s;
  if ( sfoundcpus ) *sfoundcpus = foundcpu_s;
  return;
}

/* ---------------------------------------------------------------------- */

void DisplayProcessorInformation(void)
{
  const char *scpuid, *smaxscpus, *sfoundcpus;
  GetProcessorInformationStrings( &scpuid, &smaxscpus, &sfoundcpus );

  LogScreenRaw("Automatic processor identification tag: %s\n"
    "Number of processors detected by this client: %s\n"
    "Number of processors supported by this client: %s\n",
    scpuid, sfoundcpus, smaxscpus );
  return;
}

/* ---------------------------------------------------------------------- */
