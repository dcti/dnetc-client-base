/*
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * This module contains hardware identification stuff.
 * See notes on implementing __GetRawProcessorID() below.
 *
*/
const char *cpucheck_cpp(void) {
return "@(#)$Id: cpucheck.cpp,v 1.103 1999/12/26 21:02:28 patrick Exp $"; }

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
#elif (CLIENT_OS == OS_FREEBSD)
#  include <sys/sysctl.h>
#elif (CLIENT_OS == OS_NETBSD)
#  include <sys/param.h>
#  include <sys/sysctl.h>
#endif

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

/* ------------------------------------------------------------------------ */

int GetNumberOfDetectedProcessors( void )  //returns -1 if not supported
{
  static int cpucount = -2;

  if (cpucount == -2)
  {
    cpucount = -1;
    #if (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_BSDOS) || \
        (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD)
    { /* comment out if inappropriate for your *bsd - cyp (25/may/1999) */
      int ncpus; size_t len = sizeof(ncpus);
      int mib[2]; mib[0] = CTL_HW; mib[1] = HW_NCPU;
      if (sysctl( &mib[0], 2, &ncpus, &len, NULL, 0 ) == 0)
      //if (sysctlbyname("hw.ncpu", &ncpus, &len, NULL, 0 ) == 0)
        cpucount = ncpus;
    }
    #elif (CLIENT_OS == OS_HPUX)
    {
      struct pst_dynamic psd;
      if (pstat_getdynamic(&psd, sizeof(psd), (size_t)1, 0) !=-1)
      cpucount = (int)psd.psd_proc_cnt;
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
      #if (CLIENT_CPU == CPU_ARM)
        cpucount = 1;
      #else
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
      #endif // (CLIENT_CPU == CPU_ARM)
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
      //cpucount = _system_configuration.ncpus; 
      //should work the same way but might go
    }
    #elif (CLIENT_OS == OS_RISCOS)
    {
      cpucount = riscos_count_cpus();
    }
    #elif (CLIENT_OS == OS_MACOS)
    {
      cpucount = 1; // I am just a workaround: FIX ME!
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

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER)
static long __GetRawProcessorID(const char **cpuname)
{
  /* ******* detected type reference is (PVR value >> 16) *********** */
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static int ispower = 0;
  static const char *detectedname = NULL;
  static char namebuf[30];
  static struct { long rid; const char *name; int powername; } cpuridtable[] = {
    //note: if the name is not prefixed with "Power", it defaults to "PowerPC"
    //note: Non-PVR based numbers start at 0x10000 (real PVR numbers are 16bit)
                {         1,   "601"                   },
                {         3,   "603"                   },
                {         4,   "604"                   },
                {         6,   "603e"                  },
                {         7,   "603ev"                 },
                {         8,   "740/750/G3"            },
                {         9,   "604e"                  },
                {        10,   "604ev"                 },
                {        12,   "7400/G4"               },
                {        50,   "821"                   },
                {        80,   "860"                   },
                {(1L<<16)+1,   "Power RS"              }, //not PVR based
                {(1L<<16)+2,   "Power RS2 Superchip"   }, //not PVR based
                {(1L<<16)+3,   "Power RS2"             }, //not PVR based
                {(1L<<16)+4,   "620"                   }, //not PVR based
                {(1L<<16)+5,   "630"                   }, //not PVR based
                {(1L<<16)+6,   "A35"                   }, //not PVR based
                {(1L<<16)+7,   "RS64II"                }, //not PVR based
                };
  #if (CLIENT_OS == OS_AIX)
  if (detectedtype == -2L)
  { 
    /* extract from src/bos/kernel/sys/POWER/systemcfg.h 1.12 */
    #ifndef POWER_RS1
    #  define POWER_RS1 0x0001
    #endif
    #ifndef POWER_RSC
    #  define POWER_RSC 0x0002
    #endif
    #ifndef POWER_RS2
    #  define POWER_RS2 0x0004
    #endif
    #ifndef POWER_601
    #  define POWER_601 0x0008
    #endif
    #ifndef POWER_603
    #  define POWER_603 0x0020
    #endif
    /* if compiled on older versions of 4.x ... */
    #ifndef POWER_620
    #  define POWER_620 0x0040
    #endif
    #ifndef POWER_630
    #  define POWER_630 0x0080
    #endif
    #ifndef POWER_A35
    #  define POWER_A35 0x0100
    #endif
    #ifndef POWER_RS64II
    #  define POWER_RS64II 0x0200 /* RS64-II class CPU */
    #endif
    static struct { long imp;   long rid; } cpumap[] = {
                  { POWER_601,            1 },
                  { POWER_603,            3 },
                  { POWER_604,            4 },
                  { POWER_RS1,   (1L<<16)+1 },
                  { POWER_RSC,   (1L<<16)+2 },
                  { POWER_RS2,   (1L<<16)+3 },
                  { POWER_620,   (1L<<16)+4 },
                  { POWER_630,   (1L<<16)+5 },
                  { POWER_A35,   (1L<<16)+6 },
                  { POWER_RS64II,(1L<<16)+7 },
                  };
    unsigned int imp_i;
    detectedtype = -1L; /* assume failed */
    if ( _system_configuration.architecture == POWER_RS ) 
      ispower = 1;
    for (imp_i = 0; imp_i < (sizeof(cpumap)/sizeof(cpumap[0])); imp_i++)
    {
      if (cpumap[imp_i].imp == _system_configuration.implementation )
      {
        detectedtype = cpumap[imp_i].rid;
        break;
      }
    }
    if (detectedtype == -1L) /* ident failed */
    {
      sprintf( namebuf, "impl:0x%lX", _system_configuration.implementation );
      detectedname = (const char *)&namebuf[0];
      if (ispower) /* if POWER CPU, then don't let ident fail */
      {            /*   - we need the power bit in the retval */
        detectedtype = (1L<<16)+_system_configuration.implementation;
      }
    }
  }
  #elif (CLIENT_OS == OS_MACOS)
  if (detectedtype == -2L)
  {
    // Note: need to use gestaltNativeCPUtype in order to get the correct
    // value for G3 upgrade cards in a 601 machine.
    // Some Mac people are idiots, so I'll spell it out again:
    // ******* detected type reference is (PVR value >> 16) ***********
    // PVR is a hardware value from the cpu and is available on every 
    // PPC CPU on every PPC Based OS. So, dimwits, don't just make up 
    // cpu numbers!
    long result;
    detectedtype = -1;
    if (Gestalt(gestaltNativeCPUtype, &result) == noErr)
      detectedtype = result - 0x100L; // PVR!!
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
           { "750",                  8  },
           { "604e",                 9  },
           { "604ev",               10  }
           { "821",                 50  },
           { "860",                 80  }
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
      if (cpuridtable[n].rid == detectedtype )
      {
        if (strlen( cpuridtable[n].name )>=6 &&
            memcmp( cpuridtable[n].name, "Power ", 6 )==0)
          namebuf[0] = '\0';
        else
          strcpy( namebuf, "PowerPC ");
        strcat( namebuf, cpuridtable[n].name );
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;
  if (detectedtype >= 0 && ispower)
    return ((1L<<24) | detectedtype);
  return detectedtype;
}
#endif /* (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER) */

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
  static long detectedtype = -2L;  /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static int  kKeysPerMhz = 512; /* default rate if not found */
  static int  simpleid   = 0;   /* default id if not found */
  
  if ( detectedtype == -2L )
  {
    static char namebuf[30];
    const char *vendorname = NULL;
    struct cpuxref { int cpuid, kKeysPerMhz, simpleid; 
                     const char *cpuname; } *internalxref = NULL;
    u32 dettype     = x86ident();
    int cpuidbmask  = 0xfff0; /* mask with this to find it in the table */
    int cpuid       = (((int)dettype) & 0xffff);
    int vendorid    = (((int)(dettype >> 16)) & 0xffff);

    if (vendorid == 0x6E49 ) /* 'nI': broken x86ident */
      vendorid = 0x6547; /* 'eG' */
  
    sprintf( namebuf, "%04X:%04X", vendorid, cpuid );
    detectedname = (const char *)&namebuf[0];
    detectedtype = -1; /* assume not found */
  
    if ( vendorid == 0x7943 /* 'yC' */ ) // Cyrix CPU
    {
      static struct cpuxref cyrixxref[]={
          {    0x40,  950,     6, "486"       }, // Pentium or SMC core
          {  0x0440,  950,     0, "MediaGX"   },
          {  0x0490, 1185,     0, "5x86"      },
          {  0x0520, 2090,     3, "6x86"      }, // "Cyrix 6x86/6x86MX/M2"
          {  0x0540, 1200,     0, "GXm"       }, // use Pentium core here too
          {  0x0600, 2115, 0x103, "6x86MX"    },
          {  0x0000, 2115,    -1, NULL        }
          }; internalxref = &cyrixxref[0];
      vendorname = "Cyrix ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x6952 /* 'iR' */  ) //"RiseRiseRiseRise"
    {
      static struct cpuxref risexref[]={
          {  0x0500, 1500,     0, "mP6" }, /* (0.25 æm) */
          {  0x0500, 1500,     0, "mP6" }, /* (0.18 æm) */
          {  0x0000, 2115,    -1, NULL  }
          }; internalxref = &risexref[0];
      vendorname = "Rise ";
      cpuidbmask = 0xfff0;
    }
    else if ( vendorid == 0x6543 /* 'eC' */ ) //"CentaurHauls"
    {
      static struct cpuxref centaurxref[]={
          {  0x0540, 1200,0x100, "C6"          }, // use Pentium core
          {  0x0585, 1346,0x108, "WinChip 2"   }, // pentium Pro (I think)
          {  0x0000, 1346,   -1, NULL          }
          }; internalxref = &centaurxref[0];
      vendorname = "Centaur/IDT ";
      cpuidbmask = 0xfff0;
    }
    else if ( vendorid == 0x654E /* 'eN' */  ) //"NexGenDriven"
    {   
      static struct cpuxref nexgenxref[]={
          {  0x0300, 1500,     1, "Nx586" }, //386/486 core
          {  0x0000, 1500,    -1, NULL  } //no such thing
          }; internalxref = &nexgenxref[0];
      vendorname = "NexGen ";
      cpuidbmask = 0xfff0;
    }
    else if ( vendorid == 0x4D55 /* 'MU' */  ) //"UMC UMC UMC "
    {   
      static struct cpuxref umcxref[]={
          {  0x0410, 1500,     0, "U5D" },
          {  0x0420, 1500,     0, "U5S" },
          {  0x0400, 1500,    -1, NULL  }
          }; internalxref = &umcxref[0];
      vendorname = "UMC ";
      cpuidbmask = 0xfff0;
    }
    else if ( vendorid == 0x7541 /* 'uA' */ ) // "AuthenticAMD"
    {
      static struct cpuxref amdxref[]={
          {  0x0040,  950,     0, "486"      }, // "Pentium, Pentium MMX, Cyrix 486/5x86/MediaGX, AMD 486",
          {  0x0430,  950,     0, "486DX2"   },
          {  0x0470,  950,     0, "486DX2WB" },
          {  0x0480,  950,     0, "486DX4"   },
          {  0x0490,  950,     0, "486DX4WB" },
          {  0x04E0, 1185,     0, "5x86"     },
          {  0x04F0, 1185,     0, "5x86WB"   },
          {  0x0500, 2353,     4, "K5 PR75, PR90, or PR100" }, // use K5 core
          {  0x0510, 2353,     4, "K5 PR120 or PR133" },
          {  0x0520, 2353,     4, "K5 PR166" },
          {  0x0530, 2353,     4, "K5 PR200" },
          {  0x0560, 1611, 0x105, "K6"       },
          {  0x0570, 1611, 0x105, "K6"       },
          {  0x0580, 1690, 0x105, "K6-2"     },
          {  0x0590, 1690, 0x105, "K6-3"     },
          {  0x0610, 3400, 0x109, "K7"       },
          /* There may be a split personality issue here:
             7541:0612  600 MHz K7: core #2 gets 1.798 Mkey/sec, 
                              while core #3 gets 1.809 Mkey/sec consistently.
             However, I now have two reports (no email addy unfortunately)
             of core #2 being definitely faster. That may be the later
             series K7 (7541:062x). Needs checking.
          */
          {  0x0620, 3400, 0x103, "K7-2"     },
          {  0x0000, 4096,    -1, NULL       }
          }; internalxref = &amdxref[0];
      vendorname = "AMD ";
      cpuidbmask = 0xfff0; //strip last 4 bits, don't need stepping info
    }
    else if ( vendorid == 0x6547 /* 'eG' */ ) // "GenuineIntel"
    {
      static struct cpuxref intelxref[]={
          {  0x0030,  426,     1, "80386"      },   // generic 386/486 core
          {  0x0040,  950,     1, "80486"      },
          {  0x0400,  950,     1, "486DX 25 or 33" },
          {  0x0410,  950,     1, "486DX 50" },
          {  0x0420,  950,     1, "486SX" },
          {  0x0430,  950,     1, "486DX2" },
          {  0x0440,  950,     1, "486SL" },
          {  0x0450,  950,     1, "486SX2" },
          {  0x0470,  950,     1, "486DX2WB" },
          {  0x0480,  950,     1, "486DX4" },
          {  0x0490,  950,     1, "486DX4WB" },
          {  0x0500, 1416,     0, "Pentium" }, //stepping A
          {  0x0510, 1416,     0, "Pentium" },
          {  0x0520, 1416,     0, "Pentium" },
          {  0x0530, 1416,     0, "Pentium Overdrive" },
          {  0x0540, 1432, 0x100, "Pentium MMX" },
          {  0x0570, 1416,     0, "Pentium" },
          {  0x0580, 1432, 0x100, "Pentium MMX" },
          {  0x0600, 2785,     8, "Pentium Pro" },
          {  0x0610, 2785,     8, "Pentium Pro" },
          /*
          A80522, Klamath (0.28 æm)
          A80523, Deschutes (0.25 æm)
          Tonga (0.25 æm mobile) - 0x0650+0
          Covington (no On-Die L2 Cache)
          Mendocino (128 KB On-Die L2 Cache) 0x0660
          Dixon (256 KB On-Die L2 Cache) 
          Intel P6-core 
          3 P2 (0.28 æm) 
          5 P2 (0.25 æm)  
          6 P2 with on-die L2 cache 
          */
          {  0x0630, 2785, 0x102, "Pentium II" },
          {  0x0650, 2785, 0x102, "Pentium II" }, //0x0650=mobile,651=boxed PII/Xeon
          {  0x0660, 2785, 0x102, "Celeron-A" }, //on-die L2 
          {  0x0670, 2785, 0x102, "Pentium III" },
          {  0x0680, 2785, 0x102, "Pentium III" },
          {  0x0000, 4096,    -1, NULL }
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
          simpleid    = internalxref[pos].simpleid;
          detectedtype = dettype;
          if (detectedtype < 0)
            detectedtype = -1;
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
    return ((long)simpleid);
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
  static char namebuf[60];

  #if (CLIENT_OS == OS_RISCOS)
  if ( detectedtype == -2L )
  {
    /*
       ARMident() will throw SIGILL on an ARM 2 or ARM 250, because they 
       don't have the system control coprocessor. (We ignore the ARM 1 
       because I'm not aware of any existing C++ compiler that targets it...)
     */
    signal(SIGILL, ARMident_catcher);
    if (setjmp(ARMident_jmpbuf))
      detectedtype = 0x2000;
    else
      detectedtype = ARMident();
    signal(SIGILL, SIG_DFL);
    detectedtype = (detectedtype >> 4) & 0xfff; // extract part number field

    if ((detectedtype & 0xf00) == 0) //old-style ID (ARM 3 or prototype ARM 600)
      detectedtype <<= 4;            // - shift it into the new form
    if (detectedtype == 0x300)
    {
      detectedtype = 3;
    }
    else if (detectedtype == 0x710)
    {
      // the ARM 7500 returns an ARM 710 ID - need to look at its
      // integral IOMD unit to spot the difference
      u32 detectediomd = IOMDident();
      detectediomd &= 0xff00; // just want most significant byte
      if (detectediomd == 0x5b00)
        detectedtype = 0x7500;
      else if (detectediomd == 0xaa00)
        detectedtype = 0x7500FEL;
    }
    detectedname = ((const char *)&(namebuf[0]));
    switch (detectedtype)
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
      case 0x810: sprintf( namebuf, "ARM %lX", detectedtype );
                  break;
      default:    sprintf( namebuf, "%lX", detectedtype );
                  detectedtype = 0;
                  break;
    }
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2)
  {
    FILE *cpuinfo;
    detectedtype = -1L;
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo)) 
      {
        const char *p = "Type\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static struct 
           { const char *sig;  int rid; } sigs[] ={
               { "arm2",       0x200},
               { "arm250",     0x250},
               { "arm3",       0x3},
               { "arm6",       0x600},
               { "arm610",     0x610},
               { "arm7",       0x700},
               { "arm710",     0x710},
               { "sa110",      0xA10}
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

          detectedname = ((const char *)&(namebuf[0]));
          switch (detectedtype)
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
          case 0x810: sprintf( namebuf, "ARM %lX", detectedtype );
              break;
          default:    sprintf( namebuf, "%lX (unknown)", detectedtype );
              detectedtype = 0;
              break;
          }
          break;
        }
      }
      fclose(cpuinfo);
    }
  }
  #elif (CLIENT_OS == OS_NETBSD)
  if (detectedtype == -2)
  {
      char buffer[256];
      int len = 255;
      int mib[2];
      mib[0]=CTL_HW; mib[1]=HW_MODEL;
      if (sysctl(mib, 2,&buffer[0], &len, NULL, 0 ) == 0)
      {
          int n = 0;
          char *p;
          static struct 
          { const char *sig;  int rid; } sigs[] ={
              { "ARM2",       0X200},
              { "ARM250",     0X250},
              { "ARM3",       0X3},
              { "ARM6",       0X600},
              { "ARM610",     0X610},
              { "ARM7",       0X700},
              { "ARM710",     0X710},
              { "SA-110",     0XA10}
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
          
          detectedname = ((const char *)&(namebuf[0]));
          switch (detectedtype)
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
          case 0x810: sprintf( namebuf, "ARM %lX", detectedtype );
              break;
          default:    sprintf( namebuf, "%lX (unknown)", detectedtype );
              detectedtype = 0;
              break;
          }
      }
  }
  #endif
  
  if ( cpuname )
    *cpuname = detectedname;
  return detectedtype;
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

long GetProcessorType(int quietly)
{
  long retval = -1L;
  const char *apd = "Automatic processor type detection ";
  #if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K) || \
      (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_X86) || \
      (CLIENT_CPU == CPU_ARM)     || (CLIENT_CPU == CPU_MIPS) || \
      (CLIENT_CPU == CPU_SPARC)
  {
    const char *cpuname = NULL;
    long rawid = __GetRawProcessorID(&cpuname);
    if (!quietly)
    {
      if (rawid < 0)
        LogScreen("%s%s.\n", apd, ((rawid == -1L)?("failed"):("is not supported")));
      else if (rawid == 0)
        LogScreen("%sdid not\nrecognize the processor (tag: \"%s\")\n", apd, (cpuname?cpuname:"???") );
      else if (cpuname == NULL || *cpuname == '\0')
        LogScreen("%sdid not\nrecognize the processor (id: %ld)\n", apd, rawid );
      else
        LogScreen("%sfound\na%s %s processor.\n",apd, 
           ((strchr("aeiou8", tolower(*cpuname)))?("n"):("")), cpuname);
    }
    #if (CLIENT_CPU == CPU_X86) /* simply too many core<->cpu combinations */
    if (rawid >= 0)             /* so return a simplified id */
    {
      if ((rawid = __GetRawProcessorID(NULL,'c')) >= 0)
        retval = rawid;
    }
    #else
    if (rawid >= 0)             /* let selcore figure things out */
      retval = rawid; 
    #endif
  }  
  #else
  {
    if (!quietly)
      LogScreen("%sis not supported.\n", apd );
  }
  #endif
  return retval;
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
  {
    maxcpu_s = "128"; /* just some arbitrary number */
    #if (CLIENT_OS == OS_RISCOS) && defined(HAVE_X86_CARD_SUPPORT)
    if (GetNumberOfDetectedProcessors() > 1)
      maxcpu_s = 2; /* thread 0 is ARM, thread 1 is x86 */
    #endif
  }
  #elif (!defined(CORES_SUPPORT_SMP))
    maxcpu_s = "1\n\t(cores are not thread-safe)";
  #elif (CLIENT_OS == OS_RISCOS)
    #if defined(HAVE_X86_CARD_SUPPORT)
    maxcpu_s = "2\n\t(with RiscPC x86 card)";
    #else
    maxcpu_s = "1\n\t(client-build does not support multiple processors)";
    #endif
  #else
    maxcpu_s = "1\n\t(OS or client-build does not support multiple processors)";
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
