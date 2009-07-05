/*
 * Copyright distributed.net 1997-2009 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * This module contains hardware identification stuff.
 * See notes on implementing __GetRawProcessorID() below.
 *
*/
const char *cpucheck_cpp(void) {
return "@(#)$Id: cpucheck.cpp,v 1.165 2009/07/05 17:56:03 stream Exp $"; }

#include "cputypes.h"
#include "baseincs.h"  // for platform specific header files
#include "cpucheck.h"  //just to keep the prototypes in sync.
#include "logstuff.h"  //LogScreen()/LogScreenRaw()
#include "sleepdef.h"  //usleep

#if (CLIENT_OS == OS_DEC_UNIX)
#  include <unistd.h>
#  include <sys/sysinfo.h>
#  include <machine/hal_sysinfo.h>
#  include <machine/cpuconf.h>
#elif (CLIENT_OS == OS_AIX)
#  include <sys/systemcfg.h>
#elif (CLIENT_OS == OS_MACOSX)
#  include <mach/mach.h>
#  include <mach/machine.h>
#  include <IOKit/IOKitLib.h>
#elif (CLIENT_OS == OS_DYNIX)
#  include <sys/tmp_ctl.h>
#elif (CLIENT_OS == OS_SOLARIS)
#  include <string.h>
#  include <sys/types.h>
#  include <sys/processor.h>
#elif (CLIENT_OS == OS_NETWARE6)
#include <nks/plat.h>
#elif (CLIENT_OS == OS_MORPHOS)
#  include <exec/resident.h>
#  include <exec/system.h>
#endif

#if (CLIENT_CPU == CPU_CELLBE)
#include <libspe2.h>
#endif

#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
#include "x86id.h"
#endif

#if (CLIENT_CPU == CPU_CUDA)
#include "cuda_info.h"
#endif

#if (CLIENT_CPU == CPU_ATI_STREAM)
#include "amdstream_info.h"
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

// returns -1 if not supported
// returns 0 if required co-processor (e.g. GPU) was not found
int GetNumberOfDetectedProcessors( void )
{
  static int cpucount = -2;

  if (cpucount == -2)
  {
    cpucount = -1;
    #if (CLIENT_CPU == CPU_CUDA)
    {
      if ((cpucount = GetNumberOfDetectedCUDAGPUs()) <= 0) {
        LogScreen("No CUDA-supported GPU found.\n");
        cpucount = -99;
      }
    }
    #elif (CLIENT_CPU == CPU_ATI_STREAM)
    {
      cpucount=getAMDStreamDeviceCount();
      if (cpucount<=0) {
        LogScreen("No ATI Stream compatible device found.\n");
        cpucount = -99;
      }
    }
    #elif (CLIENT_OS == OS_FREEBSD) || (CLIENT_OS == OS_BSDOS) || \
        (CLIENT_OS == OS_OPENBSD) || (CLIENT_OS == OS_NETBSD) || \
        (CLIENT_OS == OS_DRAGONFLY)
    { /* comment out if inappropriate for your *bsd - cyp (25/may/1999) */
      int ncpus; size_t len = sizeof(ncpus);
      int mib[2]; mib[0] = CTL_HW; mib[1] = HW_NCPU;
      if (sysctl( &mib[0], 2, &ncpus, &len, NULL, 0 ) == 0)
      //if (sysctlbyname("hw.ncpu", &ncpus, &len, NULL, 0 ) == 0)
        cpucount = ncpus;
    }
    #elif (CLIENT_OS == OS_MACOSX)
    {
      unsigned int    count;
      struct host_basic_info  info;
      count = HOST_BASIC_INFO_COUNT;
      if (host_info(mach_host_self(), HOST_BASIC_INFO, (host_info_t)&info,
          &count) == KERN_SUCCESS)
         cpucount=info.avail_cpus;
    }
    #elif (CLIENT_OS == OS_NEXTSTEP)
    {
      unsigned int    count;
      struct host_basic_info  info;
      count = HOST_BASIC_INFO_COUNT;
      if (host_info(host_self(), HOST_BASIC_INFO, (host_info_t)&info,
          &count) == KERN_SUCCESS)
         cpucount=info.avail_cpus;
    }
    #elif (CLIENT_OS == OS_HPUX) && defined(OS_SUPPORTS_SMP)
    {                          //multithreaded clients are special
      struct pst_dynamic psd;
      if (pstat_getdynamic(&psd, sizeof(psd), (size_t)1, 0) !=-1)
      cpucount = (int)psd.psd_proc_cnt;
    }
    #elif (CLIENT_OS == OS_BEOS) || (CLIENT_OS == OS_HAIKU)
    {
      system_info the_info;
      get_system_info(&the_info);
      cpucount = the_info.cpu_count;
    }
    #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
    {
      SYSTEM_INFO systeminfo;
      GetSystemInfo(&systeminfo);
      cpucount = systeminfo.dwNumberOfProcessors;
      if (cpucount < 1)
        cpucount = -1;
    }
    #elif (CLIENT_OS == OS_NETWARE)
    {
      cpucount = GetNumberOfRegisteredProcessors();
    }
    #elif (CLIENT_OS == OS_NETWARE6)
    {
      cpucount = NXGetCpuCount();
    }
    #elif (CLIENT_OS == OS_OS2)
    {
      int rc = (int) DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                  &cpucount, sizeof(cpucount));
      if (rc != 0 || cpucount < 1)
        cpucount = -1;
    }
    #elif ((CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_PS2LINUX)) && \
           (CLIENT_CPU != CPU_CELLBE)
    {
      #if (CLIENT_CPU == CPU_ARM) || (CLIENT_CPU == CPU_MIPS)
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
          #if (CLIENT_CPU == CPU_X86      || \
               CLIENT_CPU == CPU_AMD64    || \
               CLIENT_CPU == CPU_POWERPC  || \
               CLIENT_CPU == CPU_S390     || \
               CLIENT_CPU == CPU_S390X    || \
               CLIENT_CPU == CPU_PA_RISC)
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
          else if (memcmp(buffer, "cpus detected\t",14)==0) /* 2.4.1 */
          {                          /* "cpus detected\t\t: 4" */
            char *p = &buffer[14];
            while (*p == '\t' || *p == ':' || *p == ' ')
              p++;
            if ((p > &buffer[14]) && isdigit(*p))
            {
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
    #elif (CLIENT_CPU == CPU_CELLBE)
    {
      // Each Cell has 1 PPE, which is dual-threaded (so in fact the OS sees
      // 2 processors), but it has been found that running 2 simultaneous
      // threads degrades performance, so let's pretend there's only one
      // PPE. Then add the number of usable SPEs.
      cpucount = spe_cpu_info_get(SPE_COUNT_PHYSICAL_CPU_NODES, -1) +
                 spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);
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
    #elif (CLIENT_OS == OS_QNX) && defined(__QNXNTO__) /* neutrino */
    {
      cpucount = _syspage_ptr->num_cpu;
    }
    #elif (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_MORPHOS)
    {
      cpucount = 1;
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
    #elif (CLIENT_CPU == CPU_68K) // no such thing as 68k/mp
      cpucount = 1;               // that isn't covered above
    #elif (CLIENT_OS == OS_DYNIX)
      int nprocs = tmp_ctl(TMP_NENG, 0);
      int i;
      cpucount = 0;
      for (i = 0; i < nprocs; i++)
        if (TMP_ENG_ONLINE == tmp_ctl(TMP_QUERY, i)) cpucount++;
    #endif

    if (cpucount == -99)  // required co-processor (e.g. GPU) was not found
      cpucount = 0;
    else if (cpucount < 1)  // not supported
      cpucount = -1;
  }

  return cpucount;
}

/* ---------------------------------------------------------------------- */

int GetNumberOfLogicalProcessors ( void )
{ 
  return GetNumberOfDetectedProcessors();
}

/* ---------------------------------------------------------------------- */

int GetNumberOfPhysicalProcessors ( void )
{
  static int cpucount = -1;
  
  if (cpucount < 0)
  {
#if (CLIENT_CPU == CPU_CELLBE)
    cpucount = spe_cpu_info_get(SPE_COUNT_PHYSICAL_CPU_NODES, -1);
#else
    cpucount = GetNumberOfDetectedProcessors();
#endif
    if (cpucount < 0)
      cpucount = 1;
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
      detectedtype = 68000L; // 68000
  }
  #elif (CLIENT_OS == OS_NEXTSTEP)
  if (detectedtype == -2)
  {
    struct host_basic_info info;
    unsigned int count = HOST_BASIC_INFO_COUNT;

    if (host_info(host_self(), HOST_BASIC_INFO,
                  (host_info_t)&info, &count) == KERN_SUCCESS &&
        info.cpu_type == CPU_TYPE_MC680x0)
    {
      switch (info.cpu_subtype)
      {
          /* MC68030_ONLY shouldn't be returned since it's only used
          ** to mark 68030-only executables in the mach-o
          ** fileformat */
        case CPU_SUBTYPE_MC68030_ONLY:
        case CPU_SUBTYPE_MC68030:      detectedtype = 68030L; break;
        case CPU_SUBTYPE_MC68040:      detectedtype = 68040L; break;

          /* black hardware from NeXT only had 680[34]0 processors so
          ** there are no defines for the others *shrug* */
        default:                       detectedtype = -1;     break;
      }
    } else {
      /* something went really wrong here if cpu_type doesn't match -
      ** can we be compiled for NeXTstep on 68k but run on something
      ** else? */
      detectedtype = -1;
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

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)

/* note: Non-PVR based numbers start at 0x10000 (real PVR numbers are 16bit) */
# define NONPVR(x) ((1L << 16) + (x))

static long __GetRawProcessorID(const char **cpuname)
{
  /* ******* detected type reference is (PVR value >> 16) *********** */
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static int isaltivec = 0;
  static const char *detectedname = NULL;
  static char namebuf[30];

  /* note: "PowerPC" will be prepended to the name
  ** note: always use PVR IDs :
  ** http://www.freescale.com/files/archives/doc/support_info/PPCPVR.pdf
  */
  struct { long rid;
           const char *name; }
    cpuridtable[] = {
      {    0x0001, "601"                 },
      {    0x0003, "603"                 },
      {    0x0004, "604"                 },
      {    0x0006, "603e"                },
      {    0x0007, "603r/603ev"          }, //ev=0x0007, r=0x1007
      {    0x0008, "740/750 (G3)"        },
      {    0x0009, "604e"                },
      {    0x000A, "604ev"               },
      {    0x000C, "7400 (G4)"           },
      {    0x0020, "403G/403GC/403GCX"   },
      {    0x0039, "970 (G5)"            },
      {    0x003C, "970FX (G5)"          }, // XServe G5. See bug #3675
      {    0x0044, "970MP (G5)"          }, // Dual core
      {    0x0050, "821"                 },
      {    0x0070, "Cell Broadband Engine" },
      {    0x0080, "860"                 },
      {    0x0081, "8240"                },
      {    0x4011, "405GP"               },
      {    0x4012, "440GP"               },
      {    0x41F1, "405LP"               },
      {    0x4222, "440EP/440GR"         },
      {    0x5091, "405GPr"              },
      {    0x5121, "405EP"               },
      {    0x51B2, "440GX"               },
      {    0x5322, "440SP"               },
      {    0x7000, "750FX"               },
      {    0x8000, "7441/7450/7451 (G4)" },
      {    0x8001, "7445/7455 (G4)"      },
      {    0x8002, "7447/7457 (G4)"      },
      {    0x8003, "7447A (G4)"          },
      {    0x8004, "7448 (G4)"           },
      {    0x800C, "7410 (G4)"           },
      {    0x8081, "5200 (G2)"           },
      {    0x8082, "5200 (G2-LE)"        },
      { NONPVR(1), "620"                 }, //not PVR based
      { NONPVR(2), "630"                 }, //not PVR based
      { NONPVR(3), "A35"                 }, //not PVR based
      { NONPVR(4), "RS64II"              }, //not PVR based
      { NONPVR(5), "RS64III"             }, //not PVR based
      {    0x0800, "POWER_4"             }, //not PVR based
      {    0x2000, "POWER_5"             }, //not PVR based
    };

  #if (CLIENT_OS == OS_AIX)
  if (detectedtype == -2L) {
    if (_system_configuration.architecture != POWER_RS) {
      unsigned int i;
      struct { long imp;
               long rid; }
        cpumap[] = {
          { POWER_601,             1 },
          { POWER_603,             3 },
          { POWER_604,             4 },
          { POWER_620,     NONPVR(1) },
          { POWER_630,     NONPVR(2) },
          { POWER_A35,     NONPVR(3) },
          { POWER_RS64II,  NONPVR(4) },
          { POWER_RS64III, NONPVR(5) },
          { POWER_4,            2048 }, 
          { POWER_5,            8192 }};

      /* assume failed */
      detectedtype = -1L;

      for (i = 0; i < (sizeof(cpumap)/sizeof(cpumap[0])); i++) {
        if (cpumap[i].imp == _system_configuration.implementation ) {
          detectedtype = cpumap[i].rid;
          break;
        }
      }

      if (detectedtype == -1L) {
        /* identification failed */
        sprintf(namebuf, "impl:0x%X", _system_configuration.implementation);
        detectedname = (const char *)&namebuf[0];
      }
    } else {
      /* just signal that we've got a Power CPU here */
      detectedtype = (1 << 24);
      detectedname = "Power";
    }
  }
  #elif (CLIENT_OS == OS_MACOSX)
  if (detectedtype == -2L)
  {
    // We prefer raw PVR values over the IDs provided by host_info()
    CFDataRef value = NULL;
    io_object_t device;
    io_iterator_t objectIterator;
    CFMutableDictionaryRef properties = NULL;
    mach_port_t master_port;
    detectedtype = -1L;

    // In I/O Registry Search for "IOPlatformDevice" devices
    if (kIOReturnSuccess == IOMasterPort(MACH_PORT_NULL, &master_port)) {
      if (kIOReturnSuccess == IOServiceGetMatchingServices(master_port, IOServiceMatching("IOPlatformDevice"), &objectIterator)) {
        // Test the results for a certain property entry, in this case we look for a "cpu-version" field
        while ((device = IOIteratorNext(objectIterator))) {
          if (kIOReturnSuccess == IORegistryEntryCreateCFProperties(device, &properties, kCFAllocatorDefault, kNilOptions)) {
            if(CFDictionaryGetValueIfPresent(properties, CFSTR("cpu-version"), (const void **)&value)) {
              CFDataGetBytes(value, CFRangeMake(0,4/*CFDataGetLength((void *)value)*/ ), (UInt8 *)&detectedtype);
              //printf("PVR Hi:%04x Lo:%04x\n",(detectedtype>>16)&0xffff,detectedtype&0xffff);
              detectedtype = (detectedtype>>16)&0xffff;
            }
            CFRelease(properties);
          }
          IOObjectRelease(device);
        }
        IOObjectRelease(objectIterator);
      }
      mach_port_deallocate(mach_task_self(), master_port);
    }
  }
  #elif (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN64)
  if (detectedtype == -2L)
  {
    SYSTEM_INFO si;
    si.wProcessorArchitecture = 0;
    si.wProcessorRevision = si.wProcessorLevel = 0;
    detectedtype = -1L;
    GetSystemInfo( &si );
    if (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_PPC)
    {
      detectedtype = si.wProcessorRevision;
      if (detectedtype == 0)
        detectedtype = si.wProcessorLevel;
      if (detectedtype == 0)
        detectedtype = -1;
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
        const char *p = "cpu\t\t: ";
        unsigned int n = strlen( p );
        if ( memcmp( buffer, p, n ) == 0 )
        {
          static struct 
           { const char *sig;  int rid; } sigs[] = {
           /*
           last update Jun 13/2000 from 
           http://lxr.linux.no/source/arch/ppc/kernel/setup.c?v=2.3.99-pre5;a=ppc
           */
           { "601",             0x0001  },
           { "603",             0x0003  },
           { "604",             0x0004  },
           { "603e",            0x0006  },
           { "603ev",           0x0007  },
           { "603r",            0x0007  },
           { "740/750",         0x0008  },
           { "745/755",         0x0008  },
           { "750",             0x0008  },
           { "750CX",           0x0008  },
           { "750CXe",          0x0008  }, /* >= 2.6.11 */
           { "750FX",           0x0008  }, /* PVR coreid is actually 0x7000 */
           { "750GX",           0x0008  }, /* PVR coreid is actually 0x7000 */
           { "750P",            0x0008  },
           { "604e",            0x0009  },
           { "604r",            0x000A  }, /* >= 2.3.99 */
           { "604ev",           0x000A  }, /* < 2.3.34 */
           { "604ev5",          0x000A  }, /* >= 2.3.34 */
           { "7400",            0x000C  },
           { "7400 (1.1)",      0x000C  },
           { "403G",            0x0020  },
           { "403GC",           0x0020  },
           { "403GCX",          0x0020  },
           { "821",             0x0050  },
           { "860",             0x0080  },
           { "8240",            0x0081  },
           { "82xx",            0x0081  },
           { "8280",            0x0082  },
           { "405GP",           0x4011  },
           { "7441",            0x8000  },
           { "7450",            0x8000  },
           { "7451",            0x8000  },
           { "7445",            0x8001  },
           { "7455",            0x8001  },
           { "7447",            0x8002  },
           { "7457",            0x8002  },
           { "7447/7457",       0x8002  },
           { "7410",            0x800C  },
           { "7447A",           0x8003  },
           { "7448",            0x8004  },
           { "PPC970",          0x0039  },
           { "PPC970FX",        0x003C  },
           { "PPC970MP",        0x0044  },
           { "Cell Broadband Engine", 0x0070  }
           };
          p = &buffer[n]; buffer[sizeof(buffer)-1]='\0';
          for ( n = 0; n < (sizeof(sigs)/sizeof(sigs[0])); n++ )
          {
            unsigned int l = strlen( sigs[n].sig );
            if (memcmp( p, sigs[n].sig, l)==0 && (!p[l] || isspace(p[l]) || p[l]==','))
            {
              detectedtype = (long)sigs[n].rid;
              /* 7400, 7410, 7450, 7455 (G4), 970, 970FX (G5),Cell */
              if (detectedtype == 0x000C ||
                  detectedtype == 0x0039 ||
                  detectedtype == 0x003C ||
                  detectedtype == 0x0044 ||
                  detectedtype == 0x0070 ||
                  detectedtype & 0x8000)
              {
                if (memcmp( &p[l], ", altivec supported", 19)==0)
                  isaltivec = 1;
              }
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
  #elif (CLIENT_OS == OS_BEOS)  // BeOS PPC
  if (detectedtype == -2L)
  {
    system_info sInfo;
    sInfo.cpu_type = (cpu_type) 0;
    get_system_info(&sInfo);
    detectedtype = -1;
    if (sInfo.cpu_type) /* didn't fail */
    {
      switch (sInfo.cpu_type)
      {
        case B_CPU_PPC_601:  detectedtype = 0x0001; break;
        case B_CPU_PPC_603:  detectedtype = 0x0003; break;
        case B_CPU_PPC_603e: detectedtype = 0x0006; break;
        case B_CPU_PPC_604:  detectedtype = 0x0004; break;
        case B_CPU_PPC_604e: detectedtype = 0x0009; break;
        case B_CPU_PPC_750:  detectedtype = 0x0008; break;
        default: // some PPC processor that we don't know about
                 // set the tag (so that the user can tell us), but return 0
        sprintf(namebuf, "0x%x", sInfo.cpu_type );
        detectedname = (const char *)&namebuf[0];
        detectedtype = 0;
        break;
      }
    }
  }
  #elif (CLIENT_OS == OS_AMIGAOS)  // AmigaOS PPC
  if (detectedtype == -2L)
  {
    #if defined(__amigaos4__)
    /* AmigaOS 4.x */
    ULONG cpu;
    IExec->GetCPUInfoTags(GCIT_Model, &cpu, TAG_DONE);
    switch (cpu)
    {
       case CPUTYPE_PPC603E:        detectedtype = 0x0006; break;
       case CPUTYPE_PPC604E:        detectedtype = 0x0009; break;
       case CPUTYPE_PPC750CXE:
       case CPUTYPE_PPC750FX:
       case CPUTYPE_PPC750GX:       detectedtype = 0x0008; break;
       case CPUTYPE_PPC7410:        detectedtype = 0x800C; break;
       case CPUTYPE_PPC74XX_VGER:   detectedtype = 0x8000; break;
       case CPUTYPE_PPC74XX_APOLLO: detectedtype = 0x8001; break;
       case CPUTYPE_PPC405LP:       detectedtype = 0x41F1; break;
       case CPUTYPE_PPC405EP:       detectedtype = 0x5121; break;
       case CPUTYPE_PPC405GP:       detectedtype = 0x4011; break;
       case CPUTYPE_PPC405GPR:      detectedtype = 0x5091; break;
       case CPUTYPE_PPC440EP:       detectedtype = 0x4222; break;
       case CPUTYPE_PPC440GP:       detectedtype = 0x4012; break;
       case CPUTYPE_PPC440GX:       detectedtype = 0x51B2; break;
       case CPUTYPE_PPC440SP:       detectedtype = 0x5322; break;
       default: // some PPC processor that we don't know about
                // set the tag (so that the user can tell us), but return 0
       sprintf(namebuf, "OS4:0x%lx", cpu );
       detectedname = (const char *)&namebuf[0];
       detectedtype = 0;
       break;
    }
    #elif !defined(__POWERUP__)
    /* WarpOS */
    struct TagItem cputags[2] = { {GETINFO_CPU, 0}, {TAG_END,0} };
    GetInfo(cputags);
    switch (cputags[0].ti_Data)
    {
      case CPUF_603:  detectedtype = 0x0003; break;
      case CPUF_603E: detectedtype = 0x0006; break;
      case CPUF_604:  detectedtype = 0x0004; break;
      case CPUF_604E: detectedtype = 0x0009; break;
      case CPUF_620:  detectedtype = NONPVR(1); break;
      case CPUF_G3:   detectedtype = 0x0008; break;
      case CPUF_G4:   detectedtype = 0x8000; break;
      default: // some PPC processor that we don't know about
               // set the tag (so that the user can tell us), but return 0
      sprintf(namebuf, "WOS:0x%lx", cputags[0].ti_Data );
      detectedname = (const char *)&namebuf[0];
      detectedtype = 0;
      break;
    }
    #else
    /* PowerUp */
    ULONG cpu = PPCGetAttr(PPCINFOTAG_CPU);
    switch (cpu)
    {
      case CPU_603:  detectedtype = 0x0003; break;
      case CPU_603e: detectedtype = 0x0006; break;
      case CPU_603p: detectedtype = 0x0006; break;
      case CPU_604:  detectedtype = 0x0004; break;
      case CPU_604e: detectedtype = 0x0009; break;
      case 0x0008:   // G3
      case 0x000C:   // 7400 (G4)
      case 0x8000:   // 7450 (G4)
      case 0x8001:   // 7455 (G4)
      case 0x8002:   // 7447/7457 (G4)
      case 0x8003:   // 7447A (G4)
      case 0x8004:   // 7448 (G4)
      case 0x800C:   // 7410 (G4)
      detectedtype = cpu; break;
      default: // some PPC processor that we don't know about
               // set the tag (so that the user can tell us), but return 0
      sprintf(namebuf, "PUP:0x%lx", cpu );
      detectedname = (const char *)&namebuf[0];
      detectedtype = 0;
      break;
    }
    #endif
  }
  #elif (CLIENT_OS == OS_MORPHOS)  // MorphOS
  if (detectedtype == -2L)
  {
    /* MorphOS */
    ULONG cpu = 0;
    NewGetSystemAttrsA(&cpu, sizeof(cpu), SYSTEMINFOTYPE_PPC_CPUVERSION, NULL);

    switch (cpu)
    {
      case 0x0003:   // 603
      case 0x0004:   // 604
      case 0x0006:   // 603e
      case 0x0007:   // 603r/603ev
      case 0x0008:   // 740/750 (G3)
      case 0x0009:   // 604e
      case 0x000A:   // 604ev
      case 0x000C:   // 7400 (G4)
      case 0x8000:   // 7450 (G4)
      case 0x8001:   // 7455 (G4)
      case 0x8002:   // 7457/7447 (G4)
      case 0x8003:   // 7447A (G4)
      case 0x8004:   // 7448 (G4)
      case 0x800C:   // 7410 (G4)
      case 0x0039:   // 970 (G5)
      case 0x003C:   // 970FX (G5)
      case 0x0044:   // 970MP (G5)
      case 0x8081:   // 5200 (G2)
      case 0x8082:   // 5200 (G2-LE)
      detectedtype = cpu; break;
      default: // some PPC processor that we don't know about
               // set the tag (so that the user can tell us), but return 0
      sprintf(namebuf, "MOS:0x%lx", cpu );
      detectedname = (const char *)&namebuf[0];
      detectedtype = 0;
      break;
    }
  }
  #endif
  
  if (detectedtype > 0 && detectedname == NULL) {
    unsigned int n;
    detectedname = "";

    for (n = 0; n < (sizeof(cpuridtable)/sizeof(cpuridtable[0])); n++) {
      if (cpuridtable[n].rid == detectedtype) {
        if (detectedtype != 0x0070) { /* without Cell */
          strcpy(namebuf, "PowerPC ");
        }
        strcat(namebuf, cpuridtable[n].name);
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }
  
  if (cpuname)
    *cpuname = detectedname;

  if (detectedtype > 0 && isaltivec) {
    /* *OS* supports altivec */
    detectedtype |= (1L << 25);
  }

  return detectedtype;
}

# undef NONPVR
#endif /* (CLIENT_CPU == CPU_POWERPC) */

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_POWER)
static long __GetRawProcessorID(const char **cpuname) {
  /* -1 == failed, -2 == not supported */
  static long detectedtype = -2L;
  static const char *detectedname = NULL;
  static char namebuf[30];
  struct { long id;
           const char *name; }
    cpuidtable[] = {
      { 1, "RS"            },
      { 2, "RS2 Superchip" },
      { 3, "RS2"           }};

# if (CLIENT_OS == OS_AIX)
  if (detectedtype == -2L) {
    if (_system_configuration.architecture == POWER_RS) {
      unsigned int i;
      struct { long imp;
               long id; }
        cpumap[] = {
          { POWER_RS1, 1 },
          { POWER_RSC, 2 },
          { POWER_RS2, 3 }};

      /* assume failed */
      detectedtype = -1L;

      for (i = 0; i < (sizeof(cpumap)/sizeof(cpumap[0])); i++) {
        if (cpumap[i].imp == _system_configuration.implementation ) {
          detectedtype = cpumap[i].id;
          break;
        }
      }

      if (detectedtype == -1L) {
        /* identification failed */
        sprintf( namebuf, "impl:0x%X", _system_configuration.implementation );
        detectedname = (const char *)&namebuf[0];
      }
    } else {
      /* just signal that we've got a PowerPC CPU here */
      detectedtype = (1 << 24);
      detectedname = "PowerPC";
    }

    detectedtype = 3;
    detectedname = NULL;
  }
# endif

  if (detectedtype > 0 && detectedname == NULL) {
    unsigned int n;
    detectedname = "";

    for (n = 0; n < (sizeof(cpuidtable)/sizeof(cpuidtable[0])); n++) {
      if (cpuidtable[n].id == detectedtype) {
        strcpy(namebuf, "Power ");
        strcat(namebuf, cpuidtable[n].name);
        detectedname = (const char *)&namebuf[0];
        break;
      }
    }
  }

  if (cpuname)
    *cpuname = detectedname;

  return detectedtype;
}
#endif /* CLIENT_CPU == CPU_POWER */

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
// whattoret 0:detectedtype, 'c':simpleid, 'f':featureflags
long __GetRawProcessorID(const char **cpuname, int whattoret = 0 )
{
  static long detectedtype = -2L;  /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static int simpleid = 0xff;      /* default id if not found */
  static int featureflags = 0;     /* default flags if id not found */
  
  if ( detectedtype == -2L )
  {
    static char namebuf[80];
    const char *vendorname = NULL;
    int vendorid; u32 dettype; 
    struct cpuxref { u32 cpuid, mask, cpufeatures, simpleid;
                     const char *cpuname; } *internalxref = NULL;

    #if (CLIENT_OS == OS_WIN32) || (CLIENT_OS == OS_WIN16)
    if (winGetVersion() < 2000) /* win95 permits inb()/outb() */
      x86ident_haveioperm = 1;
    #elif (CLIENT_OS == OS_DOS) || (CLIENT_OS == OS_NETWARE) 
      x86ident_haveioperm = 1;        /* netware client runs at IOPL 0 */
    #endif

    dettype    = x86GetDetectedType();
    vendorid   = ID_VENDOR_CODE(dettype);
    vendorname = x86GetVendorName(dettype);

    sprintf( namebuf, "%08X", dettype );
    detectedname = (const char *)&namebuf[0];

    if (dettype)
      detectedtype = 0; /* allow tag to be shown */
    else
      detectedtype = -1; /* assume not found */
    simpleid = 0xff; /* default id = unknown */
    
    if ( vendorid == VENDOR_NSC )
    {
      static struct cpuxref natsemi_xref[]={
        { 0x5040, 0xFFF0, CPU_F_I586, 0x10, "Geode"     },
        { 0x0000,      0,          0,    0, NULL        }
      }; internalxref = &natsemi_xref[0];
    }
    else if ( vendorid == VENDOR_TRANSMETA )
    {
      static struct cpuxref transmeta_xref[]={
        { 0x5042, 0xFFFF, CPU_F_I586,    0, "Crusoe TM3x00" },
        { 0x5043, 0xFFFF, CPU_F_I586,    0, "Crusoe TM5x00" },
        { 0xF000, 0xF000, CPU_F_I586, 0xFF, "Efficeon TM8000/10000" },
        { 0x0000,      0,          0,    0, NULL }
      }; internalxref = &transmeta_xref[0];
    }
    else if ( vendorid == VENDOR_SIS )
    { 
      static struct cpuxref sis_xref[]={
        { 0x5000, 0xFFF0, CPU_F_I586,  0xFF, "55x" },
        { 0x0000,      0,          0,     0, NULL  }  
      }; internalxref = &sis_xref[0];
    }
    else if ( vendorid == VENDOR_CYRIX )
    {
      static struct cpuxref cyrixxref[]={
        { 0x4000, 0xFFF0, CPU_F_I486,    6, "486SLC/DLC/SR/DR"  },
        { 0x4010, 0xFFF0, CPU_F_I486,    6, "486S/Se/S2/DX/DX2" },
        { 0x4020, 0xFFF0, CPU_F_I586,    3, "5x86"              },
        { 0x4040, 0xFFF0, CPU_F_I586,    3, "MediaGX"           },
        { 0x4090, 0xFFF0, CPU_F_I586,    3, "5x86"              },
        { 0x5020, 0xFFF0, CPU_F_I586, 0x10, "6x86"              },
        { 0x5030, 0xFFF0, CPU_F_I586, 0x10, "6x86 M1"           },
        { 0x5040, 0xFFF0, CPU_F_I586, 0x10, "MediaGX MMX/GXm"   },
        { 0x6000, 0xFFF0, CPU_F_I586, 0x11, "6x86MX/M II"       },
        { 0x6050, 0xFFF0, CPU_F_I586, 0xFF, "III Joshua"        },
        { 0x0000,      0,          0,    0, NULL                }
      }; internalxref = &cyrixxref[0]; 
    }
    else if ( vendorid == VENDOR_CENTAUR )
    {
      static struct cpuxref centaurxref[]={
        {  0x5040, 0xFFF0, CPU_F_I586, 0x0A, "WinChip C6 / Centaur C6" },
        {  0x5080, 0xFFF0, CPU_F_I586, 0x0A, "WinChip 2 / Centaur C2" },
        {  0x5090, 0xFFF0, CPU_F_I586, 0x0A, "WinChip 3 / Centaur C3" },
        {  0x6050, 0xFFF0, CPU_F_I586, 0x0C, "6x86MX/MII" },
        {  0x6060, 0xFFF0, CPU_F_I586, 0x0C, "C3 (Samuel)" }, /* THIS IS NOT A P6 !!! */
        {  0x6070, 0xFFF0, CPU_F_I586, 0x0C, "C3 (Samuel 2) / Eden ESP (Ezra)" },
        {  0x6080, 0xFFF0, CPU_F_I586, 0x0C, "C3 (Ezra-T)" },
        {  0x6090, 0xFFF0, CPU_F_I586, 0x0F, "C3 / C3-M (Nehemiah)" },
        {  0x60A0, 0xFFF0, CPU_F_I586, 0x0A, "C7 (Esther)" },
        {  0x0000,      0,          0,    0, NULL }
      }; internalxref = &centaurxref[0];
    }
    else if ( vendorid == VENDOR_RISE )
    {
      static struct cpuxref risexref[]={
        { 0x5000, 0xFF80, CPU_F_I586, 0xFF, "mP6 iDragon" },
        { 0x5080, 0xFF80, CPU_F_I586, 0xFF, "mP6 iDragon II" },
        { 0x0000,      0,          0,    0, NULL  }
      }; internalxref = &risexref[0];
    }
    else if ( vendorid == VENDOR_NEXGEN )
    {   
      static struct cpuxref nexgenxref[]={
        { 0x5000, 0xFFF0, CPU_F_I586,   1, "Nx586" }, //386/486 core  /* I586 is a guess */
        { 0x0000,      0,          0,   0, NULL    }
      }; internalxref = &nexgenxref[0];
    }
    else if ( vendorid == VENDOR_UMC )
    {   
      static struct cpuxref umcxref[]={
        { 0x4010, 0xFFF0, CPU_F_I486,   0, "U5D" },
        { 0x4020, 0xFFF0, CPU_F_I486,   0, "U5S" },
        { 0x0000,      0,          0,   0, NULL  }
      }; internalxref = &umcxref[0];
    }
    else if ( vendorid == VENDOR_AMD )
    {
      /* see "AMD Processor Recognition Application Note" available at
         http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/20734.pdf 
         http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/25759.pdf 
         http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/41788.pdf 
         http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/25481.pdf
         http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/33610.pdf
         http://www.amd.com/us-en/assets/content_type/white_papers_and_tech_docs/41788.pdf
       */
      static struct cpuxref amdxref[]={
        { 0x0004000, 0xFFFFFF0, CPU_F_I486,    0, "486"      },
        { 0x0004030, 0xFFFFFF0, CPU_F_I486,    0, "486DX2"   },
        { 0x0004070, 0xFFFFFF0, CPU_F_I486,    0, "486DX2WB" },
        { 0x0004080, 0xFFFFFF0, CPU_F_I486,    0, "486DX4"   },
        { 0x0004090, 0xFFFFFF0, CPU_F_I486,    0, "486DX4WB" },
        { 0x00040E0, 0xFFFFFF0, CPU_F_I486,    6, "5x86"     },
        { 0x00040F0, 0xFFFFFF0, CPU_F_I486,    6, "5x86WB"   },
        { 0x0005000, 0xFFFFFF0, CPU_F_I586,    4, "SSA5 PR75, PR90, or PR100" }, // use K5 core
        { 0x0005010, 0xFFFFFF0, CPU_F_I586,    4, "5k86 PR120 or PR133" },
        { 0x0005020, 0xFFFFFF0, CPU_F_I586,    4, "5k86 PR166" },
        { 0x0005030, 0xFFFFFF0, CPU_F_I586,    4, "5k86 PR200" },
        { 0x0005060, 0xFFFFFF0, CPU_F_I586,    5, "K6"              },
        { 0x0005070, 0xFFFFFF0, CPU_F_I586,    5, "K6"              },
        { 0x0005080, 0xFFFFFF0, CPU_F_I586,    5, "K6-2 (Chomper)"  },
        { 0x0005090, 0xFFFFFF0, CPU_F_I586,    5, "K6-III (SharpTooth)" },
        { 0x00050D0, 0xFFFFFF0, CPU_F_I586,    5, "K6-2+/K6-III+" },
        { 0x0006010, 0xFFFFFF0, CPU_F_I686,    9, "Athlon"        }, // slot A
        { 0x0006020, 0xFFFFFF0, CPU_F_I686,    9, "Athlon"        }, // slot A
        { 0x0006030, 0xFFFFFF0, CPU_F_I686,    9, "Duron (Spitfire)"  }, // 64K L2
        { 0x0006040, 0xFFFFFF0, CPU_F_I686,    9, "Athlon (Thunderbird)" }, // 256K L2
        { 0x0006060, 0xFFFFFF0, CPU_F_I686,    9, "K7-6 (Athlon XP/MP/4 or Duron)" }, // Palomino core, 256K L2
        { 0x0006070, 0xFFFFFF0, CPU_F_I686,    9, "Duron (Morgan)"  }, // Morgan core = Palomino core w/ 64K L2
        { 0x0006080, 0xFFFFFF0, CPU_F_I686,    9, "Athlon XP/MP or Sempron (Thoroughbred)" },
        { 0x00060A0, 0xFFFFFF0, CPU_F_I686,    9, "Athlon XP/MP/XP-M or Sempron (Barton)" },   // OGR-NG: OK (-k8)
        { 0x000F000, 0xFFFF000, CPU_F_I686,    9, "Athlon (Model 15)" },
        { 0x010F000, 0xFFFF000, CPU_F_I686,    9, "Athlon 64" },
        { 0x020F000, 0xFFFF000, CPU_F_I686,    9, "Athlon 64 X2 Dual Core" },
        { 0x030F000, 0xFFFF000, CPU_F_I686,    9, "Mobile Athlon 64" },
        { 0x040F000, 0xFFFF000, CPU_F_I686,    9, "Turion 64 Mobile Technology" },
        { 0x050F000, 0xFFFF000, CPU_F_I686,    9, "Opteron" },
        { 0x060F000, 0xFFFF000, CPU_F_I686,    9, "Athlon XP-M" },
        { 0x070F000, 0xFFFF000, CPU_F_I686,    9, "Athlon XP" },
        { 0x080F000, 0xFFFF000, CPU_F_I686,    9, "Mobile Sempron" },
        { 0x090F000, 0xFFFF000, CPU_F_I686,    9, "Sempron" },
        { 0x0A0F000, 0xFFFF000, CPU_F_I686,    9, "Athlon 64 FX" },
        { 0x0B0F000, 0xFFFF000, CPU_F_I686,    9, "Dual Core Opteron" },
        { 0x0C0F000, 0xFFFF000, CPU_F_I686,    9, "Turion 64 X2 Mobile Technology" },
        /* class 0x16 - similar to Intel Core CPUs, with fast SSE (OGR-NG SSE2 core) */
        { 0x0D10000, 0xFFFF000, CPU_F_I686, 0x16, "Athlon (Model 16)" }, /* (#4120,#4196) */
        { 0x0E10000, 0xFFFF000, CPU_F_I686,    9, "Dual Core Opteron" },
        { 0x0F10000, 0xFFFF000, CPU_F_I686, 0x16, "Quad Core Opteron" },
        { 0x1010000, 0xFFFF000, CPU_F_I686,    9, "Embedded Opteron" },
        { 0x1110000, 0xFFFF000, CPU_F_I686,    9, "Phenom" },
        { 0x1211000, 0xFFFF000, CPU_F_I686,    9, "Athlon (Model 17)" }, /* (#4074) */
        { 0x1311000, 0xFFFF000, CPU_F_I686,    9, "Sempron" },
        { 0x1411000, 0xFFFF000, CPU_F_I686,    9, "Turion X2 Ultra Mobile" },
        { 0x1511000, 0xFFFF000, CPU_F_I686,    9, "Turion X2 Mobile" },
        { 0x1611000, 0xFFFF000, CPU_F_I686,    9, "Athlon X2" },
        { 0x0000000,         0,          0,    0, NULL       }
      }; internalxref = &amdxref[0];
      if ((dettype & 0xFFFFFF0) == 0x0400)        /* no such AMD ident */
        vendorname = "Intel/AMD";                 /* identifies AMD or Intel 486 */
    }
    else if ( vendorid == VENDOR_INTEL )
    {
      /* the following information has been collected from the 
         "AP-485 Intel Processor Identification and the CPUID Instruction"
         manual available at 
         http://www.intel.com/design/xeon/applnots/241618.htm
         and several "Intel XYZ Processor Specification Update" documents
         available from http://www.intel.com/design/processor/index.htm */
      static struct cpuxref intelxref[]={
        { 0x0003000, 0xFFFFFF0, CPU_F_I386,    1, "386SX/DX" },
        { 0x0004000, 0xFFFFFF0, CPU_F_I486,    1, "486DX 25 or 33" },
        { 0x0004010, 0xFFFFFF0, CPU_F_I486,    1, "486DX 50" },
        { 0x0004020, 0xFFFFFF0, CPU_F_I486,    1, "486SX" },
        { 0x0004030, 0xFFFFFF0, CPU_F_I486,    1, "486DX2" },
        { 0x0004040, 0xFFFFFF0, CPU_F_I486,    1, "486SL" },
        { 0x0004050, 0xFFFFFF0, CPU_F_I486,    1, "486SX2" },
        { 0x0004070, 0xFFFFFF0, CPU_F_I486,    1, "486DX2WB" },
        { 0x0004080, 0xFFFFFF0, CPU_F_I486,    1, "486DX4" },
        { 0x0004090, 0xFFFFFF0, CPU_F_I486,    1, "486DX4WB" },
        { 0x0005000, 0xFFFFFF0, CPU_F_I586,    0, "Pentium A-step" },
        { 0x0005010, 0xFFFFFF0, CPU_F_I586,    0, "Pentium" },
        { 0x0005020, 0xFFFFFF0, CPU_F_I586,    0, "Pentium P54C" },
        { 0x0005030, 0xFFFFFF0, CPU_F_I586,    0, "Pentium Overdrive" },
        { 0x0005045, 0xFFFFFFF, CPU_F_I586,    0, "Pentium P55C (buggy-MMX)" }, /* MMX core crash - #2204 */
        { 0x0005040, 0xFFFFFF0, CPU_F_I586,    0, "Pentium MMX P55C" },
        { 0x0005070, 0xFFFFFF0, CPU_F_I586,    0, "Pentium MMX P54C" },
        { 0x0005080, 0xFFFFFF0, CPU_F_I586,    0, "Pentium MMX P55C" },
        { 0x0006000, 0xFFFFFF0, CPU_F_I686,    8, "Pentium Pro A-step" },
        { 0x0006010, 0xFFFFFF0, CPU_F_I686,    8, "Pentium Pro" },
        { 0x0006030, 0xFFFFFF0, CPU_F_I686,    2, "Pentium II (Klamath)" },
        { 0x0006050, 0xFFFFFF0, CPU_F_I686,    2, "Pentium II PE (Deschutes)" },
        { 0x0106050, 0xFFFFFF0, CPU_F_I686,    2, "Celeron (Covington)" },
        { 0x0206050, 0xFFFFFF0, CPU_F_I686,    2, "Celeron-A (Mendocino)" },
        { 0x0306050, 0xFFFFFF0, CPU_F_I686,    2, "Pentium II/Xeon (Deschutes)" },
        { 0x0406050, 0xFFFFFF0, CPU_F_I686,    2, "Pentium II Xeon (Deschutes)" },
        { 0x0006060, 0xFFFFFF0, CPU_F_I686,    2, "Pentium II (Mendocino)" },
        { 0x0106060, 0xFFFFFF0, CPU_F_I686,    2, "Celeron-A (Mendocino/Dixon)" },
        { 0x0006070, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III (Katmai)" },
        { 0x0106070, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III Xeon (Katmai)" },
        /* Itanium IA-64 */
        { 0x0007000, 0x00FF000, CPU_F_I686,    5, "Itanium" },
        { 0x0017000, 0x00FF000, CPU_F_I686, 0xFF, "Itanium II (McKinley/Madison)" },
        { 0x0027000, 0x00FF000, CPU_F_I686, 0xFF, "Itanium II DC (Montecito)" },
        /* The following CPUs have a BrandID field from Intel */
        /* Coppermine - 0.18u */
        { 0x0106080, 0xFFFFFF0, CPU_F_I686, 0x0E, "Celeron (Coppermine)" },
        { 0x0206080, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III (Coppermine)" },
        { 0x0306080, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III Xeon (Coppermine)" },
        /* Banias - 0.13u */
        { 0x0406090, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium M (Banias)" },
        { 0x0606090, 0xFFFFFF0, CPU_F_I686, 0x0D, "Pentium M (Banias)" },
        { 0x1606090, 0xFFFFFF0, CPU_F_I686, 0x0D, "Pentium M (Banias)" }, /* (#4075) */
        { 0x0706090, 0xFFFFFF0, CPU_F_I686, 0x0E, "Celeron M (Banias)" },
        /* Cascades - 0.18u */
        { 0x01060A0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Celeron (Cascades)" },
        { 0x02060A0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III (Cascades)" },
        { 0x03060A0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III Xeon (Cascades)" },
        /* Tualatin - 0.13u */
	{ 0x02060B0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III (Tualatin)" }, /* (#4121) */
        { 0x03060B0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Celeron (Tualatin)" },
        { 0x04060B0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III (Tualatin)" },
        { 0x06060B0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Pentium III M (Tualatin)" },
        { 0x07060B0, 0xFFFFFF0, CPU_F_I686, 0x0E, "Celeron M (Tualatin)" },
        /* Dothan - 0.09u */
        { 0x12060D0, 0xFFFFFF0, CPU_F_I686, 0x0D, "Celeron M (Dothan)" },
        { 0x16060D0, 0xFFFFFF0, CPU_F_I686, 0x0D, "Pentium M (Dothan)" },
        /* Some P4-class CPUs wants RC5-72 core #7,
           they're marked as type 13 for OGR-NG core -k8
                         and type 17 for OGR-NG core -p4
        */
        /* Pentium 4 models 0/1 : 180 nm */
        { 0x080F000, 0xFFFFFF0, CPU_F_I686, 0x13, "Pentium 4 (Willamette)" },
        { 0x0E0F000, 0xFFFFFF0, CPU_F_I686, 0x0B, "Xeon (Foster)" },
        { 0x080F010, 0xFFFFFF0, CPU_F_I686, 0x13, "Pentium 4 (Willamette)" },
        { 0x0A0F010, 0xFFFFFF0, CPU_F_I686, 0x0B, "Celeron 4 (Willamette)" },
        { 0x0B0F010, 0xFFFFFF0, CPU_F_I686, 0x0B, "Xeon (Foster)" },
        { 0x0E0F013, 0xFFFFFFF, CPU_F_I686, 0x0B, "Pentium 4 M" },
        { 0x0E0F010, 0xFFFFFF0, CPU_F_I686, 0x0B, "Xeon (Foster)" },
        /* Pentium 4 model 2 : 130 nm */
        { 0x080F020, 0xFFFFFF0, CPU_F_I686, 0x0B, "Pentium 4 (Northwood)" },
        /* (#4177) conflict: 2.0GHz 1090F027 need OGR-NG -p4, 2.8Ghz 1090F029 need -k8 */
        { 0x090F027, 0xFFFFFFF, CPU_F_I686, 0x17, "Pentium 4 (Northwood)" },
        { 0x090F020, 0xFFFFFF0, CPU_F_I686, 0x13, "Pentium 4 (Northwood)" },
        { 0x0A0F020, 0xFFFFFF0, CPU_F_I686, 0x13, "Celeron 4 (Northwood)" },
        { 0x0B0F020, 0xFFFFFF0, CPU_F_I686, 0x17, "Xeon (Prestonia)" },  /* (#4186) */
        { 0x0C0F020, 0xFFFFFF0, CPU_F_I686, 0x0B, "Xeon MP (Gallatin)" },
        { 0x0E0F020, 0xFFFFFF0, CPU_F_I686, 0x0B, "Mobile Pentium 4-M (Northwood)" },
        { 0x0F0F020, 0xFFFFFF0, CPU_F_I686, 0x0B, "Mobile Celeron 4 (Northwood)" },
        /* The following CPUs no longer have a BrandID field from Intel */
        /* Pentium 4 models 3/4 : 90 nm */
        { 0x000F030, 0x00FFFF0, CPU_F_I686, 0x0B, "Pentium 4/D/4-M/Celeron/Xeon" },
        { 0x000F040, 0x00FFFF0, CPU_F_I686, 0x0B, "Pentium 4/D/4-M/Celeron/Xeon" },
        /* Pentium 4 model 6 :  65 nm */
        { 0x000F060, 0x00FFFF0, CPU_F_I686, 0x0B, "Pentium 4/D/4-M/Celeron/Xeon" },
        { 0x00060E0, 0x00FFFF0, CPU_F_I686, 0x0D, "Core" },
        { 0x00060F0, 0x00FFFF0, CPU_F_I686, 0x12, "Core 2/Xeon" },
        { 0x0006160, 0x00FFFF0, CPU_F_I686, 0xFF, "Celeron" },              /* 65 nm */
        { 0x0006170, 0xFFFFFF0, CPU_F_I686, 0x12, "Core 2/Extreme/Xeon" },  /* 45 nm */
        { 0x00061A0, 0xFFFFFF0, CPU_F_I686, 0x15, "Core i7" },  /* (#4118) */
        { 0x00061C0, 0xFFFFFF0, CPU_F_I686, 0x14, "Atom" },  /* (#4080) */
        { 0x0000000,         0,          0,    0, NULL }
      }; internalxref = &intelxref[0];
    }

    if (internalxref != NULL) /* we know about this vendor */
    {
      unsigned int pos;
  
      for (pos = 0; internalxref[pos].cpuname; pos++ )
      {
        if ((dettype & internalxref[pos].mask) == internalxref[pos].cpuid) /* found it */
        {
          simpleid     = internalxref[pos].simpleid;
          featureflags = internalxref[pos].cpufeatures;
          detectedtype = dettype;
          if ( internalxref[pos].cpuname )
          {
            strcpy( namebuf, vendorname );
            if (namebuf[0])
              strcat( namebuf, " ");

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
  else if (whattoret == 'f')
    return ((long)featureflags);
  return detectedtype;
}
#endif  /* X86 */

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
signed int default_rc5_core = -1;
signed int default_r72_core = -1;
signed int default_ogr_core = -1;

static long __GetRawProcessorID(const char **cpuname )
{
  static long detectedtype = -2L; /* -1 == failed, -2 == not supported */
  static const char *detectedname = NULL;
  static char namebuf[65];
  static struct {
    unsigned int id, mask;
    signed int r72, ogr;
    const char *name;
  } ids[] = {
    // ARM
    { 0x41560200, 0xfffffff0, 1, 1, "ARM 2" },
    { 0x41560250, 0xfffffff0, 1, 1, "ARM 250" },
    { 0x41560300, 0xfffffff0, 1, 1, "ARM 3" },
    { 0x41560600, 0xfffffff0, 1, 1, "ARM 600" },
    { 0x41560610, 0xfffffff0, 1, 1, "ARM 610" },
    { 0x41560200, 0xfffffff0, 1, 1, "ARM 620" },
    { 0x41007000, 0xfffffff0, 1, 1, "ARM 700" },
    { 0x41007100, 0xfffffff0, 1, 1, "ARM 710" },
    { 0x41007500, 0xffffffff, 1, 1, "ARM 7500" },  // 7500/FEL are "artificial" ids
    { 0x410F7500, 0xffffffff, 1, 1, "ARM 7500FEL" },  // created by IOMD detection
    { 0x41047100, 0xfffffff0, 1, 1, "ARM 7100" },
    { 0x41807100, 0xfffffff0, 1, 1, "ARM 710T" },
    { 0x41807200, 0xfffffff0, 1, 1, "ARM 720T" },
    { 0x41807400, 0xfffffff0, 1, 1, "ARM 740T8K" },
    { 0x41817400, 0xfffffff0, 1, 1, "ARM 740T4K" },
    { 0x41018100, 0xfffffff0, 0, 1, "ARM 810" },
    { 0x41129200, 0xfffffff0, 0, 1, "ARM 920T" },
    { 0x41029220, 0xfffffff0, 0, 1, "ARM 922T" },
    { 0x41009260, 0xff00fff0, 0, 1, "ARM 926" },
    { 0x41029400, 0xfffffff0, 0, 1, "ARM 940T" },
    { 0x41049460, 0xfffffff0, 0, 1, "ARM 946ES" },
    { 0x41049660, 0xfffffff0, 0, 1, "ARM 966ES" },
    { 0x41059660, 0xfffffff0, 0, 1, "ARM 966ESR" },
    { 0x4100a200, 0xff00fff0, 0, 1, "ARM 1020" },
    { 0x4100a260, 0xff00fff0, 0, 1, "ARM 1026" },
    // ?
    { 0x54029150, 0xfffffff0, 0, 1, "ARM 915" },
    { 0x54029250, 0xfffffff0, 0, 1, "ARM 925" },
    // Digital
    { 0x4401a100, 0xfffffff0, 0, 1, "Digital StrongARM 110" },
    { 0x4401a110, 0xfffffff0, 0, 1, "Digital StrongARM 1100" },
    // Intel
    { 0x6901b110, 0xfffffff0, 0, 1, "Intel StrongARM 1110" },
    { 0x69052120, 0xfffff3f0, 2, 2, "Intel PXA210" },
    { 0x69052100, 0xfffff7f0, 2, 2, "Intel PXA250" },
    { 0x69052d00, 0xfffffff0, 2, 2, "Intel PXA255" },
    { 0x69054110, 0xfffffff0, 2, 2, "Intel PXA270" },
    { 0x69052000, 0xfffffff0, 2, 2, "Intel 80200" },
    { 0x69052e20, 0xffffffe0, 2, 2, "Intel 80219" },
    { 0x69052c20, 0xffffffe0, 2, 2, "Intel IOP321" },
    { 0x00000000, 0x00000000, -1, -1, "" }
  };

  #if (CLIENT_OS == OS_RISCOS)
  if ( detectedtype == -2L )
  {
    detectedtype = ARMident();
    sprintf(namebuf, "%0lX", detectedtype);
  }
  #elif (CLIENT_OS == OS_LINUX)
  if (detectedtype == -2)
  {
    char buffer[256];
    unsigned int i, n, o;
    unsigned int id_temp, id_bits;
    FILE *cpuinfo;

    namebuf[0]='\0';
    o=0;
    id_temp=0;
    id_bits=0;
    
    if ((cpuinfo = fopen( "/proc/cpuinfo", "r")) != NULL)
    {
      while (fgets(buffer, sizeof(buffer), cpuinfo))
      {
        if (memcmp(buffer, "Type\t\t: ", 8) == 0)
          o=8;
        if (memcmp(buffer, "Processor\t: ", 12) == 0)
          o=12;
        if (o!=0)
        {
          n=strlen(buffer)-o-1;
          if (n > (sizeof(namebuf)-1))
            n=sizeof(namebuf)-1;
          for (i=0; i<n; i++)
            namebuf[i]=tolower(buffer[i+o]);
          namebuf[n]='\0';
          o=0;
        }
        if (memcmp(buffer, "CPU implementer\t: 0x", 20) == 0)
        {
          sscanf(&buffer[20], "%x", &n);
          id_bits|=0xff000000;
          id_temp|=(n&0xff)<<24;
        }
        if (memcmp(buffer, "CPU variant\t: 0x", 16) == 0)
        {
          sscanf(&buffer[16], "%x", &n);
          id_bits|=0x00f00000;
          id_temp|=(n&0xf)<<20;
        }
        if (memcmp(buffer, "CPU part\t: 0x", 13) == 0)
        {
          sscanf(&buffer[13], "%x", &n);
          id_bits|=0x0000fff0;
          id_temp|=(n&0xfff)<<4;
        }
        if (memcmp(buffer, "CPU revision\t: ", 15) == 0)
        {
          sscanf(&buffer[15], "%d", &n);
          id_bits|=0x0000000f;
          id_temp|=n&0xf;
        }
        if (memcmp(buffer, "CPU architecture: ", 18) == 0)
        {
          n=0;
          if (buffer[18] == '4')
          {
            if (buffer[19] == 'T')
              n=2;
            else
              n=1;
          }
          if (buffer[18] == '5')
          {
            if (buffer[19] == 'T')
            {
              if (buffer[20] == 'E')
              {
                if (buffer[21] == 'J')
                  n=6;
                else
                  n=5;
              }
              else
                n=4;
            }
            else
              n=3;
          }
          if (buffer[18] == '6')
            n=7;

          id_bits|=0x000f0000;
          id_temp|=(n&0xf)<<16;
        }
      }
      fclose(cpuinfo);
    }
    
    if (id_bits == 0xffffffff)
    {
      detectedtype=id_temp;
    }
    else if (namebuf[0])
    {
      static struct { const char *sig;  int id; } sigs[] ={
                    { "arm2",           0x41560200},
                    { "arm 2",          0x41560200},
                    { "arm250",         0x41560250},
                    { "arm 250",        0x41560250},
                    { "arm3",           0x41560300},
                    { "arm 3",          0x41560300},
                    { "arm610",         0x41560610},
                    { "arm 610",        0x41560610},
                    { "arm6",           0x41560600},
                    { "arm 6",          0x41560600},
                    { "arm710",         0x41007100},
                    { "arm 710",        0x41007100},
                    { "arm720t",        0x41807200},
                    { "arm7",           0x41007000},
                    { "arm 7",          0x41007000},
                    { "arm920t",        0x41129200},
                    { "arm922t",        0x41029220},
                    { "arm926",         0x41009260},
                    { "arm915t",        0x54029150},
                    { "arm925t",        0x54029150},
                    { "arm1020",        0x4100a200},
                    { "arm1026",        0x4100a260},
                    { "sa110",          0x4401a100},
                    { "strongarm-110",  0x4401a100},
                    { "strongarm-1100", 0x4401a110},
                    { "strongarm-1110", 0x6901b110},
                    { "80200",          0x69052000},
                    { "80219",          0x69052e30},
                    { "iop80321",       0x69052c30},
                    { "pxa210",         0x69052120},
                    { "pxa250",         0x69052100},
                    { "pxa255",         0x69052d00},
                    { "pxa270",         0x69054110},
                    { "",               0x00000000}
                    };

      for ( i=0; detectedtype==-2; i++ )
      {
        if (strstr(namebuf, sigs[i].sig) != NULL)
          detectedtype=sigs[i].id;
      }
    }
  }
  #elif (CLIENT_OS == OS_NETBSD)
  if (detectedtype == -2)
  {
    char buffer[256]; int mib[2];
    size_t len = (size_t)(sizeof(buffer)-1);
    mib[0]=CTL_HW; mib[1]=HW_MODEL;
    detectedtype = -1;

    if (sysctl(mib, 2, &buffer[0], &len, NULL, 0 ) != 0)
    {
      buffer[0] = '\0';
    }
    else if (len > 0)
    {
      buffer[len] = '\0';
      len = 0;
      while (buffer[len]=='-' || isalpha(buffer[len]) || isdigit(buffer[len]))
        len++;
      buffer[len] = '\0';
    }
    if (buffer[0]) /* sysctl() succeeded and name if clean */
    {
      static struct
        { const char *sig;  int rid; } sigs[] ={
        { "ARM2",       0X200},
        { "ARM250",     0X250},
        { "ARM3",       0X300},
        { "ARM6",       0X600},
        { "ARM610",     0X610},
        { "ARM7",       0X700},
        { "ARM710",     0X710},
        { "SA-110",     0XA10}
        };
      unsigned int n;
      detectedtype = 0;
      for ( n = 0; n < (sizeof(sigs)/sizeof(sigs[0])); n++ )
      {
        if (strcmp(buffer,sigs[n].sig)==0)
        {
          detectedtype = (long)sigs[n].rid;
          break;
        }
      }
      if (detectedtype == 0) /* unrecognized name */
      {
        strncpy( namebuf, buffer, sizeof(namebuf) );
        namebuf[sizeof(namebuf)-1] = '\0';
        detectedname = ((const char *)&(namebuf[0]));
      }
    } /* if (len > 0) */
  } /* if (detectedtype == -2) */
  #endif

  if (detectedtype != -2)
  {
    for (int n=0; detectedname==NULL; n++)
    {
      if ((detectedtype & ids[n].mask) == (ids[n].id & ids[n].mask))
      {
        if (ids[n].id == 0)
        {
          char tempbuf[sizeof(namebuf)];
          
          strncpy(tempbuf, namebuf, sizeof(tempbuf));
          snprintf(namebuf, sizeof(namebuf), "%s,\nid 0x%08x", tempbuf, detectedtype);
          detectedname = namebuf;
          detectedtype = 0;
        }
        else
        {
          detectedname = ids[n].name;
        }
        default_r72_core = ids[n].r72;
        default_ogr_core = ids[n].ogr;
      }
    }
  }

  if ( cpuname )
    *cpuname = detectedname;
  return detectedtype;
}
#endif

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_MIPS)
static long __GetRawProcessorID(const char **cpuname)
{
  static const int ridPS2 = 99;      /* Please set a same rid of R5900 */

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
                { "R5900"         ,      99  },   /* ridPS2 */
                { "R6000"         ,      17  },
                { "R6000A"        ,      18  },
                { "R8000"         ,      19  },
                { "R10000"        ,      20  }
                };
  
  #if (CLIENT_OS == OS_LINUX) || (CLIENT_OS == OS_PS2LINUX)

  /*  CPU detect algorithm was changed:  2002-05-31 by jt@distributed.net  /
  /   to suport R5900MM variants(PlayStation 2)                            /
  /   SCPH-10000: R5900 V1.4   laters: R5900 V2.0                          /
  /   Then I changed to detect space or null code.                        */

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
            unsigned int l = strlen( cpuridtable[n].name );
            if ((!p[l] || isspace(p[l])) && memcmp( p, cpuridtable[n].name, l)==0)
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
        if (detectedtype == ridPS2)
          strcpy( namebuf, "MIPS R5900MM(PS2 Emotion Engine)" );
        else
        {
          strcpy( namebuf, "MIPS " );
          strcat( namebuf, cpuridtable[n].name );
        }
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
  
#if (CLIENT_OS != OS_SOLARIS)
  /* from linux kernel - last synced with kernel 2.4.19 */
  static struct { int psr_impl, psr_vers; const char *name; } cpuridtable[] = {
  /* from linux/arch/sparc/kernel/cpu.c */
  /* Sun4/100, 4/200, SLC */
  { 0, 0, "Fujitsu  MB86900/1A or LSI L64831 SparcKIT-40"},
  /* borned STP1012PGA */
  { 0, 4, "Fujitsu  MB86904"},
  { 0, 5, "Fujitsu TurboSparc MB86907"},
  /* SparcStation2, SparcServer 490 & 690 */
  { 1, 0, "LSI Logic Corporation - L64811"},
  /* SparcStation2 */
  { 1, 1, "Cypress/ROSS CY7C601"},
  /* Embedded controller */
  { 1, 3, "Cypress/ROSS CY7C611"},
  /* Ross Technologies HyperSparc */
  { 1, 0xf, "ROSS HyperSparc RT620"},
  { 1, 0xe, "ROSS HyperSparc RT625 or RT626"},
  /* ECL Implementation, CRAY S-MP Supercomputer... AIEEE! */
  /* Someone please write the code to support this beast! ;) */
  { 2, 0, "Bipolar Integrated Technology - B5010"},
  { 3, 0, "LSI Logic Corporation - unknown-type"},
  { 4, 0, "Texas Instruments, Inc. - SuperSparc-(II)"},
  /* SparcClassic  --  borned STP1010TAB-50*/
  { 4, 1, "Texas Instruments, Inc. - MicroSparc"},
  { 4, 2, "Texas Instruments, Inc. - MicroSparc II"},
  { 4, 3, "Texas Instruments, Inc. - SuperSparc 51"},
  { 4, 4, "Texas Instruments, Inc. - SuperSparc 61"},
  { 4, 5, "Texas Instruments, Inc. - unknown"},
  { 5, 0, "Matsushita - MN10501"},
  { 6, 0, "Philips Corporation - unknown"},
  { 7, 0, "Harvest VLSI Design Center, Inc. - unknown"},
  /* Gallium arsenide 200MHz, BOOOOGOOOOMIPS!!! */
  { 8, 0, "Systems and Processes Engineering Corporation (SPEC)"},
  { 9, 0, "Fujitsu or Weitek Power-UP"},
  { 9, 1, "Fujitsu or Weitek Power-UP"},
  { 9, 2, "Fujitsu or Weitek Power-UP"},
  { 9, 3, "Fujitsu or Weitek Power-UP"},
  /* from linux/arch/sparc64/kernel/cpu.c */
  { 0x17, 0x10, "TI UltraSparc I   (SpitFire)"},
  { 0x22, 0x10, "TI UltraSparc II  (BlackBird)"},
  { 0x17, 0x11, "TI UltraSparc II  (BlackBird)"},
  { 0x17, 0x12, "TI UltraSparc IIi"},
  { 0x17, 0x13, "TI UltraSparc IIe"},
  { 0x3e, 0x14, "TI UltraSparc III (Cheetah)"},
  /* old speling from earlier kernel versions */
  { 1, 0xe, "ROSS HyperSparc RT625"},
  { 9, 0, "Fujitsu #3"},
  { 4, 0, "Texas Instruments, Inc. - SuperSparc 50"},
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
        const char *p = "cpu\t\t: ";
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
#elif (CLIENT_OS == OS_SOLARIS)
  FILE *prtconf;
  char buf[256], name[256], c_name[256], *work_s, *work_t;
  int i, foundcpu = 0;

  /* DON'T RENUMBER */
  static struct { int rid; const char *raw_name, *name; } cpuridtable[] =
  {
  /* sun4 */
  /* sun4c */
  { 1, "Sun 4/20", "SPARCstation SLC"},
  { 2, "Sun 4/25", "SPARCstation ELC"},
  { 3, "Sun 4/40", "SPARCstation IPC"},
  { 4, "Sun 4/50", "SPARCstation IPX"},
  { 5, "Sun 4/60", "SPARCstation 1"}, 
  { 6, "Sun 4/65", "SPARCstation 1+"},
  { 7, "Sun 4/75", "SPARCstation 2"},
  /* sun4m */
  { 8, "TMS390S10", "microSPARC"},
  { 9, "MB86904", "microSPARC II"},
  {10, "MB86907", "TurboSPARC"},
  {11, "RT620", "hyperSPARC"},  /* used? */
  {11, "RT625", "hyperSPARC"},
  {11, "RT626", "hyperSPARC"},
  {12, "TMS390Z50", "SuperSPARC"},
  {13, "TMS390Z55", "SuperSPARC SC"},
  {14, "TMS390Z50", "SuperSPARC II"},
  {15, "TMS390Z55", "SuperSPARC II SC"},
  /* sun4d */
  /* sun4u */
  {16, "UltraSPARC", "UltraSPARC-I"},
  {17, "UltraSPARC-II", "UltraSPARC-II"},
  {18, "UltraSPARC-IIe", "UltraSPARC-IIe"},
  {19, "UltraSPARC-IIi", "UltraSPARC-IIi"},
  {20, "UltraSPARC-III", "UltraSPARC-III"},
  {20, "UltraSPARC-III+", "UltraSPARC-III"},  /* UltraSPARC-III Cu */
  {21, "UltraSPARC-IIIi", "UltraSPARC-IIIi"}, /* .13u */
  {21, "UltraSPARC-IIIi+", "UltraSPARC-IIIi+"}, /* unconfirmed, .09u, adds cache? */
  {22, "UltraSPARC-IV", "UltraSPARC-IV"},
  {22, "UltraSPARC-IV+", "UltraSPARC-IV+"}, /* unconfirmed, .09u adds 2MB L2 cache, external L3 doubled to 32MB */
  {23, "UltraSPARC-T1", "UltraSPARC-T1"},
  {23, "UltraSPARC-T2", "UltraSPARC-T2"}, /* unconfirmed */
  {24, "SPARC64-IV", "SPARC64-IV"},
  {25, "SPARC64-V", "SPARC64-V"},
  {26, "SPARC64-VI", "SPARC64-VI"}, /* untested, no data */
  };

  detectedtype = -1L;  /* detection supported, but failed */

  /* parse the prtconf output looking for the cpu name */
  strncpy (name, "", 256);
  strncpy (c_name, "", 256);
  /* 'prtconf -vp' outputs the detailed device node list from openboot ROM */
  if ((prtconf = popen ("/usr/sbin/prtconf -vp", "r")) != NULL) {
    while (fgets(buf, 256, prtconf) != NULL) {
      if (strstr (buf, "Node") != NULL) {  /* if new device node, clear name */
        if (foundcpu) {
          if (strlen (name) != 0) { /* we found a cpu name */
            break;
          } else if (strlen (c_name) != 0) { /* we found a cpu name in the 
                                                compatible field */
            strncpy (name, c_name, 256);
            break;
          } else {
            foundcpu = 0;
          }
        }
        strncpy (name, "", 256);
        strncpy (c_name, "", 256);
      }
      if (strstr (buf, "'cpu'") != NULL) {  /* if device is cpu */
        foundcpu = 1;
      }
      if ((work_s = strstr (buf, "name:")) != NULL) {  /* if value is 
                                                          device name */
        /* extract cpu name, format: "name:  '<manufacturer>,<cpu name>' */
        if ((work_s = strstr (buf, ",")) != NULL) {
          work_s++;
          work_t = strstr (work_s, "'");
          *(work_t) = '\0';
          strncpy (name, work_s, 256);
        }
      }
      if ((work_s = strstr (buf, "compatible:")) != NULL) {  /* if value is
                                                          device name held
                                                          in compatible field */
        /* extract cpu name, format: "name:  '<manufacturer>,<cpu name>' */
        if ((work_s = strstr (buf, ",")) != NULL) {
          work_s++;  
          work_t = strstr (work_s, "'");
          *(work_t) = '\0';
          strncpy (c_name, work_s, 256);
        }
      }
    }
    pclose (prtconf);
    if (strlen (name) != 0) {  /* if we found a cpu name */
      detectedtype = 0;
      for (i = 1; i <= (sizeof(cpuridtable)/sizeof(cpuridtable[0])); i++) {
        if (strcmp (name, cpuridtable[i-1].raw_name) == 0) {  /* found cpu ID */
          detectedname = cpuridtable[i-1].name;
          detectedtype = cpuridtable[i-1].rid;
          break;
        }
      }
    }
  }

  /* Detected SuperSPARC, **special case** */
  if ((detectedtype == 12) || (detectedtype == 13)) {
    if (GetProcessorFrequency() >= 75) {  
    /* If cpu speed is 75Mhz or more, then we have a SuperSPARC II */
      detectedtype += 2;          /* table must not be renumbered */
      detectedname = cpuridtable[detectedtype].name;
    }
  }

  if (detectedtype == 0) {  /* if we found an unknown cpu name, no ID */
    detectedname = name;
  }
  *cpuname = detectedname;
  return detectedtype;
#endif
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
                {       6, "EV45 (21064)"        },
                {       7, "EV56 (21164A)"       },
                {       8, "EV6 (21264)",        },
                {       9, "EV56 (21164PC)"      },
                {      10, "EV57"                },
                {      11, "EV67"                },
                {      12, "EV68CB"              },
                {      13, "EV68AL"              },
                {      14, "EV68CX"              },
                {      15, "EV69"                },
                {      16, "EV7"                 },
                {      17, "EV79"                }
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
  #elif (CLIENT_OS == OS_WIN32)
  if (detectedtype == -2L)
  {
    SYSTEM_INFO si;
    si.wProcessorLevel = si.wProcessorArchitecture = 0;
    detectedtype = -1L;
    GetSystemInfo( &si );
    if (si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ALPHA &&
        si.wProcessorLevel != 0)
    {
      unsigned int ref;
      for (ref = 0; (ref < sizeof(cpuridtable)/sizeof(cpuridtable[0])); ref++)
      {
        char *q = strchr(cpuridtable[ref].name,'(');
        if (q) 
        {
          if (si.wProcessorLevel == atoi(q+1))
          {
            detectedtype = cpuridtable[ref].rid;
            break;
          }
        }  
      }
      if (detectedtype == -1)
      {
        sprintf( namebuf, "%u", si.wProcessorLevel );
        detectedname = namebuf;
        detectedtype = 0;
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
           { "EV67",          11      },
           { "EV68CB",        12      },
           { "EV68AL",        13      },
           { "EV68CX",        14      },
           { "EV69",          15      },
           { "EV7",           16      },
           { "EV79",          17      }

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

#if (CLIENT_CPU == CPU_CUDA)
static inline long __GetRawProcessorID(const char **cpuname)
{
  return GetRawCUDAGPUID(cpuname);
}
#endif

/* ---------------------------------------------------------------------- */

#if (CLIENT_CPU == CPU_ATI_STREAM)
static inline long __GetRawProcessorID(const char **cpuname)
{
  return getAMDStreamRawProcessorID(cpuname);
}
#endif

/* ---------------------------------------------------------------------- */

//get (simplified) cpu ident by hardware detection
long GetProcessorType(int quietly)
{
  // only successful detection / detection of a new unknown cpu type gets logged to file
  long retval = -1L;
  const char *apd = "Automatic processor type detection ";
  #if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K)   || \
      (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER) || \
      (CLIENT_CPU == CPU_CELLBE)  || (CLIENT_CPU == CPU_X86)   || \
      (CLIENT_CPU == CPU_AMD64)   || (CLIENT_CPU == CPU_MIPS)  || \
      (CLIENT_CPU == CPU_SPARC)   || (CLIENT_CPU == CPU_ARM)   || \
      (CLIENT_CPU == CPU_CUDA)    || (CLIENT_CPU == CPU_ATI_STREAM)
  {
    const char *cpuname = NULL;
    long rawid = __GetRawProcessorID(&cpuname);
    if (rawid == -1L || rawid == -2L)
    {
      retval = -1L;  
      if (!quietly)
        LogScreen("%s%s.\n", apd, ((rawid == -1L)?("failed"):("is not supported")));
    }
    else if (rawid == 0)
    {
      retval = -1L;  
      if (!quietly)
        Log("%sdid not\nrecognize the processor (tag: \"%s\")\n", apd, (cpuname?cpuname:"???") );
    }
    else 
    {
      if (!quietly)
      {
        if (cpuname == NULL || *cpuname == '\0')
          Log("%sdid not\nrecognize the processor (id: %ld)\n", apd, rawid );
        else
          Log("%sfound\na%s %s processor.\n",apd, 
             ((strchr("aeiou8", tolower(*cpuname)))?("n"):("")), cpuname);
      }
      retval = rawid; /* let selcore figure things out */

      /* simply too many core<->cpu combinations */
      #if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64) 
      if ((retval = __GetRawProcessorID(NULL,'c')) < 0)
        retval = -1;
      #endif
    }
  }  
  #else
  {
    if (!quietly)
      LogScreen("%sis not supported.\n", apd );
  }
  #endif
  return retval;
}  

#if 0
//get cpu ident by hardware detection
long GetProcessorID()
{
  long retval = -1L;
  #if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K)   || \
      (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER) || \
      (CLIENT_CPU == CPU_CELLBE)  || (CLIENT_CPU == CPU_X86)   || \
      (CLIENT_CPU == CPU_ARM)     || (CLIENT_CPU == CPU_MIPS)  || \
      (CLIENT_CPU == CPU_SPARC)
  {
    long rawid = __GetRawProcessorID(NULL);
    if (rawid == -1L || rawid == -2L)
    {
      retval = -1L;  
    }
    else if (rawid == 0)
    {
      retval = -1L;  
    }
    else 
    {
      retval = rawid;
    }
  }  
  #endif
  return retval;
}
#endif

//Return the frequency in MHz, or 0.
unsigned int GetProcessorFrequency()
{
  unsigned int freq = 0;   /* Unknown */

  #if (CLIENT_CPU == CPU_CUDA)
    freq = GetCUDAGPUFrequency();
  #elif (CLIENT_CPU == CPU_ATI_STREAM)
    freq = getAMDStreamDeviceFreq();
  #elif (CLIENT_OS == OS_MACOSX)
    int mib[2] = {CTL_HW, HW_CPU_FREQ};
    unsigned long frequency;
    size_t len = sizeof(frequency);
    if (sysctl(mib, 2, &frequency, &len, NULL, 0) == 0) {
      if (frequency != 0)
        freq = (frequency + 500000) / 1000000;
    }
  #elif (CLIENT_OS == OS_SOLARIS)
    int i, n;
    processor_info_t infop;
    n = GetNumberOfDetectedProcessors ();
    for (i = 0; i < n; i++) {
      if (p_online (i, P_STATUS) == P_ONLINE) {
        break;
      }
    }
    if (processor_info (i, &infop) == 0) {
      freq = infop.pi_clock;
    }      
  #elif (CLIENT_OS == OS_AMIGAOS) && (CLIENT_CPU == CPU_POWERPC)
    #if defined(__amigaos4__)
      uint64 freqhz;
      IExec->GetCPUInfoTags(GCIT_ProcessorSpeed, &freqhz, TAG_DONE);
      if (freqhz != 0)
        freq = (freqhz + 500000) / 1000000;
    #elif !defined(__POWERUP__)
      struct TagItem cputags[2] = { {GETINFO_CPUCLOCK, 0}, {TAG_END,0} };
      GetInfo(cputags);
      if (cputags[0].ti_Data != 0)
        freq = (cputags[0].ti_Data + 500000) / 1000000;
    #else
      freq = PPCGetAttr(PPCINFOTAG_CPUCLOCK);
    #endif
  #elif (CLIENT_OS == OS_MORPHOS)
    UQUAD freqhz;
    if (NewGetSystemAttrsA(&freqhz, sizeof(freqhz),
                           SYSTEMINFOTYPE_PPC_CPUCLOCK, NULL))
    {
      freq = (freqhz + 500000) / 1000000;
    }
  #elif (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
  struct timeval tv1, tv2, elapsed_time;
    ui64 calltime = x86ReadTSC();
    sleep(0);
    calltime = x86ReadTSC() - calltime;
    
    CliGetMonotonicClock(&tv1); 
    ui64 prevtime = x86ReadTSC();
    sleep(1);
    ui64 newtime = x86ReadTSC();
    CliGetMonotonicClock(&tv2);
    CliTimerDiff(&elapsed_time,&tv1,&tv2);

    freq = (unsigned int)((newtime - prevtime - calltime) / (elapsed_time.tv_usec + elapsed_time.tv_sec * 1000000));
    if (freq != 0)
    {
      if (freq < 250) {
        unsigned int nearest25, nearest30, nearest33;
        if ((freq - ((unsigned int)(freq / 25) * 25)) < 
          (unsigned int)abs(freq - (((unsigned int)(freq / 25) + 1) * 25)))
        {
          nearest25 = (unsigned int)(freq / 25) * 25;
        } else {
          nearest25 = ((unsigned int)(freq / 25) + 1) * 25;
        }
        if ((freq - ((unsigned int)(freq / 30) * 30)) < 
          (unsigned int)abs(freq - (((unsigned int)(freq / 30) + 1) * 30)))
        {
          nearest30 = (unsigned int)(freq / 30) * 30;
        } else {
          nearest30 = ((unsigned int)(freq / 30) + 1) * 30;
        }
        if ((freq - ((unsigned int)(freq / (100.0/3.0)) * (100.0/3.0))) < 
          (unsigned int)abs((int)(freq - (((unsigned int)(freq / (100.0/3.0)) + 1) * (100.0/3.0)))))
        {
          nearest33 = (unsigned int)((unsigned int)(freq / (100.0/3.0)) * (100.0/3.0));
        } else {
          nearest33 = (unsigned int)(((unsigned int)(freq / (100.0/3.0)) + 1) * (100.0/3.0));
        }
        if (abs(freq - nearest25) < abs(freq - nearest30))
        {
          if (abs(freq - nearest25) < abs(freq - nearest33))
          {
            freq = nearest25;
          } else {
            freq = nearest33;
          }
        } 
        else if (abs(freq - nearest30) < abs(freq - nearest33))
        {
          freq = nearest30;
        } else {
          freq = nearest33;
        }
      } else {
        unsigned int nearest50, nearest66, nearest166;
        if ((freq - ((unsigned int)(freq / (50.0)) * (50.0))) <
          (unsigned int)abs((int)(freq - (((unsigned int)(freq / (50.0)) + 1) * (50.0)))))
        {
          nearest50 = (unsigned int)((unsigned int)(freq / (50.0)) * (50.0)); 
        } else {
          nearest50 = (unsigned int)(((unsigned int)(freq / (50.0)) + 1) * (50.0));
        }
        if ((freq - ((unsigned int)(freq / (200.0/3.0)) * (200.0/3.0))) < 
          (unsigned int)abs((int)(freq - (((unsigned int)(freq / (200.0/3.0)) + 1) * (200.0/3.0)))))
        {
          nearest66 = (unsigned int)((unsigned int)(freq / (200.0/3.0)) * (200.0/3.0));
        } else {
          nearest66 = (unsigned int)(((unsigned int)(freq / (200.0/3.0)) + 1) * (200.0/3.0));
        }
        if ((freq - ((unsigned int)(freq / (500.0/3.0)) * (500.0/3.0))) < 
          (unsigned int)abs((int)(freq - (((unsigned int)(freq / (500.0/3.0)) + 1) * (500.0/3.0)))))
        {
          nearest166 = (unsigned int)((unsigned int)(freq / (500.0/3.0)) * (500.0/3.0));
        } else {
          nearest166 = (unsigned int)(((unsigned int)(freq / (500.0/3.0)) + 1) * (500.0/3.0));
        }
        if (abs(freq - nearest50) < abs(freq - nearest66))
        {
          if (abs(freq - nearest50) < abs(freq - nearest166))
          {
            freq = nearest50;
          } else {
            freq = nearest166;
          }
        } 
        else if (abs(freq - nearest66) < abs(freq - nearest166))
        {
          freq = nearest66;
        } else {
          freq = nearest166;
        }
      }
    }
  #elif  (CLIENT_OS == OS_LINUX) && \
        ((CLIENT_CPU == CPU_POWERPPC) || (CLIENT_CPU == CPU_CELLBE))
    FILE *cpuinfo = fopen("/proc/cpuinfo", "r");
    if ( cpuinfo )
    {
      char buffer[256];
      while(fgets(buffer, sizeof(buffer), cpuinfo))
      {
        buffer[sizeof(buffer) - 1] = '\0';
        if (strncmp(buffer, "clock\t\t: ", 9) == 0)
        {
          freq = (unsigned int)strtod(buffer+9, NULL);
          break;
        }
      }
      fclose(cpuinfo);
    }
  #endif

  return freq;
}

//get a set of supported processor features
//cores may get disabled due to missing features
unsigned long GetProcessorFeatureFlags()
{
  #if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
    return (__GetRawProcessorID(NULL, 'f')) | (x86GetFeatures());
  #elif (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
    unsigned long ppc_features = 0;
    #if (CLIENT_OS == OS_MACOSX)
      // AltiVec support now has a proper sysctl value HW_VECTORUNIT to check
      // for
      int mib[2] = {CTL_HW, HW_VECTORUNIT};
      const char *sysname = "hw.cpusubtype";
      int hasVectorUnit, cpuSubType;
      size_t len = sizeof(hasVectorUnit);
      if (sysctl( mib, 2, &hasVectorUnit, &len, NULL, 0 ) == 0) {
        if (hasVectorUnit != 0)
          ppc_features |= CPU_F_ALTIVEC;
      }

      len = sizeof(cpuSubType);
      if (sysctlbyname( sysname, &cpuSubType, &len, NULL, 0 ) == 0) {
        if (cpuSubType == CPU_SUBTYPE_POWERPC_970)
          ppc_features |= CPU_F_64BITOPS;
      }
    #elif (CLIENT_OS == OS_LINUX)
      // Can someone write something better ?
      long type = __GetRawProcessorID(NULL);
      if ( (type & (1L << 25)) != 0)
        ppc_features |= CPU_F_ALTIVEC;
      if ( (type & 0xFFFF) == 0x0070) {     /* Cell Broadband Engine */
        ppc_features |= CPU_F_64BITOPS;
        ppc_features |= CPU_F_SYNERGISTIC;
      }
    #elif (CLIENT_OS == OS_AMIGAOS)  // AmigaOS PPC
      #if defined(__amigaos4__)
        /* AmigaOS 4.x */
        ULONG vec;
        char *extensions;
        IExec->GetCPUInfoTags(GCIT_VectorUnit, &vec, GCIT_Extensions, &extensions, TAG_DONE);

        if ((vec == VECTORTYPE_ALTIVEC) &&
            (extensions && strstr(extensions,"altivec")) &&
            ((SysBase->LibNode.lib_Version == 51 && SysBase->LibNode.lib_Revision >= 12) || SysBase->LibNode.lib_Version > 51))
        {
           ppc_features |= CPU_F_ALTIVEC;
        }
      #endif
    #elif (CLIENT_OS == OS_MORPHOS)  // MorphOS
      /* Altivec support was added in MorphOS 1.5 */
      struct Resident *m_res = FindResident("MorphOS");
      if (m_res && (m_res->rt_Flags & RTF_EXTENDED) &&
            ((m_res->rt_Version == 1 && m_res->rt_Revision >= 5) ||
            m_res->rt_Version > 1))
      {
        if ((SysBase->LibNode.lib_Version == 50 && SysBase->LibNode.lib_Revision >= 60)
          || SysBase->LibNode.lib_Version > 50)
      {
        ULONG avf = 0;
        NewGetSystemAttrsA(&avf, sizeof(avf), SYSTEMINFOTYPE_PPC_ALTIVEC, NULL);
        if (avf)
        {
          ppc_features |= CPU_F_ALTIVEC;
        }
      }
    }
    #endif
    return ppc_features;

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

#if (CLIENT_CPU == CPU_ALPHA)   || (CLIENT_CPU == CPU_68K)   || \
    (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_POWER) || \
    (CLIENT_CPU == CPU_CELLBE)  || (CLIENT_CPU == CPU_X86)   || \
    (CLIENT_CPU == CPU_AMD64)   || (CLIENT_CPU == CPU_MIPS)  || \
    (CLIENT_CPU == CPU_SPARC)   || (CLIENT_CPU == CPU_ARM)   || \
    (CLIENT_CPU == CPU_CUDA)    || (CLIENT_CPU == CPU_ATI_STREAM)
  long rawid = __GetRawProcessorID(&cpuid_s);
  if (rawid == -1L || rawid == -2L)
    cpuid_s = ((rawid==-1)?("?\n\t(identification failed)"):
              ("none\n\t(client does not support identification)"));
  else
  {
    static char namebuf[200];
    if (cpuid_s == NULL) cpuid_s = "*unknown*";
    if (*cpuid_s =='\0') cpuid_s = "???";
  #if (CLIENT_CPU == CPU_ARM)
    namebuf[0] = '\0';
    if (rawid != 0) /* if rawid == 0, then cpuid_s == "%lX" */
      sprintf( namebuf, "%lX\n\tname: ", rawid );
    strcat( namebuf, cpuid_s ); /* always valid */
    #if (CLIENT_OS == OS_RISCOS && defined(HAVE_X86_CARD_SUPPORT))
    if (riscos_count_cpus() == 2)
      strcat(strcat(namebuf,"\n\t+ "),riscos_x86_determine_name());
    #endif
  #elif (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
    long features;
    namebuf[0] = '\0';
    if (rawid != 0) /* if rawid == 0, then cpuid_s == "%08x" */
      sprintf( namebuf, "%08X\n\tname: ",(int)rawid);
    strcat( namebuf, cpuid_s ); /* always valid */
    strcat( namebuf, "\n\tfeatures: " );
    features = GetProcessorFeatureFlags();
    if (features & CPU_F_MMX) {
      strcat( namebuf, "MMX " );
    }
    if (features & CPU_F_CYRIX_MMX_PLUS) {
      strcat( namebuf, "Cyrix_MMX+ " );
    }
    if (features & CPU_F_AMD_MMX_PLUS) {   
      strcat( namebuf, "AMD_MMX+ " );   
    }
    if (features & CPU_F_3DNOW) {   
      strcat( namebuf, "3DNOW " );   
    }
    if (features & CPU_F_3DNOW_PLUS) {   
      strcat( namebuf, "3DNOW+ " );   
    }
    if (features & CPU_F_SSE) {   
      strcat( namebuf, "SSE " );   
    }
    if (features & CPU_F_SSE2) {   
      strcat( namebuf, "SSE2 " );   
    }
    if (features & CPU_F_SSE3) {   
      strcat( namebuf, "SSE3 " );   
    }
    if (features & CPU_F_AMD64) {
      strcat( namebuf, "AMD64 " );
    }
    if (features & CPU_F_EM64T) {
      strcat( namebuf, "EM64T " );
    }
    if (features & CPU_F_SSSE3) {
      strcat( namebuf, "SSSE3 ");
    }
    if (features & CPU_F_SSE4_1) {
      strcat( namebuf, "SSE4.1 ");
    }
    if (features & CPU_F_SSE4_2) {
      strcat( namebuf, "SSE4.2 ");
    }
  #else
    sprintf(namebuf, "%ld\n\tname: %s", rawid, cpuid_s );
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
  unsigned int clockmhz = GetProcessorFrequency();

  LogRaw("Automatic processor identification tag: %s\n"
    "Estimated processor clock speed (0 if unknown): %u MHz\n"
    "Number of processors detected by this client: %s\n"
    "Number of processors supported by this client: %s\n",
    scpuid, clockmhz, sfoundcpus, smaxscpus );

  #if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
    x86ShowInfos();
  #endif
  return;
}

/* ---------------------------------------------------------------------- */
