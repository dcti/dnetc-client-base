// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cpucheck-conflict.cpp,v $
// Revision 1.8  1998/07/05 06:55:06  silby
// Change to the bitmask for intel CPUIDs so that secondary CPUs will be id'd correctly.
//
// Revision 1.7  1998/06/28 19:48:13  silby
// Changed default amd 486 core selection to pentium core and changed strings to reflect that.
//
// Revision 1.6  1998/06/23 20:22:05  cyruspatel
// Added new function: GetTimesliceBaseline() returns a value that the
// ideal RC5 keyrate (kKeys per Mhz) would be IF a machine were running
// at peak efficiency. For non-preemptive systems, it is thus a good
// indicator of how low we can set the timeslice/rate-of-yield without
// losing efficiency. Or inversely, at what point OS responsiveness starts
// to suffer - which also applies to preemptive but non-mt systems handling
// of a break request. - Currently only supports all the x86 OS's.
//
// Revision 1.5  1998/06/22 10:28:22  kbracey
// Just tidying
//
// Revision 1.4  1998/06/22 09:37:47  cyruspatel
// Fixed another cosmetic bug.
//
// Revision 1.3  1998/06/22 03:40:23  cyruspatel
// Fixed a numcputemp/cpu_count variable mixup. (thanks Silby)
//
// Revision 1.1  1998/06/21 17:12:02  cyruspatel
// Created from code spun off from cliconfig.cpp
//
//

#include "client.h"

#if (!defined(lint) && defined(__showids__))
static const char *id="@(#)$Id: cpucheck-conflict.cpp,v 1.8 1998/07/05 06:55:06 silby Exp $";
#endif

// --------------------------------------------------------------------------

static int __GetProcessorCount()  //returns -1 if not supported
{
  static int cpucount = -2;

  if (cpucount == -2)
    {
    cpucount = -1;
    #if (CLIENT_OS == OS_BEOS)
      {
      system_info the_info;
      get_system_info(&the_info);
      cpucount = the_info.cpu_count;
      if (cpucount < 1)
        cpucount = 1;
      }
    #elif (CLIENT_OS == OS_WIN32)
      {
      SYSTEM_INFO systeminfo;
      GetSystemInfo(&systeminfo);
      cpucount = systeminfo.dwNumberOfProcessors;
      if (cpucount < 1)
        cpucount = 1;
      }
    #elif (CLIENT_OS == OS_NETWARE)
      {
      cpucount = CliGetNumberOfProcessors();
      }
    #elif (CLIENT_OS == OS_OS2)
      {
      int rc = DosQuerySysInfo(QSV_NUMPROCESSORS, QSV_NUMPROCESSORS,
                  &cpucount, sizeof(cpucount));
      // check if call is valid if not, default to one
      if (rc!=0 || cpucount < 1)
        cpucount = 1;
      }
    #endif
    if (cpucount < 1)  //not supported
      cpucount = -1;
    }
  return cpucount;
}

// --------------------------------------------------------------------------

void Client::ValidateProcessorCount( void )
{
  //-------------------------------------------------------------
  // Validate processor/thread count.
  // --------------------
  // numcpu is the value read/written from the ini/cmdline - we never modify
  // it. numcputemp is the value actually used/modified internally by the
  // client and is not written out to the ini.
  //
  // The use of -numcpu on the command line is now *not* illegal
  // for non-mt clients (think of shell scripts used for testing).
  // The use of numcpu is not illegal in the ini either and is a valid
  // option in the config menu. (think of shared/cross-configured ini
  // files, in dos/win for instance).
  //
  // Simply put, if it ain't MULTITHREAD, there's ain't gonna be a thread,
  // no matter what the user puts on the command line or in the ini file.
  //--------------------

  numcputemp = numcpu;
  #ifndef MULTITHREAD  //this is the only place in the config where we
  numcputemp = 1;      //check this
  #endif

  if (numcputemp < 1)
    {
    static int cpu_count = -2;
    if (cpu_count == -2)
      {
      cpu_count = __GetProcessorCount(); //in cpuinfo.cpp
      // returns -1 if no hardware detection
      if ( cpu_count < 1 )
        {
        // LogScreen("Automatic processor count detection is not supported "
        // "on this platform.\nA single processor machine is assumed.\n");
        cpu_count = 1;
        }
      else
        {
        LogScreenf("Automatic processor detection found %d processor%s\n",
           cpu_count, ((cpu_count==1)?(""):("s")) );
        }
      }
    numcputemp = cpu_count;
    }
  if (numcputemp < 1)
    numcputemp = 1;
  if (numcputemp > MAXCPUS)
    numcputemp = MAXCPUS;
  #if ((CLIENT_CPU != CPU_X86) && (CLIENT_CPU != CPU_88K) && \
      (CLIENT_CPU != CPU_SPARC) && (CLIENT_CPU != CPU_POWERPC))
    if ( numcputemp > 1 )
      {
      numcputemp = 1;
      LogScreen("Core routines not yet updated for thread safe operation. "
                "Using 1 processor.\n");
      }
  #endif
  return;
}

// --------------------------------------------------------------------------

#if (!((CLIENT_CPU == CPU_X86) || \
      ((CLIENT_CPU == CPU_ARM) && (CLIENT_OS == OS_RISCOS)) ))
int Client::GetProcessorType()
{ return -1; }
#endif

// --------------------------------------------------------------------------

#if (CLIENT_CPU == CPU_X86)

#ifdef __WATCOMC__
  #define x86ident _x86ident
#endif
#if (CLIENT_OS == OS_LINUX) && !defined(__ELF__)
  extern "C" u32 x86ident( void ) asm ("x86ident");
#else
  extern "C" u32 x86ident( void );
#endif

struct _cpuxref { int cpuidb;
                  unsigned int kKeysPerMhz; //numbers are from Alde's tables
                  int coretouse; 
                  char *cpuname; };

struct _cpuxref *__GetProcessorXRef( int *cpuidbP, int *vendoridP,
                               char **pronounP, char **vendornameP )
{
  char *pronoun = NULL; //"an" "a"
  char *vendorname = NULL; //"Cyrix", "Centaur", "AMD", "Intel", ""
  static struct _cpuxref *cpuxref = NULL;

  u32 detectedvalue = x86ident(); //must be interpreted
  int vendorid = (int)(detectedvalue >> 16);
  int cpuidb  = (int)(detectedvalue & 0xffff);
  
  if (cpuidbP) *cpuidbP = cpuidb;
  if (vendoridP) *vendoridP = vendorid;

  if ( vendorid == 0x7943) // Cyrix CPU
    {
    pronoun = "a";
    vendorname = "Cyrix";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {    0x40, 1024,   0, "486"     }, // use Pentium core
      {  0x0440, 1024,   0, "MediaGX" },
      {  0x0490, 1185,   0, "5x86"    },
      {  0x0520, 2090,   3, "6x86"    }, // "Cyrix 6x86/6x86MX/M2"
      {  0x0540, 1200,   0, "GXm"     }, // use Pentium core here too
      {  0x0600, 2115,   3, "6x86MX"  },
      {  0x0000, 2115,   3, NULL      } //default core == 6x86
      }; cpuxref = &__cpuxref[0];
    }
  else if ( vendorid == 0x6543) //centaur/IDT cpu
    {
    pronoun = "a";
    vendorname = "Centaur/IDT";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {  0x0540, 1200,   0, "C6"      }, // use Pentium core
      {  0x0000, 1200,   0, NULL      }  // default core == Pentium
      }; cpuxref = &__cpuxref[0];
    }
  else if ( vendorid == 0x7541) // AMD CPU
    {
    pronoun = "an";
    vendorname = "AMD";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {  0x0040, 1024,   0, "486"      },   // "Pentium, Pentium MMX, Cyrix 486/5x86/MediaGX, AMD 486",
      {  0x0430, 1024,   0, "486DX2"   },
      {  0x0470, 1024,   0, "486DX2WB" },
      {  0x0480, 1024,   0, "486DX4"   },
      {  0x0490, 1024,   0, "486DX4WB" },
      {  0x04E0, 1185,   0, "5x86"     },
      {  0x04F0, 1185,   0, "5x86WB"   },
      {  0x0500, 2353,   4, "K5 PR75, PR90, or PR100" }, // use K5 core
      {  0x0510, 2353,   4, "K5 PR120 or PR133" },
      {  0x0520, 2353,   4, "K5 PR166" },
      {  0x0530, 2353,   4, "K5 PR200" },
      {  0x0560, 1611,   5, "K6"       },
      {  0x0570, 1611,   5, "K6"       },
      {  0x0580, 1611,   5, "K6-2"     },
      {  0x0590, 1611,   5, "K6-3"     },
      {  0x0000, 1611,   5, NULL       }   // for the future - default core = K6
      }; cpuxref = &__cpuxref[0];
    }
  else if (vendorid == 0x6E49 || vendorid == 0x6547) // Intel CPU
    {
    pronoun = "an";
    vendorname = "Intel";
    if ((cpuidb == 0x30) || (cpuidb == 0x40))
      vendorname = ""; //generic 386/486
    cpuidb &= 0x0ff0; //strip last 4 bits, don't need stepping info
    static struct _cpuxref __cpuxref[]={
      {  0x0030, 0426,  1, "80386"    },   // generic 386/486 core
      {  0x0040, 1024,  1, "80486"    },   // - 946 ('95) + 1085 (NetWare) 
      {  0x0400, 1024,  1, "486DX 25 or 33" },
      {  0x0410, 1024,  1, "486DX 50" },
      {  0x0420, 1024,  1, "486SX" },
      {  0x0430, 1024,  1, "486DX2" },
      {  0x0440, 1024,  1, "486SL" },
      {  0x0450, 1024,  1, "486SX2" },
      {  0x0470, 1024,  1, "486DX2WB" },
      {  0x0480, 1024,  1, "486DX4" },
      {  0x0490, 1024,  1, "486DX4WB" },
      {  0x0500, 1416,  0, "Pentium" }, //stepping A
      {  0x0510, 1416,  0, "Pentium" },    
      {  0x0520, 1416,  0, "Pentium" },
      {  0x0530, 1416,  0, "Pentium Overdrive" },
      {  0x0540, 1432,  0, "Pentium MMX" },
      {  0x0570, 1416,  0, "Pentium" },
      {  0x0580, 1432,  0, "Pentium MMX" },
      {  0x0600, 2785,  2, "Pentium Pro" },
      {  0x0610, 2785,  2, "Pentium Pro" },
      {  0x0630, 2785,  2, "Pentium II" },
      {  0x0650, 2785,  2, "Pentium II" },
      {  0x0000, 2785,  2, NULL         }  // default core = PPro/PII
      }; cpuxref = &__cpuxref[0];
    }

  if (pronounP) *pronounP = pronoun;
  if (vendornameP) *vendornameP = vendorname;

  if ( cpuxref != NULL ) // we have a mfg's table
    {
    unsigned int pos;
    for (pos=0 ; ; pos++)
      {
      if (( (cpuxref[pos].cpuname)==NULL ) ||
            ( cpuidb == (cpuxref[pos].cpuidb)) )
        return (&(cpuxref[pos]));
      }
    }
  return NULL;
}  

// ---------------------

int Client::GetProcessorType()
{
  int coretouse = 0; // the core the client should. Default is Pentium
  int vendorid, cpuidb;                           
  char *pronoun, *vendorname;
  struct _cpuxref *cpuxref = 
          __GetProcessorXRef( &cpuidb, &vendorid, &pronoun, &vendorname );

  LogScreen( "Automatic processor detection " );
  if ( cpuxref == NULL ) // fell through
    {
    LogScreenf( "failed. (id: %04X:%04X)\n", vendorid, cpuidb );
    coretouse = 0;
    }
  else if ( cpuxref->cpuname == NULL )  // fell through to last element
    {
    coretouse = (cpuxref->coretouse);
    LogScreenf("found an unrecognized %s processor. (id: %04X)",
                                                    vendorname, cpuidb );
    }
  else // if ( cpuidb == (cpuxref->cpuidb))
    {
    coretouse = (cpuxref->coretouse);      
    if ( !vendorname || !*vendorname )  // generic type - no vendor name
      LogScreenf( "found %s %s.\n", pronoun, (cpuxref->cpuname));
    else
      LogScreenf( "found %s %s %s.\n", pronoun,
                                     vendorname, (cpuxref->cpuname));
    }
  return coretouse;
}
#endif // client_cpu == x86 or not

// --------------------------------------------------------------------------

#if ((CLIENT_CPU == CPU_ARM) && (CLIENT_OS == OS_RISCOS))

#include <setjmp.h>

static jmp_buf ARMident_jmpbuf;

static void ARMident_catcher(int)
{
  longjmp(ARMident_jmpbuf, 1);
}

int Client::GetProcessorType()
{
  u32 realid, detectedvalue; // value ARMident returns, must be interpreted
  int coretouse; // the core the client should use

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

  LogScreen("Automatic processor detection ");

  switch (detectedvalue)
  {
    case 0x200:
      LogScreen("found an ARM 2 or ARM 250.\n");
      coretouse=2;
      break;
    case 0x3:
    case 0x600:
    case 0x610:
    case 0x700:
    case 0x7500:
    case 0x7500FE:
      LogScreenf("found an ARM %X.\n", detectedvalue);
      coretouse=0;
      break;
    case 0x710:
      LogScreenf("found an ARM %X.\n", detectedvalue);
      coretouse=3;
      break;
    case 0x810:
      LogScreenf("found an ARM %X.\n", detectedvalue);
      coretouse=1;
      break;
    case 0xA10:
      LogScreen("found a StrongARM 110.\n");
      coretouse=1;
      break;
    default:
      LogScreenf("failed. (id: %08X)\n", realid);
      coretouse=-1;
      break;
  }
  return coretouse;
}

#endif //Arm/riscos cpucheck

// --------------------------------------------------------------------------

// GetTimesliceBaseline() returns a value that the ideal RC5 keyrate (kKeys 
// per Mhz) would be IF a machine were running at peak efficiency. For 
// non-preemptive systems, it is thus a good indicator of how low we can 
// set the timeslice/rate-of-yield without losing efficiency. Or inversely, 
// at what point OS responsiveness starts to suffer - which also applies to 
// preemptive but non-mt systems handling of a break request. 

unsigned int GetTimesliceBaseline(void) 
{ 
#if (CLIENT_CPU == CPU_X86)    
  struct _cpuxref *cpuxref = __GetProcessorXRef( NULL, NULL, NULL, NULL );
  return ((!cpuxref) ? ( 1024 ) : ( cpuxref->kKeysPerMhz ));
#else 
  return 0;
#endif    
}  
 
// --------------------------------------------------------------------------
