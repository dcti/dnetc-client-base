// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cpucheck-conflict.cpp,v $
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
static const char *id="@(#)$Id: cpucheck-conflict.cpp,v 1.5 1998/06/22 10:28:22 kbracey Exp $";
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

int Client::GetProcessorType()
{
  struct _cpuxref { int cpuidb, coretouse; char *cpuname; } *cpuxref = NULL;
  char *pronoun = NULL; //"an" "a"
  char *vendorname = NULL; //"a Cyrix", "a Centaur", "an AMD", "an Intel"
  int coretouse = 0; // the core the client should use of the 5(6?)

  u32 detectedvalue = x86ident(); //must be interpreted
  int vendorid = (int)(detectedvalue >> 16);
  int cpuidb  = (int)(detectedvalue & 0xffff);

  if ( vendorid == 0x7943) // Cyrix CPU
    {
    pronoun = "a";
    vendorname = " Cyrix";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    struct _cpuxref __cpuxref[]={
      {    0x40,   0, "486"     }, // use Pentium core
      {  0x0490,   0, "5x86"    },
      {  0x0440,   0, "MediaGX" },
      {  0x0520,   3, "6x86"    }, // "AMD 486, Cyrix 6x86/6x86MX/M2"
      {  0x0540,   0, "GXm"     }, // use Pentium core here too
      {  0x0600,   3, "6x86MX"  },
      {  0x0000,   3, NULL      } //default core == 6x86
      }; cpuxref = &__cpuxref[0];
    }
  else if ( vendorid == 0x6543) //centaur/IDT cpu
    {
    pronoun = "a";
    vendorname = " Centaur/IDT";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    struct _cpuxref __cpuxref[]={
      {  0x0540,   0, "C6"      }, // use Pentium core
      {  0x0000,   0, NULL      }  // default core == Pentium
      }; cpuxref = &__cpuxref[0];
    }
  else if ( vendorid == 0x7541) // AMD CPU
    {
    pronoun = "an";
    vendorname = " AMD";
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    struct _cpuxref __cpuxref[]={
      {  0x0040,   3, "486"      },   // "AMD 486, Cyrix 6x86/6x86MX/M2",
      {  0x0430,   3, "486DX2"   },
      {  0x0470,   3, "486DX2WB" },
      {  0x0480,   3, "486DX4"   },
      {  0x0490,   3, "486DX4WB" },
      {  0x04E0,   3, "5x86"     },
      {  0x04F0,   3, "5x86WB"   },
      {  0x0500,   4, "K5 PR75, PR90, or PR100" }, // use K5 core
      {  0x0510,   4, "K5 PR120 or PR133" },
      {  0x0520,   4, "K5 PR166" },
      {  0x0530,   4, "K5 PR200" },
      {  0x0560,   5, "K6"       },
      {  0x0570,   5, "K6"       },
      {  0x0580,   5, "K6-2"     },
      {  0x0590,   5, "K6-3"     },
      {  0x0000,   5, NULL       }   // for the future - default core = K6
      }; cpuxref = &__cpuxref[0];
    }
  else if (vendorid == 0x6E49 || vendorid == 0x6547) // Intel CPU
    {
    pronoun = "an";
    vendorname = " Intel";
    if ((cpuidb == 0x30) || (cpuidb == 0x40))
      vendorname = ""; //generic 386/486
    cpuidb &= 0xfff0; //strip last 4 bits, don't need stepping info
    struct _cpuxref __cpuxref[]={
      {  0x0030,   1, "80386"    },   // generic 386/486 core
      {  0x0040,   1, "80486"    },
      {  0x0400,   1, "486DX 25 or 33" },
      {  0x0410,   1, "486DX 50" },
      {  0x0420,   1, "486SX" },
      {  0x0430,   1, "486DX2" },
      {  0x0440,   1, "486SL" },
      {  0x0450,   1, "486SX2" },
      {  0x0470,   1, "486DX2WB" },
      {  0x0480,   1, "486DX4" },
      {  0x0490,   1, "486DX4WB" },
      {  0x0500,   0, "Pentium" }, //stepping A
      {  0x0510,   0, "Pentium" },
      {  0x0520,   0, "Pentium" },
      {  0x0530,   0, "Pentium Overdrive" },
      {  0x0540,   0, "Pentium MMX" },
      {  0x0570,   0, "Pentium" },
      {  0x0580,   0, "Pentium MMX" },
      {  0x0600,   2, "Pentium Pro" },
      {  0x0610,   2, "Pentium Pro" },
      {  0x0630,   2, "Pentium II" },
      {  0x0650,   2, "Pentium II" },
      {  0x0000,   2, NULL         }  // default core = PPro/PII
      }; cpuxref = &__cpuxref[0];
    }

  LogScreen( "Automatic processor detection " );
  if ( cpuxref == NULL ) // fell through
    {
    cpuidb = (detectedvalue & 0xffff); //restore all bits
    LogScreenf( "failed. (id: %04X:%04X)\n", vendorid, cpuidb );
    }
  else // we have a mfg's table
    {
    unsigned int pos;
    for (pos=0 ; ; pos++)
      {
      if ( (cpuxref[pos].cpuname)==NULL )
        {
        coretouse = (cpuxref[pos].coretouse);
        cpuidb = (detectedvalue & 0xffff); //restore all bits
        LogScreenf("found an unrecognized%s processor. (id: %04X)",
                                                    vendorname, cpuidb );
        break;
        }
      if ( cpuidb == (cpuxref[pos].cpuidb))
        {
        coretouse = (cpuxref[pos].coretouse);  //show the name
        LogScreenf( "found %s%s %s.\n", pronoun,
                                     vendorname, (cpuxref[pos].cpuname));
        break;
        }
      }
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
