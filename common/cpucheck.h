/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2026 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CPUCHECK_H__
#define __CPUCHECK_H__

// cpu feature flags (use by selcore.cpp)

#define CPU_F_MASK              (0x0000ffffL)

#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_AMD64)
  #define CPU_F_I386            (0x00000001L)
  #define CPU_F_I486            (0x00000002L | CPU_F_I386)
  #define CPU_F_I586            (0x00000004L | CPU_F_I486)
  #define CPU_F_I686            (0x00000008L | CPU_F_I586)
  #define CPU_F_MMX             (0x00000100L)
  #define CPU_F_CYRIX_MMX_PLUS  (0x00000200L)
  #define CPU_F_AMD_MMX_PLUS    (0x00000400L)
  #define CPU_F_3DNOW           (0x00000800L)
  #define CPU_F_3DNOW_PLUS      (0x00001000L)
  #define CPU_F_SSE             (0x00002000L)
  #define CPU_F_SSE2            (0x00004000L)
  #define CPU_F_SSE3            (0x00008000L)
  #define CPU_F_HYPERTHREAD     (0x00010000L)   /* supported and enabled */
  #define CPU_F_AMD64           (0x00020000L)
  #define CPU_F_EM64T           (0x00040000L)
  #define CPU_F_SSE4_1          (0x00080000L)
  #define CPU_F_SSE4_2          (0x00100000L)
  #define CPU_F_SSSE3           (0x00200000L)
  #define CPU_F_LZCNT           (0x00400000L)
  #define CPU_F_AVX_DISABLED    (0x00800000L)   /* supported but disabled (no OS support) */
  #define CPU_F_AVX             (0x01000000L)   /* supported and enabled */
  #define CPU_F_AVX2            (0x02000000L)   /* supported and enabled */
  #define CPU_F_AVX512          (0x04000000L)   /* F+CD supported and enabled */

// "core hint": ability to override core selection for some CPUs in the family
  #define CH_R72_X86_GO2B       (0x00000001L)

unsigned long GetProcessorCoreHints(void);

#endif

#if (CLIENT_CPU == CPU_POWERPC) || (CLIENT_CPU == CPU_CELLBE)
  #define CPU_F_ALTIVEC         (0x00000001L)
  #define CPU_F_64BITOPS        (0x00000002L)
  #define CPU_F_SYNERGISTIC     (0x00000004L)
#endif

// constants for Apple Silicon CPU families
#if (CLIENT_CPU == CPU_ARM64)
  #define CPU_ARM64_UNKNOWN             (0xFF)
  #define CPU_APPLE_M1                  (0x10)      // Firestorm/Icestorm
  #define CPU_APPLE_M2                  (0x20)      // Blizzard/Avalanche
  #define CPU_APPLE_M3                  (0x30)      // Ibiza
  #define CPU_APPLE_M3_PRO              (0x31)      // Lobos
  #define CPU_APPLE_M3_MAX_ULTRA        (0x32)      // Palma
  #define CPU_APPLE_M4                  (0x40)      // Donan
  #define CPU_APPLE_M4_PRO_MAX          (0x41)      // Brava
  #define CPU_APPLE_M5                  (0x50)      // Hidra
  #define CPU_APPLE_M5_PRO_MAX_ULTRA    (0x51)      // Sotra
  #define CPU_APPLE_M6                  (0x60)
  
  #define CPU_APPLE_A18_PRO             (0xA18)     // Tahiti

  #define CPU_APPLE_M1_NAME                 "Apple M1"
  #define CPU_APPLE_M1_PRO_NAME             "Apple M1 Pro"
  #define CPU_APPLE_M1_MAX_NAME             "Apple M1 Max"
  #define CPU_APPLE_M2_NAME                 "Apple M2"
  #define CPU_APPLE_M2_PRO_NAME             "Apple M2 Pro"
  #define CPU_APPLE_M2_MAX_NAME             "Apple M2 Max"
  #define CPU_APPLE_M3_NAME                 "Apple M3"
  #define CPU_APPLE_M3_PRO_NAME             "Apple M3 Pro"
  #define CPU_APPLE_M3_MAX_NAME             "Apple M3 Max"
  #define CPU_APPLE_M3_MAX_ULTRA_NAME       "Apple M3 Max/Ultra"
  #define CPU_APPLE_M4_NAME                 "Apple M4"
  #define CPU_APPLE_M4_PRO_NAME             "Apple M4 Pro"
  #define CPU_APPLE_M4_MAX_NAME             "Apple M4 Max"
  #define CPU_APPLE_M4_PRO_MAX_NAME         "Apple M4 Pro/Max"
  #define CPU_APPLE_M5_NAME                 "Apple M5"
  #define CPU_APPLE_M5_PRO_MAX_ULTRA_NAME   "Apple M5 Pro/Max/Ultra"
  #define CPU_APPLE_M6_NAME                 "Apple M6"

  #define CPU_APPLE_A18_PRO_NAME            "Apple A18 Pro"
#endif

//return number of processors detected (by the hardware/from the OS)
//returns -1 if detection is not supported.
//returns 0 if a required co-processor (e.g. GPU) was not found
int GetNumberOfDetectedProcessors( void );

//currently returns GetNumberOfDetectedProcessors()
int GetNumberOfLogicalProcessors ( void );

//if hyperthreading supported and enabled returns number of physical cpus
int GetNumberOfPhysicalProcessors ( void );

//get (simplified) cpu ident by hardware detection
long GetProcessorType(int quietly, int device = 0);

#if 0
//get cpu ident by hardware detection
long GetProcessorID();
#endif

//get a set of supported processor features
//cores may get disabled due to missing features
unsigned long GetProcessorFeatureFlags(int device = 0);

//Return cpuid/cputag, maxcpus, foundcpus as descriptive strings.
//Assists in debugging user complaints/bug reports.
void GetProcessorInformationStrings( int device, const char ** scpuid,
                  const char ** smaxcpus, const char ** sfoundcpus );

//Wrapper for GetProcessorInformationStrings()
//(used to be a client class function for access to the log functions)
void DisplayProcessorInformation( void );

//Return the CPU frequency (in MHz)
unsigned int GetProcessorFrequency(int device = 0);

#endif /* __CPUCHECK_H__ */
