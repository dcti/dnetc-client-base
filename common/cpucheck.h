/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2003 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CPUCHECK_H__
#define __CPUCHECK_H__ "@(#)$Id: cpucheck.h,v 1.16 2003/11/01 14:20:13 mweiser Exp $"

// cpu feature flags (use by selcore.cpp)

#define CPU_F_MASK              (0x0000ffffL)

#if (CLIENT_CPU == CPU_X86)
  #define CPU_F_I386            (0x00000001L)
  #define CPU_F_I486            (0x00000002L | CPU_F_I386)
  #define CPU_F_I586            (0x00000004L | CPU_F_I486)
  #define CPU_F_I686            (0x00000008L | CPU_F_I586)
  #define CPU_F_MMX             (0x00000100L)
// CPU_F_SSE, CPU_F_SSE2, ...
  #define CPU_F_I586MMX         (CPU_F_I586  | CPU_F_MMX)
  #define CPU_F_I686MMX         (CPU_F_I686  | CPU_F_MMX)
#endif


// cpu feature flags (use by selcore.cpp)

#define CPU_F_MASK              (0x0000ffffL)

#if (CLIENT_CPU == CPU_X86) || (CLIENT_CPU == CPU_X86_64)
  #define CPU_F_I386            (0x00000001L)
  #define CPU_F_I486            (0x00000002L | CPU_F_I386)
  #define CPU_F_I586            (0x00000004L | CPU_F_I486)
  #define CPU_F_I686            (0x00000008L | CPU_F_I586)
  #define CPU_F_MMX             (0x00000100L)
// CPU_F_SSE, CPU_F_SSE2, ...
  #define CPU_F_I586MMX         (CPU_F_I586  | CPU_F_MMX)
  #define CPU_F_I686MMX         (CPU_F_I686  | CPU_F_MMX)
#endif


//return number of processors detected (by the hardware/from the OS)
//returns -1 if detection is not supported.
int GetNumberOfDetectedProcessors( void );

//get (simplified) cpu ident by hardware detection
long GetProcessorType(int quietly);

//get cpu ident by hardware detection
long GetProcessorID();

//get a set of supported processor features
//cores may get disabled due to missing features
unsigned long GetProcessorFeatureFlags();

//Return cpuid/cputag, maxcpus, foundcpus as descriptive strings.
//Assists in debugging user complaints/bug reports.
void GetProcessorInformationStrings( const char ** scpuid, 
                  const char ** smaxcpus, const char ** sfoundcpus );

//Wrapper for GetProcessorInformationStrings()
//(used to be a client class function for access to the log functions)
void DisplayProcessorInformation( void );

#endif /* __CPUCHECK_H__ */
