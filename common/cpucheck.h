/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CPUCHECK_H__
#define __CPUCHECK_H__ "@(#)$Id: cpucheck.h,v 1.10.2.1 1999/04/13 19:45:21 jlawson Exp $"

//get core type by hardware detection
int GetProcessorType(int quietly);

//returns the number of cpus (>=0). 
//Zero is valid and symbolizes 'force-single-threading'.
unsigned int ValidateProcessorCount(int numcpu, int quietly);

//Return cpuid/cputag, maxcpus, foundcpus as descriptive strings.
//Assists in debugging user complaints/bug reports.
void GetProcessorInformationStrings( const char ** scpuid, 
                  const char ** smaxcpus, const char ** sfoundcpus );

//Wrapper for GetProcessorInformationStrings()
//(used to be a client class function for access to the log functions)
void DisplayProcessorInformation( void );

//returns number of processors supported by the client.
//should this be in client.cpp?
unsigned int GetNumberOfSupportedProcessors( void );

//return number of processors detected (by the hardware/from the OS)
//returns -1 if detection is not supported.
int GetNumberOfDetectedProcessors( void );

// GetTimesliceBaseline() returns a value that the ideal RC5 keyrate (kKeys 
// per Mhz) would be IF a machine were running at peak efficiency. For 
// non-preemptive systems, it is thus a good indicator of how low we can 
// set the timeslice/rate-of-yield without losing efficiency. Or inversely, 
// at what point OS responsiveness starts to suffer - which also applies to 
// preemptive but non-mt systems handling of a break request. 
//
// The function can also be used on non-mt systems to check for an excessive
// timeslice - on x86 systems an excessive timeslice is > 3*baseline

unsigned int GetTimesliceBaseline(void);

#endif /* __CPUCHECK_H__ */
