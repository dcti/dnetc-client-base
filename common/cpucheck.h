/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __CPUCHECK_H__
#define __CPUCHECK_H__ "@(#)$Id: cpucheck.h,v 1.13 2000/06/02 06:24:56 jlawson Exp $"

//return number of processors detected (by the hardware/from the OS)
//returns -1 if detection is not supported.
int GetNumberOfDetectedProcessors( void );

//get cpu ident by hardware detection
long GetProcessorType(int quietly);

//Return cpuid/cputag, maxcpus, foundcpus as descriptive strings.
//Assists in debugging user complaints/bug reports.
void GetProcessorInformationStrings( const char ** scpuid, 
                  const char ** smaxcpus, const char ** sfoundcpus );

//Wrapper for GetProcessorInformationStrings()
//(used to be a client class function for access to the log functions)
void DisplayProcessorInformation( void );

#endif /* __CPUCHECK_H__ */
