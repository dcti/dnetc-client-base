// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cpucheck.h,v $
// Revision 1.6  1998/10/11 00:43:22  cyp
// Implemented 'quietly' in SelectCore() and ValidateProcessorCount()
//
// Revision 1.5  1998/10/09 12:25:27  cyp
// ValidateProcessorCount() is no longer a client method [is now standalone].
//
// Revision 1.4  1998/10/08 10:04:23  cyp
// GetProcessorType() is now standalone (no longer a Client::method).
//
// Revision 1.3  1998/09/28 02:49:12  cyp
// Added prototypes for GetNumberOf[Supported|Detected]Processors()
//
// Revision 1.2  1998/07/13 23:39:30  cyruspatel
// Added functions to format and display raw cpu info for better management
// of the processor detection functions and tables. Well, not totally raw,
// but still less cooked than SelectCore(). All platforms are supported, but
// the data may not be meaningful on all. The info is accessible to the user
// though the -cpuinfo switch.
//
// Revision 1.1  1998/07/05 12:42:43  cyruspatel
// Created cpucheck.h to support makefiles that rely on autodependancy info
// to detect file changes.
//
//

#ifndef __CPUCHECK_H__
#define __CPUCHECK_H__

//get core type by hardware detection
int GetProcessorType(int quietly);

//-------

//returns the number of cpus (>=0). 
//Zero is valid and symbolizes 'no-multithreading'.
unsigned int ValidateProcessorCount(int numcpu, int quietly);

//-------

//Return cpuid/cputag, maxcpus, foundcpus as descriptive strings.
//Assists in debugging user complaints/bug reports.
void GetProcessorInformationStrings( const char ** scpuid, 
                  const char ** smaxcpus, const char ** sfoundcpus );

//-------

//Wrapper for GetProcessorInformationStrings()
//(used to be a client class function for access to the log functions)
void DisplayProcessorInformation( void );

//-------

//returns number of processors supported by the client.
//should this be in client.cpp?
unsigned int GetNumberOfSupportedProcessors( void );

//------

//return number of processors detected (by the hardware/from the OS)
//returns -1 if detection is not supported.
int GetNumberOfDetectedProcessors( void );

// ------

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

// ------

#endif
