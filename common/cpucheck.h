// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: cpucheck.h,v $
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

void GetProcessorInformationStrings( const char ** scpuid, const char ** smaxcpus, const char ** sfoundcpus );
void DisplayProcessorInformation( void );

// --------------------------------------------------------------------------

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

// --------------------------------------------------------------------------

#endif
