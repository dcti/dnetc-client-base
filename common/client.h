// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

//
// ----------------------------------------------------------------------
// This file contains the basic types used in a lot of places: Client class;
// Operation, contest_id_t enums; Packet, FileHeader and FileEntry structs; 
// none of them depend on anything other than cputypes.h, and network.h
// ----------------------------------------------------------------------
//
// $Log: client.h,v $
// Revision 1.94.2.10  1999/02/04 23:21:43  remi
// Synced with :
//
//  Revision 1.115  1999/01/31 20:19:08  cyp
//  Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.94.2.9  1999/01/30 16:20:29  remi
// Fixed the previous merge.
//
// Revision 1.94.2.8  1999/01/30 15:59:50  remi
// Synced with :
//
//  Revision 1.114  1999/01/29 19:03:14  jlawson
//  fixed formatting.  changed some int vars to bool.
//
// Revision 1.94.2.7  1999/01/23 14:03:44  remi
// In sync with 1.113
//
// Revision 1.94.2.6  1999/01/17 12:25:59  remi
// In sync with 1.112
//
// Revision 1.94.2.5  1999/01/09 11:09:45  remi
// Fixed the previous merge.
//
// Revision 1.94.2.4  1999/01/04 02:06:39  remi
// Synced with :
//
//  Revision 1.111  1999/01/01 02:45:14  cramer
//  Part 1 of 1999 Copyright updates...
//
// Revision 1.94.2.3  1998/12/28 16:42:41  remi
// Synced with :
//   Revision 1.100  1998/11/28 19:44:34  cyp
//   InitializeLogging() and DeinitializeLogging() are no longer Client class
//   methods.
//
// Revision 1.94.2.2  1998/11/08 11:50:31  remi
// Lots of $Log tags.


#ifndef __CLIBASICS_H__
#define __CLIBASICS_H__

#include "cputypes.h"

class Client
{
public:
  s32  timeslice;
  s32  cputype;

//protected:
  u32 totalBlocksDone[2];
  u32 old_totalBlocksDone[2];

public:
  Client();
  ~Client() {};


  int Main( int argc, const char *argv[], int /*restarted*/ );
    // encapsulated main().  client.Main() may restart itself

  int ParseCommandline( int runlevel, int argc, const char *argv[], 
                        int *retcodeP, int logging_is_initialized );
                        
  //runlevel == 0 = ReadConfig() (-quiet, -ini, -guistart etc done here too)
  //         >= 1 = post-readconfig (override ini options)
  //         == 2 = run "modes"

  void ValidateConfig( void );
    // verifies configuration and forces valid values

  int SelectCore(int quietly);
    // always returns zero.
    // to configure for cpu. called before Run() from main(), or for 
    // "modes" (Benchmark()/Test()) from ParseCommandLine().

  unsigned int LoadSaveProblems(unsigned int load_problem_count, int retmode);
    // returns actually loaded problems 

  void Client::DisplayHelp( void );  
    // Display help text
};

// --------------------------------------------------------------------------

#endif // __CLIBASICS_H__

