// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: confrwv.h,v $
// Revision 1.7  1999/01/21 21:49:02  cyp
// completed toss of ValidateConfig().
//
// Revision 1.6  1999/01/04 02:47:30  cyp
// Cleaned up menu options and handling.
//
// Revision 1.4  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.3  1998/12/28 03:32:47  silby
// WIN32GUI internalread/writeconfig procedures are back.
//
// Revision 1.2  1998/12/25 02:32:11  silby
// ini writing functions are now not part of client object.
// This allows the win32 (and other) guis to have
// configure modules that act on a dummy client object.
// (Client::Configure should be seperated as well.)
// Also fixed bug with spaces being taken out of pathnames.
//
// Revision 1.1  1998/12/23 03:24:56  silby
// Client once again listens to keyserver for next contest start time,
// tested, it correctly updates.  Restarting after des blocks have
// been recieved has not yet been implemented, I don't have a clean
// way to do it yet.  Writing of contest data to the .ini has been
// moved back to confrwv with its other ini friends.
//

#ifndef __CONFRWV_H__
#define __CONFRWV_H__

int ReadConfig(Client *client);
int WriteConfig(Client *client, int writefull /* defaults to 0*/);
void RefreshRandomPrefix( Client *client, int no_trigger = 0 );

#if defined(WIN32GUI)
void InternalReadConfig( IniSection &ini );
void InternalWriteConfig( IniSection &ini );
#endif

#endif // __CONFRWV_H__
