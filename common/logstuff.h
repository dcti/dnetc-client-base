// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: logstuff.h,v $
// Revision 1.1  1998/08/02 16:00:44  cyruspatel
// Created. Check the FIXMEs! Please get in touch with me before implementing
// support for the extended file logging types (rotate/fifo/restart types).
//
//

#ifndef __LOGSTUFF_H__
#define __LOGSTUFF_H__

#if (defined(NEEDVIRTUALMETHODS))  // gui clients have this elsewhere
extern void InternalLogScreen( const char *msgbuffer, unsigned int msglen );
#endif

extern void LogFlush( int forceflush );
extern void LogScreen( const char *format, ... ); //identical to LogScreenf()
//extern void LogScreenf( const char *format, ... );
extern void Log( const char *format, ... );
extern void LogScreenPercent( unsigned int load_problem_count );

extern void CliScreenClear( void );  // SLIGHTLY out of place.... :) - cyp

#endif //__LOGSTUFF_H__

