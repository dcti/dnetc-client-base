// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: buffupd.h,v $
// Revision 1.2  1998/12/20 23:00:35  silby
// Descontestclosed value is now stored and retrieved from the ini file,
// additional updated of the .ini file's contest info when fetches and
// flushes are performed are now done.  Code to throw away old des blocks
// has not yet been implemented.
//
// Revision 1.1  1998/11/26 07:19:09  cyp
// Created. blah.
//
//
#ifndef __BUFFUPD_H__
#define __BUFFUPD_H__

#define BUFFERUPDATE_FETCH 0x01
#define BUFFERUPDATE_FLUSH 0x02

void __RefreshRandomPrefix( Client *client );

#endif /* __BUFFUPD_H__ */
