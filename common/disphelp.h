// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// $Log: disphelp.h,v $
// Revision 1.1  1998/11/08 19:01:07  cyp
// Removed lots and lots of junk; DisplayHelp() is no longer a client method;
// unix-ish clients no longer use the internal pager.
//
//

#ifndef __DISPHELP_H__
#define __DISPHELP_H__

/* provide a full-screen, interactive help for an invalid option (argv[x])
** 'unrecognized_option' may be NULL or a null string
*/
void DisplayHelp( const char * unrecognized_option );

#endif /* __DISPHELP_H__ */
