/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Log: base64.h,v $
 * Revision 1.1  1999/03/18 03:52:44  cyp
 * Split from confmenu.cpp so Client::Configure() doesn't need to be virtual
 * for clients that don't use it.
 *
 *
*/

#ifndef __BASE64_H__
#define __BASE64_H__ 

int base64_encode(char *outbuf, const char *inbuf );
int base64_decode(char *outbuf, const char *inbuf );

#endif /* __BASE64_H__ */

