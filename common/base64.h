/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 
#ifndef __BASE64_H__
#define __BASE64_H__ "@(#)$Id: base64.h,v 1.2 1999/04/06 10:20:47 cyp Exp $"

int base64_encode(char *outbuf, const char *inbuf );
int base64_decode(char *outbuf, const char *inbuf );

#endif /* __BASE64_H__ */
