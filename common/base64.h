/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
*/ 

#ifndef __BASE64_H__
#define __BASE64_H__ "@(#)$Id: base64.h,v 1.3 1999/08/09 17:01:49 cyp Exp $"

/* 
  On success, both functions return the number of bytes in the outbuf (not 
  counting the terminating '\0'). On error, both return < 0.
  note: The outbuffer is terminated only if there is sufficient space 
  remaining after conversion.
*/ 

int base64_encode(char *outbuf, const char *inbuf, 
                  unsigned int outbuflen, unsigned int inbuflen );
int base64_decode(char *outbuf, const char *inbuf, 
                  unsigned int outbuflen, unsigned int inbuflen );

#endif /* __BASE64_H__ */
