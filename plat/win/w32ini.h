/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
*/ 

#ifndef __W32CONF_H__
#define __W32CONF_H__ "@(#)$Id: w32ini.h,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $"

/* the next four funcs follow the same format at [Get|Write]ProfileString().
   For win32 (registry) HKLM\SoftWare\D C T I\sect.key = value
   For win16 (win.ini)               [D C T I]sect.key = value 
   ('D C T I' expands to "Distributed Computing Technologies, Inc.")
   'sect' is optional. If NULL or "", the format becomes key=value.
*/   
extern int WriteDCTIProfileString( const char *sect, const char *key, 
                                   const char *val);
extern unsigned int GetDCTIProfileString(const char *sect, const char *key, 
                                   const char *defaultval, char *buf, 
                                   unsigned int bufsize );
extern int WriteDCTIProfileInt(const char *sect, const char *entry, 
                                   int val );
extern int GetDCTIProfileInt(const char *sect, const char *entry, 
                                   int defaultval);

/* use something other than "Distributed Computing Technologies, Inc." */
extern const char *SetDCTIProfileContext(const char *ctx);

#endif /* __W32CONF_H__ */
