/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * -----------------------------------------------------------------
 * this .h yanks in all the other non-standard or dos-port specific 
 * header files
 * -----------------------------------------------------------------
 *
*/
#ifndef __CLIDOSCON_H__
#define __CLIDOSCON_H__ "@(#)$Id: cdoscon.h,v 1.1.2.1 2001/01/21 15:10:18 cyp Exp $"

extern int dosCliConIsScreen(void);
extern int dosCliConGetPos( int *colP, int *rowP );
extern int dosCliConSetPos( int acol, int arow );
extern int dosCliConGetSize( int *cols, int *rows );
extern int dosCliConClear(void);

#endif /* __CLIDOSCON_H__ */
