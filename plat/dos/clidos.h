/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * -----------------------------------------------------------------
 * this yanks in all the other non-standard or dos-port specific .h files
 * -----------------------------------------------------------------
*/

#ifndef __CLIDOS_H__
#define __CLIDOS_H__ "@(#)$Id: clidos.h,v 1.1.2.1 2001/01/21 15:10:23 cyp Exp $"

//#include <sys/timeb.h>
#include <io.h>
#include <conio.h>
#include <share.h>
#include <fcntl.h>
#include <sys/stat.h>

#include "cdosemu.h"
#include "cdoscon.h"
#include "cdosidle.h"
#include "cdostime.h"
#include "cdosinet.h"

#define dosCliSleep(x)   delay((x)*1000)
#define dosCliUSleep(x)  delay((x)/1000)
#define dosCliYield(x)   delay(0) /* sleep/usleep may be redefined */

#endif //__CLIDOS_H__
