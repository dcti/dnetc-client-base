/* Copyright distributed.net 1997-1999 - All Rights Reserved			*/
/* For use in distributed.net projects only.                         */
/* Any other distribution or use of this source violates copyright.  */
/*                                                                   */
/* @(#)$Id: version.r,v 1.1.2.1 2001/01/21 15:14:28 cyp Exp $     */

#include "Types.r"
#include "SysTypes.r"
#include "version.h"

#define the_copystr "©1997-2001 distributed.net"
#define the_stage final 

resource 'vers' (1, preload) {
  CLIENT_MAJOR_VER_HEX, 
  CLIENT_CONTEST_HEX,
  the_stage,
  CLIENT_BUILD_HEX,
  verUs,
  CLIENT_VERSIONSTRING, 
  CLIENT_VERSIONSTRING, 
};

resource 'vers' (2, preload) {
  CLIENT_MAJOR_VER_HEX,
  CLIENT_CONTEST_HEX,
  the_stage,
  CLIENT_BUILD_HEX,
  verUs,
  CLIENT_VERSIONSTRING,
  the_copystr
};

#ifndef MAC_FBA

resource 'DLOG' (128, "About", preload) {
	{0, 0, 180, 400},
//	{0, 0, 75, 500},
	dBoxProc,
 	visible,
 	goAway,
 	0x0,
 	128,
	"About",
	kWindowCenterMainScreen
};


resource 'DITL' (128, "About", preload) {
	{
	
		{150, 171, 170, 229},
		Button {
			enabled,
			"OK"
		}
	}
};


#endif