// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.

#include "cputypes.h"
#include "console.h"
#include "client.h"   // MAXCPUS, Packet, FileHeader, Client class, etc
#include "baseincs.h" // basic (even if port-specific) #includes
#include "version.h"
#include "triggers.h" //[Check|Raise][Pause|Exit]RequestTrigger()/InitXHandler()
#include "logstuff.h"  //Log()/LogScreen()/LogScreenPercent()/LogFlush()
#include "selcore.h"   //SelectCore() and GetCoreNameFromCoreType()

#if (CLIENT_OS == OS_WIN32)
  #include "sleepdef.h" //used by RunStartup()
#endif


// --------------------------------------------------------------------------

void Client::ValidateConfig( void ) //DO NOT PRINT TO SCREEN HERE!
{

#if (CLIENT_OS == OS_RISCOS)
  if ( timeslice < 1 ) timeslice = 2048;
#else
  if ( timeslice < 1 ) timeslice = 65536;
#endif

  //validate numcpu is now in SelectCore(); //1998/06/21 cyrus

}
