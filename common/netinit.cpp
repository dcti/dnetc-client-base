/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * -------------------------------------------------------------------
 * The Network constructor and destructor methods are encapsulated in
 * this module, thereby permitting us (a) to set up and tear down non-static
 * network connections on the fly, (b) to increase portability (c) to
 * simplify the code and eliminate multiple possible points of failure.
 * - cyp 08. Aug 1998
 * -------------------------------------------------------------------
*/
const char *netinit_cpp(void) {
return "@(#)$Id: netinit.cpp,v 1.26.2.6 2000/05/06 20:36:19 mfeiri Exp $"; }

#include "cputypes.h"
#include "baseincs.h"
#include "network.h"
#include "logstuff.h" //for messages
#include "clitime.h"  //for the time stamp string
#include "sleepdef.h" //for sleep();
#include "triggers.h" //for break checks
#include "lurk.h"

#if (CLIENT_OS == OS_AMIGAOS)
static struct Library *SocketBase;
#endif

//--------------------------------------------------------------------------

static unsigned int net_init_level = 0;

/*
  __netInitAndDeinit( ... ) combines both init and deinit so statics can
  be localized. The function is called with (> 0) to init, (< 0) to deinint
  and (== 0) to return the current 'isOK' state.
*/

static int __netInitAndDeinit( int doWhat )
{
  int success = 1;

  if (( doWhat < 0 ) && ( net_init_level == 0 ))
  {
    Log("Squawk! Unbalanced Network Init/Deinit!\n");
    return 0;
  }
  else if (( doWhat == 0 ) && ( net_init_level == 0 ))
    return 0;  //isOK() always returns 0 if we are not initialized

  //----------------------------

  #if (!defined(AF_INET) || !defined(SOCK_STREAM))
  if (success)  //no networking capabilities
  {
    if ( doWhat == 0 )     //query online mode
      return 0; //always fail - should never get called
    else if ( doWhat > 0)  //initialize request
      success = 0; //always fail
    else // (doWhat < 0)   //deinitialize request
      success = 1; //always succeed - should never get called
  }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_NETWARE)
  if (success)
  {
    if ( doWhat == 0 )     //query online mode
    {
      return nwCliIsNetworkAvailable(0);  //test if tcpip is still loaded
    }
    else if (doWhat > 0)   //init request
    {
      success = nwCliIsNetworkAvailable(0); //test if tcpip is loaded
      if (success)
        net_init_level++;
    }
    else                   //de-init request
    {
      success = 1;
      net_init_level--;
    }
  }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_WIN16)
  if (success)
  {
    if ( doWhat == 0 )                     //request to check online mode
    {
      return w32sockIsAlive();
    }
    else if (doWhat > 0)                  //request to initialize
    {
      if ((++net_init_level)!=1)     //don't initialize more than once
        success = 1;
      else if ((success = w32sockInitialize()) == 0)
        --net_init_level;
    }
    else if (doWhat < 0)
    {
      if ((--net_init_level)==0) //don't deinitialize more than once
        w32sockDeinitialize();
      success = 1;
    }
  }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_AMIGAOS)
  if (success)
  {
    //static struct Library *SocketBase;
    if ( doWhat == 0 )     //request to check online mode
    {
      return 1;            //assume always online once initialized
    }
    else if (doWhat > 0)   //request to initialize
    {
      if ((++net_init_level)!=1) //don't initialize more than once
        success = 1;
      else
      {
        #define SOCK_LIB_NAME "bsdsocket.library"
        SocketBase = OpenLibrary((unsigned char *)SOCK_LIB_NAME, 4UL);
        if (SocketBase)
          success = 1;
        else
        {
          LogScreen("Network::Failed to open " SOCK_LIB_NAME "\n");
          success = 0;
          net_init_level--;
        }
      }
    }
    else //if (doWhat < 0) //request to de-initialize
    {
      if ((--net_init_level)!=0) //don't deinitialize more than once
        success = 1;
      else
      {
        if (SocketBase)
        {
          CloseLibrary(SocketBase);
          SocketBase = NULL;
        }
        success = 1;
      }
    }
  }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------------------------

  #ifndef DOWHAT_WAS_HANDLED
  if (success)
  {
    if ( doWhat == 0 )     //request to check online mode
    {
      #if defined(LURK)
      if (dialup.CheckForStatusChange()) //- no longer-online?
        return 0;                        //oops, return 0
      #endif
      return 1;            //assume always online once initialized
    }
    else if (doWhat > 0)   //request to initialize
    {
      success = 1;
      if ((++net_init_level)==1) //don't initialize more than once
      {
        #if defined(LURK)
        if (dialup.DialIfNeeded(1) != 0 ) /* not connected and dialup failed */
          success = 0;
        #endif
      }
      if (!success)
        net_init_level--;
    }
    else //if (doWhat < 0) //request to de-initialize
    {
      success = 1;
      if ((--net_init_level)==0) //don't deinitialize more than once
      {
        #if defined(LURK)
        dialup.HangupIfNeeded();
        #endif
      }
    }
  }
  #endif

  //----------------------------

  return ((success) ? (0) : (-1));
}

//======================================================================

//  __globalInitAndDeinit() gets called once (to init) when the
// client starts and once (to deinit) when the client stops.

static int __globalInitAndDeinit( int doWhat )
{
  static unsigned int global_is_init = 0;
  int success = 1; //assume ok

  if (doWhat > 0)                            //initialize
  {
    if (global_is_init == 0)
    {
      global_is_init = 1; //assume all success

      #ifdef SOCKS
      LIBPREFIX(init)("rc5-client");
      #endif
      #if ((CLIENT_OS == OS_OS2) && !defined(__EMX__))
      sock_init();
      #endif
      #if (CLIENT_OS == OS_MACOS)
      if (socket_glue_init() != 0)
        global_is_init = 0;
      #endif
      #if (CLIENT_OS == OS_WIN32)
      WSADATA wsaData;
      if ( WSAStartup( 0x0101, &wsaData ) != 0 )
        global_is_init = 0;
      #endif
    }
    success = (global_is_init != 0);
  }
  else if (doWhat < 0)                      //deinitialize
  {
    if (global_is_init != 0)
    {
      global_is_init = 0; // assume all success
      
      #if (CLIENT_OS == OS_WIN32)
      WSACleanup();
      #elif (CLIENT_OS == OS_MACOS)
      global_is_init = 1; // a hack to prevent a 2nd global initialization
      // real global network deinitialization gets done at client shutdown
      #endif
    }
  }
  else //if (doWhat == 0)                   //query state
  {
    success = (global_is_init != 0);
  }
  return ((success == 0)?(-1):(0));
}

//======================================================================

int NetCheckIsOK(void)
{
  //get the (platform specific) network state
  return __netInitAndDeinit( 0 );
}

//----------------------------------------------------------------------

int NetClose( Network *net )
{
  if ( net )
  {
    delete net;

    // do platform specific network deinit
    return __netInitAndDeinit( -1 );
  }
  return 0;
}

//----------------------------------------------------------------------

Network *NetOpen( const char *servname, int servport,
           int _nofallback/*= 1*/, int _iotimeout/*= -1*/, int _enctype/*=0*/,
           const char *_fwallhost /*= NULL*/, int _fwallport /*= 0*/,
           const char *_fwalluid /*= NULL*/ )
{
  Network *net;
  int success;

  //check if connectivity has been initialized
  if (__globalInitAndDeinit( 0 ) < 0)
    return NULL;

  // do platform specific socket init
  if ( __netInitAndDeinit( +1 ) < 0)
    return NULL;

  net = new Network( servname, servport,
           _nofallback /*=1*/, _iotimeout/*=-1*/, _enctype /*= 0*/,
           _fwallhost /*= NULL*/, _fwallport /*= 0*/, _fwalluid /*= NULL*/ );
  success = ( net != NULL );

  if (success)
    success = ((net->Open()) == 0); //opened ok

  if (!success)
  {
    if (net)
      delete net;
    net = NULL;

    // do platform specific sock deinit
    __netInitAndDeinit( -1 );
  }

  return (net);
}

//----------------------------------------------------------------------

int InitializeConnectivity(void)
{
  __globalInitAndDeinit( +1 );
  return 0; //don't care about errors - NetOpen will handle that
}

int DeinitializeConnectivity(void)
{
  __globalInitAndDeinit( -1 );
  return 0;
}
