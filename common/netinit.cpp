// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
// 
// The Network constructor and destructor methods are encapsulated in
// this module, thereby permitting us (a) to set up and tear down non-static 
// network connections on the fly, (b) to increase portability (c) to 
// simplify the code and eliminate multiple possible points of failure.
// - cyp 08. Aug 1998
//
//
// $Log: netinit.cpp,v $
// Revision 1.20  1999/02/01 08:19:59  silby
// Network class once again allows autofindkeyserver to be disabled.
//
// Revision 1.19  1999/01/31 20:19:09  cyp
// Discarded all 'bool' type wierdness. See cputypes.h for explanation.
//
// Revision 1.18  1999/01/29 18:53:53  jlawson
// fixed formatting.  changed some int vars to bool.
//
// Revision 1.17  1999/01/29 04:12:37  cyp
// NetOpen() no longer needs the autofind setting to be passed from the client.
//
// Revision 1.16  1999/01/23 21:25:40  patrick
// sock_init not used by OS2-EMX
//
// Revision 1.15  1999/01/01 02:45:15  cramer
// Part 1 of 1999 Copyright updates...
//
// Revision 1.14  1999/01/01 01:38:08  cramer
// use explicit casts... NULL ain't always 'char *'.
//
// Revision 1.13  1998/12/31 19:58:13  silby
// Commented out a line of debug code.
//
// Revision 1.12  1998/12/31 17:55:50  cyp
// changes to Network::Open(): (a) retry loop is inside ::Open() (was from
// the external NetOpen()) (b) cleaned up the various hostname/addr/port
// variables to make sense and be uniform throughout. (c) nofallback handling
// is performed by ::Open() and not by the external NetOpen().
//
// Revision 1.11  1998/12/21 17:54:23  cyp
// (a) Network connect is now non-blocking. (b) timeout param moved from
// network::Get() to object scope.
//
// Revision 1.10  1998/12/08 05:49:51  dicamillo
// Add initialization for MacOS networking.
//
// Revision 1.9  1998/10/26 02:55:03  cyp
// win16 changes
//
// Revision 1.8  1998/09/28 20:27:54  remi
// Cleared a warning.
//
// Revision 1.7  1998/09/20 15:23:26  blast
// AmigaOS changes
//
// Revision 1.6  1998/08/28 21:41:37  cyp
// Stopped network->Open() from being retried in a no-network environment.
//
// Revision 1.5  1998/08/25 08:21:33  cyruspatel
// Removed the default values from the declaration of NetOpen().
//
// Revision 1.4  1998/08/25 00:01:16  cyruspatel
// Merged (a) the Network destructor and DeinitializeNetwork() into NetClose()
// (b) the Network constructor and InitializeNetwork() into NetOpen().
// These two new functions (in netinit.cpp) are essentially what the static
// FetchFlush[Create|Destroy]Net() functions in buffupd.cpp used to be.
//
// Revision 1.3  1998/08/24 07:09:19  cyruspatel
// Added FIXME comments for "lurk"ers.
//
// Revision 1.2  1998/08/20 19:27:16  cyruspatel
// Made the purpose of NetworkInitialize/Deinitialize a little more
// transparent.
//
// Revision 1.1  1998/08/10 21:53:55  cyruspatel
// Created - see documentation above.
//

#if (!defined(lint) && defined(__showids__))
const char *netinit_cpp(void) {
return "@(#)$Id: netinit.cpp,v 1.20 1999/02/01 08:19:59 silby Exp $"; }
#endif

//--------------------------------------------------------------------------

#include "cputypes.h"
#include "network.h"
#include "logstuff.h" //for messages
#include "clitime.h"  //for the time stamp string
#include "sleepdef.h" //for sleep();
#include "triggers.h" //for break checks
#include "lurk.h"

#if (CLIENT_OS == OS_MACOS)
  Boolean myNetInit(void);
#endif
//--------------------------------------------------------------------------

/*
  __netInitAndDeinit( ... ) combines both init and deinit so statics can
  be localized. The function is called with (> 0) to init, (< 0) to deinint 
  and (== 0) to return the current 'isOK' state.
*/

#if (CLIENT_OS == OS_AMIGAOS)
static struct Library *SocketBase;
#endif

static int __netInitAndDeinit( int doWhat )  
{                                            
  static unsigned int initializationlevel = 0;
  int success = 1;

  if (( doWhat < 0 ) && ( initializationlevel == 0 ))
    {
    Log("Squawk! Unbalanced Network Init/Deinit!\n");
    abort();
    }
  else if (( doWhat == 0 ) && ( initializationlevel == 0 ))
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

  #if (CLIENT_OS == OS_MACOS)
  if (success)
    {
    if ( doWhat == 0 )     //query online mode
      {
      return 1;			// for now, always online
      }
    else if (doWhat > 0)   //init request
      {
      success = myNetInit();
      if (success)
        initializationlevel++;
      }
    else                   //de-init request
      {
      success = 1;
      initializationlevel--;
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_NETWARE)
  if (success)
    {
    if ( doWhat == 0 )     //query online mode
      {
      return nwCliIsNetworkAvailable(0);  //test if still online
      }
    else if (doWhat > 0)   //init request
      {
      success = nwCliIsNetworkAvailable(0); //test if online
      if (success)
        initializationlevel++;
      }
    else                   //de-init request
      {
      success = 1;
      initializationlevel--;
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32S)
  if (success)
    {
    if ( doWhat == 0 )                     //request to check online mode
      {
      return w32sockIsAlive();
      }
    else if (doWhat > 0)                  //request to initialize
      {
      if ((++initializationlevel)!=1)     //don't initialize more than once
        success = 1;
      else if ((success = w32sockInitialize()) == 0)
        --initializationlevel;
      }
    else if (doWhat < 0)
      {
      if ((--initializationlevel)==0) //don't deinitialize more than once
        w32sockDeinitialize();
     success = 1;
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS==OS_WIN32) 
  if (success)
    {
    if ( doWhat == 0 )                     //request to check online mode
      {
      //- still-online? 
      #if defined(LURK)
      if (dialup.CheckForStatusChange() == -1)
        return 0;                          //oops, return 0
      #endif
      return 1;                           //yeah, we are still online
      }
    else if (doWhat > 0)                  //request to initialize
      {
      if ((++initializationlevel)!=1)     //don't initialize more than once
        success = 1;
      else
          {
        WSADATA wsaData;
        success = 0;
        if ( WSAStartup( 0x0101, &wsaData ) == 0 )
          {
          success = 1; 
          #if defined(LURK)
          if ( dialup.DialIfNeeded(1) < 0 )
            {
            success = 0;
            WSACleanup();  // cleanup because we won't be called to deinit
            }
          #endif
          }
        if (!success)
          {
          --initializationlevel;
          }
        }
      }
    else //if (doWhat < 0) //request to de-initialize
      {
      if ((--initializationlevel)!=0) //don't deinitialize more than once
        success = 1;
      else
        {
        #if defined(LURK)
        dialup.HangupIfNeeded();
        #endif

        WSACleanup();
        success = 1;
        }
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_OS2)
  if (success)
    {
    if ( doWhat == 0 )                   //request to check online mode
      {
      //- still-online? 
      #if defined(LURK)
      if (dialup.CheckForStatusChange() == -1)
        return 0;                        //oops, return 0
      #endif
      return 1;                          //yeah, we are still online
      }
    else if (doWhat > 0)                 //request to initialize
      {
      if ((++initializationlevel)!=1) //don't initialize more than once
        success = 1;
      else
        {
    #if !defined(__EMX__)
        sock_init();
    #endif
        success = 1;

        #if defined(LURK)
        if ( dialup.DialIfNeeded(1) < 0 )
          success = 0;           //FIXME: sock_deinit()? missing (?)
        #endif
        }
      }
    else //if (doWhat < 0) //request to de-initialize
      {
      if ((--initializationlevel)!=0) //don't deinitialize more than once
        success = 1;
      else
        {
        #if defined(LURK)
        dialup.HangupIfNeeded();
        #endif
                                //FIXME: sock_deinit()? missing (?) 
        success = 1;
        }
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------------------------

  #if (CLIENT_OS == OS_AMIGAOS)
  if (success)
    {
//    static struct Library *SocketBase;
    if ( doWhat == 0 )     //request to check online mode
      {
      return 1;            //assume always online once initialized
      }
    else if (doWhat > 0)   //request to initialize
      {
      if ((++initializationlevel)!=1) //don't initialize more than once
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
          initializationlevel--;
          }
        }
      }
    else //if (doWhat < 0) //request to de-initialize
      {
      if ((--initializationlevel)!=0) //don't deinitialize more than once
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
      return 1;            //assume always online once initialized
      }
    else if (doWhat > 0)   //request to initialize
      {
      #ifdef SOCKS
      LIBPREFIX(init)("rc5-client");
      #endif
      initializationlevel++;
      success = 1;
      }
    else //if (doWhat < 0) //request to de-initialize
      {
      initializationlevel--;
      success = 1;
      }
    }
  #endif

  //----------------------------

  return ((success) ? (0) : (-1));
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
           int _nofallback/*= 1*/, int _noautofindkeyserver /*=0*/,
           int _iotimeout/*= -1*/, int _enctype/*=0*/, 
           const char *_fwallhost /*= NULL*/, int _fwallport /*= 0*/, 
           const char *_fwalluid /*= NULL*/ )
{
  Network *net;
  int success;
  
  // do platform specific socket init
  if ( __netInitAndDeinit( +1 ) < 0)
    return NULL; 

  net = new Network( servname, servport, 
           _nofallback /*=1*/, _noautofindkeyserver/*=0*/,
           _iotimeout/*=-1*/, _enctype /*= 0*/, 
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

