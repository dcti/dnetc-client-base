// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/* 
   The Network constructor and destructor methods are encapsulated in
   this module, thereby permitting us (a) to set up and tear down non-static 
   network connections on the fly, (b) to increase portability (c) to 
   simplify the code and eliminate multiple possible points of failure.
   - cyp 08. Aug 1998
*/
//
// $Log: netinit.cpp,v $
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
return "@(#)$Id: netinit.cpp,v 1.12 1998/12/31 17:55:50 cyp Exp $"; }
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
  static unsigned int initializationlevel=0;
  int success = 1;

  if (( doWhat < 0 ) && ( initializationlevel == 0 ))
    {
    Log("Squawk! Unbalanced Network Init/Deinit!\n");
    abort();
    }
  else if (( doWhat == 0 ) && ( initializationlevel == 0 ))
    return 0;  //isOK() always returns false if we are not initialized

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
      return true;			// for now, always online
      }
    else if (doWhat > 0)   //init request
      {
      success = myNetInit();
      if (success)
        ++initializationlevel;
      }
    else                   //de-init request
      {
      success = 1;
      --initializationlevel;
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
        ++initializationlevel;
      }
    else                   //de-init request
      {
      success = 1;
      --initializationlevel;
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
        return 0;                          //oops, return false
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
        return 0;                        //oops, return false
      #endif
      return 1;                          //yeah, we are still online
      }
    else if (doWhat > 0)                 //request to initialize
      {
      if ((++initializationlevel)!=1) //don't initialize more than once
        success = 1;
      else
        {
        sock_init();
        success = 1;

        #if defined(LURK)
        if ( dialup.DialIfNeeded(1) < 0 )
          {
          success = 0;           //FIXME: sock_deinit()? missing (?)
          } 
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

  return ((success)?(0):(-1));
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
    //do (platform specific) network deinit
    return __netInitAndDeinit( -1 ); 
    }
  return 0;
}    

//----------------------------------------------------------------------

// prototype is: 
// ork *NetOpen(const char *keyserver, s32 keyserverport, int nofallback = 1, 
//       int autofindks = 0, int iotimeout = -1, s32 proxytype = 0, 
//       const char *proxyhost = NULL, s32 proxyport = 0, 
//       const char *proxyuid = NULL)

Network *NetOpen(const char *keyserver, s32 keyserverport, int nofallback, 
          int autofindks, int iotimeout, s32 proxytype, const char *proxyhost, 
          s32 proxyport, const char *proxyuid)
{
  Network *net;
  int success;
  
  // do platform specific socket init
  if ( __netInitAndDeinit( +1 ) < 0)
    return NULL; 

  net = new Network( ((keyserver && !*keyserver)?(NULL):(keyserver)), 
                     (s16)keyserverport, nofallback, autofindks, iotimeout );
  success = ( net != NULL );
    
  if (success)
    {
LogScreen("hostname:%s:%d proxyuid:'%s'\n", proxyhost, proxyport,proxyuid);
    switch (proxytype)
      {
      case 1:  // uue
        net->SetModeUUE(true);
        break;
      case 2:  // http
        net->SetModeHTTP(proxyhost, (s16) proxyport, proxyuid);
        break;
      case 3:  // uue + http
        net->SetModeHTTP(proxyhost, (s16) proxyport, proxyuid);
        net->SetModeUUE(true);
        break;
      case 4:  // SOCKS4
        net->SetModeSOCKS4(proxyhost, (s16) proxyport, proxyuid);
        break;
      case 5:  // SOCKS5
        net->SetModeSOCKS5(proxyhost, (s16) proxyport, proxyuid);
        break;
      }
    }

  if (success)    
    {    
    success = ((net->Open()) == 0); //opened ok
    }
            
  if (!success)
    {
    if (net)
      delete net;
    net = NULL;
    //do platform specific sock deinit
    __netInitAndDeinit( -1 ); 
    }

  return (net);
}  

//----------------------------------------------------------------------
