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
return "@(#)$Id: netinit.cpp,v 1.26.2.12 2000/09/21 16:46:08 oliver Exp $"; }

//#define TRACE

#include "baseincs.h" //standard stuff
#include "util.h"     //trace
#include "logstuff.h" //for messages
#include "lurk.h"     //#ifdef LURK
#include "network.h"  //ourselves

//--------------------------------------------------------------------------

/*
  __netInitAndDeinit( ... ) combines both init and deinit so statics can
  be localized. The function is called with (> 0) to init, (< 0) to deinint
  and (== 0) to return the current 'isOK' state.
*/

static int __netInitAndDeinit( int doWhat )
{
  static int init_level = 0;
  int rc = 0; /* assume success */

  TRACE_OUT((+1,"__netInitAndDeinit(doWhat=%d)\n", doWhat));

  /* ----------------------- */

  if (rc == 0 && doWhat < 0)       //request to deinitialize
  {
    if (init_level == 0) /* ACK! */
    {
      Log("Beep! Beep! Unbalanced Network Init/Deinit!\n");
      rc = -1;
    }
    else if ((--init_level)==0)  //don't deinitialize more than once
    {
      #if defined(LURK) 
      // HangupIfNeeded will hang up a connection if previously 
      // initiated with DialIfNeeded(). Otherwise it does nothing.
      dialup.HangupIfNeeded(); 
      #endif
      #if (CLIENT_OS == OS_AMIGAOS)
      amigaSocketDeinit();
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
      if (winGetVersion() < 400) /* win16 or win32s */
        w32sockDeinitialize();
      #endif
    }
  }  

  /* ----------------------- */

  if (rc == 0 && doWhat > 0)  //request to initialize
  {
    if ((++init_level)!=1) //already initialized?
      doWhat = 0; //convert into request to check online mode
    else  
    {
      #if (!defined(AF_INET) || !defined(SOCK_STREAM))
        rc = -1;  //no networking capabilities
      #elif (CLIENT_OS == OS_AMIGAOS)
      #if defined(LURK)
      if (!amigaSocketInit(dialup.IsWatching())) // some libs not needed if lurking
      #else
      if (!amigaSocketInit(0))
      #endif
        rc = -1;
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
      if (rc == 0 && winGetVer() < 400) /* win16 or win32s */
        rc = ((!w32sockInitialize())?(-1):(rc));
      #endif    
      if (rc == 0)
        rc = __netInitAndDeinit(0); //recurse as request to check online mode
      if (rc != 0)
      {
        init_level--;
        #if (CLIENT_OS == OS_AMIGAOS)
        amigaSocketDeinit();
        #endif
      }
    }    
  } 

  /* ----------------------- */
  
  if ( rc == 0 && doWhat == 0 )     //request to check online mode
  {
    if (init_level == 0) /* ACK! haven't been initialized yet */
      rc = -1;
    else
    {  
      #if (!defined(AF_INET) || !defined(SOCK_STREAM))
      rc = -1;  //no networking capabilities
      #elif (CLIENT_OS == OS_WIN16) || (CLIENT_OS == OS_WIN32)
      if (rc == 0 && winGetVersion() < 400) /* win16 or win32s */
        rc = ((!w32sockIsAlive())?(-1):(rc));
      #elif(CLIENT_OS == OS_AMIGAOS)
      if (!amigaIsNetworkingActive())  //test if tcpip is still available
        rc = -1;
      #elif (CLIENT_OS == OS_NETWARE)  
      if (!FindNLMHandle("TCPIP.NLM")) /* tcpip is still loaded? */
        rc = -1;
      #endif    
    }	  

    #if defined(LURK)
    if (rc == 0)
    {
      // dialup.IsWatching() returns zero if 'dialup' isn't initialized.
      // Otherwise it returns a bitmask of things it is configured to do,
      // ie CONNECT_LURK|CONNECT_LURKONLY|CONNECT_DOD
      int confbits = dialup.IsWatching();
      if (confbits) /* 'dialup' initialized and have LURK[ONLY] and/or DOD */
      {       
        // IsConnected() returns non-zero when 'dialup' is initialized and
        // a link is up. Otherwise it returns zero.
        if (!dialup.IsConnected()) /* no longer online? */
        {
	  rc = -1; /* conn dropped and assume not restartable */
	  if ((confbits & CONNECT_DOD)!=0) /* configured for dial-on-demand?*/
	  {                              
            // DialIfNeeded(1) returns zero if already connected OR 
            // not-configured-for-dod OR dial success. Otherwise it returns -1 
            // (either 'dialup' isn't initialized or dialing failed).
            // Passing '1' makes it ignore any lurkonly restriction.
    	    if (dialup.DialIfNeeded(1) == 0) /* reconnect to complete */
  	    {                                /* whatever we were doing */
	      rc = 0; /* (re-)dial was successful */
	    }
          }
        } /* !dialup.IsConnected() */
      } /* if dialup.IsWatching() */
    }	/* if (rc == 0) */
    #endif /* if defined(LURK) */
  } /* if ( rc == 0 && doWhat == 0 ) */

  /* ----------------------- */

  TRACE_OUT((-1,"__netInitAndDeinit() => %d\n", rc));
  return rc;
}

//======================================================================

//  __globalInitAndDeinit() gets called once (to init) when the
// client starts and once (to deinit) when the client stops/restarts.

static int __globalInitAndDeinit( int doWhat )
{
  static unsigned int global_is_init = 0;
  int rc = 0; //assume ok

  TRACE_OUT((+1,"__globalInitAndDeinit(doWhat=%d)\n", doWhat));
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
      #if (CLIENT_OS == OS_WIN32)
      WSADATA wsaData;
      if ( WSAStartup( 0x0101, &wsaData ) != 0 )
        global_is_init = 0;
      #endif
    }
    if (global_is_init == 0)
      rc = -1;
  }
  else if (doWhat < 0)                      //deinitialize
  {
    if (global_is_init != 0)
    {
      global_is_init = 0; // assume all success
  
      #if (CLIENT_OS == OS_WIN32)
      WSACleanup();
      #endif
    }
  }
  else //if (doWhat == 0)                   //query state
  {
    if (global_is_init == 0)
      rc = -1;
  }
  TRACE_OUT((-1,"__globalInitAndDeinit() => %d\n", rc));
  return rc;
}

//======================================================================

int NetCheckIsOK(void)
{
  //get the (platform specific) network state
  int isok = 0; /* !OK */
  TRACE_OUT((+1,"NetCheckIsOK(void)\n"));
  if (__netInitAndDeinit( 0 ) == 0)
    isok = 1; /* OK */
  TRACE_OUT((-1,"NetCheckIsOK(void)=>%d\n",isok));
  return isok;
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
  Network *net = (Network *)0;
  TRACE_OUT((+1,"NetOpen(...)\n"));
  if (__globalInitAndDeinit( 0 ) == 0) //has global connectivity been initialized?
  {
    if ( __netInitAndDeinit( +1 ) == 0) // do platform specific socket init
    {
      net = new Network( servname, servport,
           _nofallback /*=1*/, _iotimeout/*=-1*/, _enctype /*= 0*/,
           _fwallhost /*= NULL*/, _fwallport /*= 0*/, _fwalluid /*= NULL*/ );
      TRACE_OUT((0,"new Network => %p\n", net));
      if (net)
      {           
        if ((net->OpenConnection()) != 0)
        {
          delete net;
          net = (Network *)0;
        }
        TRACE_OUT((0,"net->OpenConnection() => %s\n",((net)?("ok"):("err"))));
      }    
      if (!net)
      {
        __netInitAndDeinit( -1 );
      }
    }  
  }    
  TRACE_OUT((-1,"NetOpen(...) => %s\n",((net)?("ok"):("failed"))));
  return net;
}

//----------------------------------------------------------------------

int InitializeConnectivity(void)
{
  TRACE_OUT((+1,"InitializeConnectivity(void)\n"));
  __globalInitAndDeinit( +1 );
  TRACE_OUT((-1,"InitializeConnectivity(void) => 0\n"));
  return 0; //don't care about errors - NetOpen will handle that
}

int DeinitializeConnectivity(void)
{
  TRACE_OUT((+1,"DeinitializeConnectivity(void)\n"));
  __globalInitAndDeinit( -1 );
  TRACE_OUT((-1,"DeinitializeConnectivity(void) => 0\n"));
  return 0;
}
