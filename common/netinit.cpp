// Hey, Emacs, this a -*-C++-*- file !

// Copyright distributed.net 1997-1998 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
/* 
   NetworkInitialize() is no longer a one-shot-and-be-done-with-it affair.
   I had to do this to work around the lack of definition of what
   client.offlinemode is, and the lack of a method to detect if the network 
   availability state had changed (or existed to begin with). The file
   expects to be #included from network.cpp 
 
   Expected usage: Every construction of a new Network object *must* be 
   preceded with a call to NetworkInitialize() which returns a value < 0 if
   network services are not available. Call NetworkDeinitialize when finished 
   with the object. Do not call NetworkDeinitialize() if initialization 
   failed. Do not assume that if one NetworkInitialize() succeeds, the next
   one will too. Calls to the functions can (obviously) be nested. 
   - cyp 08. Aug 1998
*/
//
// $Log: netinit.cpp,v $
// Revision 1.3  1998/08/24 07:09:19  cyruspatel
// Added FIXME comments for "lurk"ers.
//
// Revision 1.2  1998/08/20 19:27:16  cyruspatel
// Made the purpose of NetworkInitialize/Deinitialize a little more
// transparent.
//
// Revision 1.1  1998/08/10 21:53:55  cyruspatel
// Created - see documentatin above.
//

#if (!defined(lint) && defined(__showids__))
const char *netinit_cpp(void) {
return "@(#)$Id: netinit.cpp,v 1.3 1998/08/24 07:09:19 cyruspatel Exp $"; }
#endif

//--------------------------------------------------------------------------

static int netInitAndDeinit( int doWhat )     //combines both init and deinit
{                                             //so statics can be localized
  static unsigned int initializationlevel=0;
  int success = 1;

  if (( doWhat < 0 ) && ( initializationlevel == 0 ))
    {
    Log("[%s] Squawk! Unbalanced Network Init/Deinit!\n",Time());
    abort();
    }

  //---------------------------

  if (initializationlevel == 0)  // cross-platform stuff done at top level
    {
    if ( doWhat == 0 )           //isAvaliable() check
      return 0;
    else if ( doWhat > 1)        //request for initialization
      {
      size_t dummy;
      if (((dummy = offsetof(SOCKS4, USERID[0])) != 8) ||
          ((dummy = offsetof(SOCKS5METHODREQ, Methods[0])) != 2) ||
          ((dummy = offsetof(SOCKS5METHODREPLY, end)) != 2) ||
          ((dummy = offsetof(SOCKS5USERPWREPLY, end)) != 2) ||
          ((dummy = offsetof(SOCKS5, end)) != 10))
        {
        LogScreenf("[%s] Network::Socks Incorrectly packed structures.\n",Time());
        success = 0;  
        #if 0
        // check that the packet structures have been correctly packed
        // do it here to make sure the asserts go off
        // if all is correct, the asserts should get totally optimised out :)
        assert(offsetof(SOCKS4, USERID[0]) == 8);
        assert(offsetof(SOCKS5METHODREQ, Methods[0]) == 2);
        assert(offsetof(SOCKS5METHODREPLY, end) == 2);
        assert(offsetof(SOCKS5USERPWREPLY, end) == 2);
        assert(offsetof(SOCKS5, end) == 10);
        #endif
        }
      }
    }

  //----------------------------

  #ifdef STUBIFY_ME
  if (success)
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

  #if (CLIENT_OS == OS_WIN32) 
  if (success)
    {
    if ( doWhat == 0 )     //request to check online mode
      {
      //----------------------- add still-online? check here --------------
      return 1;            //always online once initialized?  //FIXME!!
      }
    else if (doWhat > 0)   //request to initialize
      {
      if ((++initializationlevel)!=1) //don't initialize more than once
        success = 1;
      else
        {
        WSADATA wsaData;
        WSAStartup(0x0101, &wsaData);
        success = 1;         // --- dialup initialize goes here -- FIXME!!
        }
      }
    else //if (doWhat < 0) //request to de-initialize
      {
      if ((--initializationlevel)!=0) //don't deinitialize more than once
        success = 1;
      else
        {
        WSACleanup();
        success = 1;                  // --- hangup goes here -- FIXME!!
        }
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------

  #if (CLIENT_OS == OS_OS2)
  if (success)
    {
    if ( doWhat == 0 )     //request to check online mode
      {
      //----------------------- add still-online? check here --------------
      return 1; //always online once initialized?                //FIXME!!
      }
    else if (doWhat > 0)   //request to initialize
      {
      if ((++initializationlevel)!=1) //don't initialize more than once
        success = 1;
      else
        {
        sock_init();
        success = 1;         // --- dialup initialize goes here -- FIXME!!
        }
      }
    else //if (doWhat < 0) //request to de-initialize
      {
      if ((--initializationlevel)!=0) //don't deinitialize more than once
        success = 1;
      else
        {
        //nothing else to do
        success = 1;                  // --- hangup goes here -- FIXME!!
        }
      }
    }
  #define DOWHAT_WAS_HANDLED
  #endif

  //----------------------------------------------

  #if (CLIENT_OS == OS_AMIGAOS)
  if (success)
    {
    static struct Library *SocketBase;
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
          LogScreen("[%s] Network::Failed to open " SOCK_LIB_NAME "\n",Time());
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

//----------------------------------------------------------------------

int NetworkInitialize(void)   // perform platform specific socket init
{ return netInitAndDeinit( +1 ); }  
int NetworkDeinitialize(void) // perform platform specific socket deinit
{ return netInitAndDeinit( -1 ); }  
static int NetworkCheckIsOnline(void)
{ return netInitAndDeinit( 0 ); }  

//----------------------------------------------------------------------
