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
// Revision 1.1  1998/08/10 21:53:55  cyruspatel
// Two major changes to work around a lack of a method to detect if the network
// availability state had changed (or existed to begin with) and also protect
// against any re-definition of client.offlinemode. (a) The NO!NETWORK define is
// now obsolete. Whether a platform has networking capabilities or not is now
// a purely network.cpp thing. (b) NetworkInitialize()/NetworkDeinitialize()
// are no longer one-shot-and-be-done-with-it affairs. ** Documentation ** is
// in netinit.cpp.
//
//
//
#if (!defined(lint) && defined(__showids__))
const char *netinit_cpp(void) {
return "@(#)$Id: netinit.cpp,v 1.1 1998/08/10 21:53:55 cyruspatel Exp $"; }
#endif

//--------------------------------------------------------------------------

static int netInitAndDeinit( int doWhat )     //combines both init and deinit
{                                             //so statics can be localized
  static unsigned int initializationlevel=0;
  int success;

  //this nest check is presently cross-platform, but that may change...
  if ( doWhat == 0 )
    return (initializationlevel);
  else if ( doWhat < 0 )
    {
    if ( initializationlevel == 0 )
      {
      Log("**** SQUAWK!!! UNBALANCED NETWORK INIT/DEINIT!!! ***");
      abort();
      }
    if ((--initializationlevel)!=0)
      return 0;
    }
  else if ( doWhat > 0)
    {
    if ((++initializationlevel)!=1)
      return 0;
    }

  success = 1;

  if (doWhat > 0)
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
  
  //----------------------------

  #ifdef STUBIFY_ME
  if (success)
    {
    if (doWhat > 0) //init
      success = 0;
    else            //de-init
      success = 1;
    }
  #endif

  //----------------------------------------------

  #if (CLIENT_OS == OS_NETWARE)
  if (success)
    {
    if (doWhat > 0) //init
      success = nwCliIsNetworkAvailable(0);
    else            //de-init
      success = 1;
    }
  #endif

  //----------------------------------------------

  #if (CLIENT_OS == OS_WIN32) 
  if (success)
    {
    if (doWhat > 0) //init
      {
      WSADATA wsaData;
      WSAStartup(0x0101, &wsaData);
      success = 1;
      }
    else            //de-init
      {
      WSACleanup();
      success = 1;
      }
    }
  #endif

  //----------------------------------------------

  #if (CLIENT_OS == OS_OS2)
  if (success)
    {
    if (doWhat > 0) //init
      {
      sock_init();
      success = 1;
      }
    else            //de-init
      {
      success = 1;
      }
    }
  #endif

  //----------------------------------------------

  #if (CLIENT_OS == OS_AMIGAOS)
  if (success)
    {
    static struct Library *SocketBase;
    if (doWhat > 0) //init
      {
      #define SOCK_LIB_NAME "bsdsocket.library"
      SocketBase = OpenLibrary((unsigned char *)SOCK_LIB_NAME, 4UL);
      if (!SocketBase)
        {
        LogScreen("[%s] Network::Failed to open " SOCK_LIB_NAME "\n",Time());
        success = 0;
        }
      }
    else            //de-init
      {
      if (SocketBase) 
        {
        CloseLibrary(SocketBase);
        SocketBase = NULL;
        }
      success = 1;
      }
    }
  #endif

  //----------------------------------------------

  #ifdef SOCKS
  if (success) 
    {
    if (doWhat > 0) //init
      {
      LIBPREFIX(init)("rc5-client");
      }
    else            //de-init
      {
      }
    }
  #endif

  //----------------------------------------------

  if (!success)
    {
    if (doWhat > 0) //was it init that failed?
      --initializationlevel; //then decrement
    else            //otherwise it was deinit
      ++initializationlevel; //so increment
    return -1;
    }
  return 0;
}  

//----------------------------------------------------------------------

int NetworkInitialize(void)   // perform platform specific socket init
{ return netInitAndDeinit( +1 ); }  
int NetworkDeinitialize(void) // perform platform specific socket deinit
{ return netInitAndDeinit( -1 ); }  
static int QueryInitializationLevel(void) //staticified to prevent abuse 
{ return netInitAndDeinit( 0 ); }  

//----------------------------------------------------------------------
