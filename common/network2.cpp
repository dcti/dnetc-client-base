// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#include "cputypes.h"
#include "dcticonn.h"         // DCTIConnServer class
#include "network2.h"         // ourself
#include "logstuff.h"         // LogScreen()/Log()
#include "triggers.h"         // CheckExitRequestTriggerNoIO()



////////////////////////////////////////////////////////////////////////////

// Copies a hostname into a fixed-length buffer, taking care to
// ensure that the result is null terminated, and stripped of
// leading and trailing whitespace, and also forced to lowercase.

static void __hostnamecpy( char *dest,
    const char *source, unsigned int maxlen)
{
  unsigned int len = 0;
  while (*source && isspace(*source))
    source++;
  while (((++len) < maxlen) && *source && !isspace(*source))
    *dest++ = (char)tolower(*source++);
  *dest = 0;
}

////////////////////////////////////////////////////////////////////////////

Network::Network( const char *servname, int servport, 
         int _nofallback, int _iotimeout, int _enctype, 
         const char *_fwallhost, int _fwallport, 
         const char *_fwalluid)
{
  connserver = NULL;
  proxyver = 0;


  // server name, server port
  server_name[0] = 0;
  if (servname)
     __hostnamecpy( server_name, servname, sizeof(server_name));
  server_port = servport;
  autofindkeyserver = !server_name[0];


  // make sure server port matches hostname.
  const char *sname = strchr(server_name, 0);
  if (sname - 20 > server_name &&
      strcmpi(sname - 20, ".v27.distributed.net") == 0)
  {
    sname = server_name;
    server_port = 0;
    while (*sname != '.')
    {
      if (!isdigit(*sname)) sname++;
      else { server_port = atoi(sname); break; }
    }
  }
  if (!server_port) server_port = 2064;


  // fallback.
  nofallback = _nofallback;


  // connection mode.
  startmode = 0;
  if (_enctype == 1 /*uue*/ || _enctype == 3 /*http+uue*/)
  {
    startmode |= MODE_UUE;
  }
  if (_enctype == 2 /*http*/ || _enctype == 3 /*http+uue*/)
  {
    startmode |= MODE_HTTP;
    if (_fwallhost && _fwallhost[0])
    {
      fwall_hostport = _fwallport;
      if (_fwalluid)
        strncpy( fwall_userpass, _fwalluid, sizeof(fwall_userpass));
      __hostnamecpy( fwall_hostname, _fwallhost, sizeof(fwall_hostname));
    }
  }
  else if (_enctype == 4 /*socks4*/ || _enctype == 5 /*socks5*/)
  {
    if (_fwallhost && _fwallhost[0])
    {
      startmode |= ((_enctype == 4)?(MODE_SOCKS4):(MODE_SOCKS5));
      fwall_hostport = _fwallport;
      __hostnamecpy(fwall_hostname, _fwallhost, sizeof(fwall_hostname));
      if (_fwalluid)
        strncpy(fwall_userpass, _fwalluid, sizeof(fwall_userpass));
      if (fwall_hostport == 0)
        fwall_hostport = 1080;
    }
  }


  // i/o timeout value.
  iotimeout = _iotimeout; /* if iotimeout is <=0, use blocking calls */
  if (iotimeout <= 0)
    iotimeout = -1;
  else if (iotimeout < 5)
    iotimeout = 5;
  else if (iotimeout > 300)
    iotimeout = 300;
  
}

////////////////////////////////////////////////////////////////////////////

Network::~Network( void )
{
  Close();
}

////////////////////////////////////////////////////////////////////////////

int Network::Open(void)
{  
  unsigned int retries = 0;
  unsigned int maxtries = 5; /* 3 for preferred server, 2 for fallback */
  unsigned int preftries = 3;
  if (nofallback) maxtries = preftries;

  while (retries++ < maxtries) 
  {
    Close();

//    if (CheckExitRequestTriggerNoIO())
//      break; /* return -1; */


    // If we require a firewall hostname/port, check it now.
    if ((startmode & (MODE_SOCKS4 | MODE_SOCKS5 | MODE_HTTP)) != 0 &&
        (fwall_hostname[0] == 0 || fwall_hostport == 0))
    {
      Log("Network::Invalid %s proxy hostname or port.\n"
          "Connect cancelled.\n",
          ((startmode & (MODE_SOCKS4 | MODE_SOCKS5)) ?
          ("SOCKS") : ("HTTP")));
      break;
    }


    // Decide the servername we will actually connect to.
    char connect_name[64];
    if (autofindkeyserver)
      AutoFindServer( connect_name, sizeof(connect_name) );
    else if (!nofallback || retries < preftries)
      __hostnamecpy( connect_name, server_name, sizeof(connect_name) );
    else
      AutoFindServer( connect_name, sizeof(connect_name) );


    // Make sure we were able to find a servername.
    if (!connect_name[0])
    {
      Log("Network::Invalid keyserver hostname or port.\n"
          "Connect cancelled.\n");
      break;
    }


    // Create the new connection.
    connserver = new DCTIConnServer(connect_name, server_port);
    if (startmode & MODE_SOCKS5)
      connserver->SetSOCKS5Mode(fwall_hostname, fwall_hostport,
          NULL, fwall_userpass);
    else if (startmode & MODE_SOCKS4)
      connserver->SetSOCKS4Mode(fwall_hostname, fwall_hostport, fwall_userpass);
    else if (startmode & MODE_HTTP)
    {
      connserver->SetHTTPMode(fwall_hostname, fwall_hostport,
           fwall_userpass);
      if (startmode & MODE_UUE)
        connserver->SetUUEMode();
    }


    // Wait until the connection is established.
    if (EstablishConnection() == 0)
      return 0;

  }
  Close();
  return -1;
}

////////////////////////////////////////////////////////////////////////////

int Network::EstablishConnection(void)
{
  if (!connserver) return -1;
  if (iotimeout <= 0)
  {
    // Perform the establishment using blocking operations.
    while (connserver->IsConnected() &&
        !connserver->IsEstablished())
    {
      connserver->EstablishConnection(true, true);
    }
  }
  else
  {
    // Perform the establishment using non-blocking operations.
    while (connserver->IsConnected() &&
        !connserver->IsEstablished() &&
        (int) connserver->GetLastActivity() <= iotimeout)
    {
      SOCKET sock = connserver->GetSocket();
      
      // set up the file descriptor sets.
      fd_set readfds, writefds;
      FD_ZERO(&readfds);
      FD_ZERO(&writefds);
      if (sock)
      {
        // add the socket to our list.
        FD_SET(sock, &readfds);
        FD_SET(sock, &writefds);

        // perform the select.
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        if (select(sock + 1, &readfds, &writefds, NULL, &tv) < 0)
          return -1;
      }

      // attempt to establish to next state.
      bool readready = (FD_ISSET(sock, &readfds) != 0);
      bool writeready = (FD_ISSET(sock, &writefds) != 0);
      connserver->EstablishConnection(readready, writeready);
    }

  }
  return (connserver->IsEstablished() ? 0 : -1);
}

////////////////////////////////////////////////////////////////////////////

int Network::Close(void)
{
  if (connserver)
  {
    delete connserver;
    connserver = NULL;
  }
  return 0;
}

////////////////////////////////////////////////////////////////////////////

int Network::Reset( int fallbacknow )
{
  Close();
  return Open();
}

////////////////////////////////////////////////////////////////////////////

int Network::Get( net_packet_t *data )
{
  if (connserver)
  {
    if (EstablishConnection() < 0) return -1;
    if (iotimeout < 0)
    {
      // Perform the fetch using blocking operations.
      while (connserver->IsConnected() &&
          connserver->IsEstablished() )
      {
        if (connserver->PullPacket(data)) return 0;
        connserver->FlushOutgoing();
        connserver->FetchIncoming();
      }      
    }
    else
    {
      // Perform the fetch using non-blocking operations.
      connserver->RequestFlush();
      while (connserver->IsConnected() &&
          connserver->IsEstablished() &&
          (int) connserver->GetLastActivity() <= iotimeout)
      {
        if (connserver->PullPacket(data)) return 0;
        SOCKET sock = connserver->GetSocket();
      
        // set up the file descriptor sets.
        fd_set readfds, writefds;
        FD_ZERO(&readfds);
        FD_ZERO(&writefds);
        FD_SET(sock, &readfds);
        if (connserver->IsFlushNeeded())
          FD_SET(sock, &writefds);

        // perform the select.
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        if (select(sock + 1, &readfds, &writefds, NULL, &tv) < 0)
          return -1;

        // attempt to establish to next state.
        bool readready = (FD_ISSET(sock, &readfds) != 0);
        bool writeready = (FD_ISSET(sock, &writefds) != 0);
        if (writeready)
          connserver->FlushOutgoing();
        if (readready)
          connserver->FetchIncoming();
      }
    }    
  }
  return -1;
}

////////////////////////////////////////////////////////////////////////////

int Network::Put( net_packet_t *data )
{
  if (connserver)
  {
    if (connserver->QueuePacket(data))
      return 0;
  }
  return -1;
}

////////////////////////////////////////////////////////////////////////////

// show who we are connected to. (::Open() no longer does this)

void Network::ShowConnection(void)
{
}

////////////////////////////////////////////////////////////////////////////

void Network::SetScramble(u32 scram)
{
  if (connserver)
  {
    connserver->bScrambled = true;
    connserver->scramkey = scram;
  }
}

////////////////////////////////////////////////////////////////////////////
