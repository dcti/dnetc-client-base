/*
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * El Cheapo Network class for compatibility.
*/
const char *network_cpp(void) {
return "@(#)$Id: network.cpp,v 1.97.2.42 2000/10/20 21:05:51 cyp Exp $"; }

#include "cputypes.h" /* u32 */
#include "netconn.h"  /* netconn_xxx() */
#include "network.h"  /* ourselves */

int Network::Get( char * data, int length )
{
  return netconn_read(conn_handle, data, length);
}

int Network::Put( const char * data, int length )
{
  return netconn_write(conn_handle, data, length);
}

int Network::GetHostName( char *buffer, unsigned int len )
{
  return netconn_getname(conn_handle, buffer, len );
}

int Network::SetPeerAddress( u32 addr ) 
{
  return netconn_setpeer(conn_handle, addr);
}
    
int Network::Reset( u32 thataddress )
{
  return netconn_reset(conn_handle, thataddress);
}

u32 Network::GetPeerAddress(void)
{ 
  return netconn_getpeer(conn_handle);
}

Network::~Network( void )
{
  netconn_close(conn_handle);
  conn_handle = (void *)0;
  return;
}

Network::Network(void) 
{
  conn_handle = (void *)0;
  return;
}

int NetClose( Network *net )
{
  if ( net )
    delete net;
  return 0;
}

Network *NetOpen( const char *servname, int servport,
           int _nofallback/*= 1*/, int _iotimeout/*= -1*/, int _enctype/*=0*/,
           const char *_fwallhost /*= NULL*/, int _fwallport /*= 0*/,
           const char *_fwalluid /*= NULL*/ )
{
  Network *net;

  net = new Network();
  if (net)
  {
    net->conn_handle = netconn_open( servname, servport, 
                                     _nofallback, _iotimeout, _enctype,
                                     _fwallhost, _fwallport, _fwalluid );
    if (!net->conn_handle)
    {
      delete net;
      net = (Network *)0;
    }
  }
  return net;
}


