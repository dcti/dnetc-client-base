/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * El Cheapo network class
*/

#ifndef __NETWORK_H__
#define __NETWORK_H__ "@(#)$Id: network.h,v 1.68.2.15 2000/10/20 21:05:52 cyp Exp $"

class Network
{
protected:
  void *conn_handle;
  Network( void );
  ~Network( void );
public:
  friend Network *NetOpen( const char *servname, int servport, 
           int _nofallback = 1, int _iotimeout = -1, int _enctype = 0, 
           const char *_fwallhost = ((const char* )0), int _fwallport = 0, 
           const char *_fwalluid = ((const char* )0) );
  friend int NetClose( Network *net );
  int Get( char * data, int length );
  int Put( const char * data, int length );
  int GetHostName( char *buffer, unsigned int len );
  int SetPeerAddress( u32 addr );
  int Reset( u32 thataddress );
  u32 GetPeerAddress(void);
  void ShowConnection(void) { }; /* nothing */
};

#endif /* __NETWORK_H__ */

