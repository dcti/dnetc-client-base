/* Hey, Emacs, this a -*-C++-*- file !
 *
 * Copyright distributed.net 1997-2000 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * this module contains netconn_xxx() and support routines which are
 * high level connection open/close/read/write/reset with optional
 * on-the-fly http/uue tunneling and http/socks proxy support.
 *
*/

#ifndef __NETCONN_H__
#define __NETCONN_H__ "@(#)$Id: netconn.h,v 1.1.2.2 2000/10/24 21:36:35 cyp Exp $"

/* netconn_open(): create a new connection. Returns a 'handle' for
 * subsequent netconn_xxx() operations or NULL on error.
 * Blocks until connected/timeout/error. possible _enctype values are:
 * 0=no encoding/proxification, 1=uue, 2=http, 3=uue+http, 4=socks4, 5=socks5
*/
void *netconn_open( const char * _servname = 0, int _servport = 0,  
                    int _nofallback = 1, int _iotimeout = -1, int _enctype = 0, 
                    const char *_fwallhost = 0, int _fwallport = 0, 
                    const char *_fwalluid = 0 );

/* netconn_read(): receives data from the connection with any 
 * necessary decoding. Returns number of bytes copied to 'data'
 * Blocks until request has been fulfilled or timeout.
*/
int netconn_read( void *cookie, char * data, int numRequested );

/* netconn_write(): sends data over the connection with any 
 * necessary encoding. Returns 'length' on success, or -1 on error.
*/
int netconn_write( void *cookie, const char * data, int length );

/* netconn_getname(): name of host as determined at open time.
 * Returns zero on success or -1 on error.
*/
int netconn_getname(void *cookie, char *buffer, unsigned int len );

/* netconn_getpeer(): get address of host connected to, or zero
 * on error. Probably only useful for debugging.
*/
u32 netconn_getpeer(void *cookie);

/* netconn_getaddr(): get address connected from, or zero
 * on error (or not connected).
*/
u32 netconn_getaddr(void *cookie);

/* netconn_setpeer(): set address of host to connect to in the event
 * of a disconnect. (in the event of an HTTP/1.0 close)
*/
int netconn_setpeer(void *cookie, u32 address);

/* netconn_reset(): reset the connection. Fails (by design) if 
 * thataddress is zero. Blocks until reset is complete (or error)
*/
int netconn_reset(void *cookie, u32 thataddress);

/* netconn_close(): close the connection. Cookie is then no longer
 * usable.
*/
int netconn_close(void *cookie);

#endif /* __NETCONN_H__ */

