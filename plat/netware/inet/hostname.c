/*
 * Classification: 4.2 BSD
 * Service: unistd.h
 * Author: Cyrus Patel <cyp@fb14.uni-mainz.de>
 * Copyright: none
 *
 * $Id: hostname.c,v 1.1.2.1 2001/01/21 15:10:30 cyp Exp $
 *
*/

#ifdef __cplusplus
extern "C" {
#endif
extern void *ImportPublicSymbol( int nlmHandle, const char *lname ); /* krnl/a3112 */
extern int UnImportPublicSymbol( int nlmHandle, const char *lname ); /* krnl/a3112 */
extern unsigned int GetNLMHandle(void);                              /* clib */
extern int GetFileServerName( int conn, char *buf /*>=49 chars*/);   /* clib */
extern unsigned int GetThreadGroupID(void);                          /* clib */
int gethostname( char *buffer, int bufflen );
int sethostname( char *hostname );
int getdomainname( char *buffer, int bufflen );
int setdomainname( char *domainname );
#ifdef __cplusplus
}
#endif

/* ===================================================================== */

int sethostname( char *hostname ) /* BSD compat */
{
  hostname = hostname;
  return -1;
}

int setdomainname( char *domainname ) /* BSD compat */
{
  domainname = domainname;
  return -1;
}

static char __domname[128+1]; /* technically... 1025 */

int getdomainname( char *buffer, int bufflen )
{
  int rc = -1;
  char hostname[8];
  if (buffer && bufflen)
  {
    if (bufflen < 1)
    {
      buffer[0] = '\0';
      rc = 0;
    }
    else if (gethostname( hostname, sizeof(hostname) ) != 0)
      ;
    else if (__domname[0])
    {
      rc = 0;
      while (__domname[rc] && rc < (bufflen-1))
      *buffer++ = __domname[rc++];
      *buffer = '\0';
      rc = 0;
    }
  }
  return rc;
}

int gethostname( char *buffer, int bufflen ) /* unlike TCPIP.NLM's function, */
{                                            /* this requires CLIB context */
  int rc = -1;
  if (buffer && bufflen)
  {
    if (bufflen < 1)
    {
      buffer[0] = '\0';
      rc = 0;
    }
    else
    {  
      static char servername[64+1]; /* actually, 48+1 should suffice */
      static int nlmHandle = -2;
      int len;      
      if (nlmHandle == -2)
      {
        servername[0] = '\0';
        __domname[0] = '\0';
        nlmHandle = -1;
      }  
      if ( nlmHandle == -1 )
      {
        /* we need to ensure that we have threadgroup (CLIB) context because */
        /* a) GetNLMHandle() requires context (the nlmHandle is in the */
        /*    NLMID struct, the pointer to which is in the thrgroup) */
        /* b) connection translation requires context. We _could_ */
        /*    use the kernel's serverName or ReturnServerName() but that */
        /*    requires ImportSymbol (and thus the nlmHandle) too. */
        len = GetThreadGroupID();
        if (len != -1 && len != 0)
        {
          nlmHandle = GetNLMHandle();
          if (GetFileServerName( 0 /*conn*/, servername ) == 0) 
          {        
            len = 0;
            servername[sizeof(servername)-1]='\0';
            while (servername[len])
            {
              if (servername[len]>='A' && servername[len]<='Z')
                servername[len] += ('a'-'A'); /* tolower(scratch[len]) */
              len++;
            }
          }
        }
      }

      if (nlmHandle != -1 && nlmHandle != 0) /* we had context at some time */
      {
        const char *fname = "\x0B""gethostname";
        int (*_gethostname)(char *, int ) = 
              (int (*)(char *, int ))ImportPublicSymbol( nlmHandle, fname );
        if (_gethostname)
        {
          rc = (*_gethostname)( __domname, sizeof(__domname) );
          UnImportPublicSymbol( nlmHandle, fname );
          if (rc != 0)
          {
            long ipaddr = -1L;
            long (*_gethostid)(void);
            fname = "\x09""gethostid";
            _gethostid = (long (*)())ImportPublicSymbol( nlmHandle, fname );
            if (_gethostid)
            {
              ipaddr = (*_gethostid)();
              UnImportPublicSymbol( nlmHandle, fname );
            }
            __domname[0] = '\0';
            if (ipaddr != 0 && ipaddr != -1L)
            {
              len = 0;
              for (rc=0;rc<4;rc++)
              {
                if (rc) __domname[len++]='.';
                if (( ((char *)&ipaddr)[rc] ) >= 100)
                  __domname[len++] = ((((char *)&ipaddr)[rc])/100)+'0';
                if (( ((char *)&ipaddr)[rc] ) >= 10) 
                  __domname[len++] = (((((char *)&ipaddr)[rc])%100)/10)+'0';
                __domname[len++] = ((((char *)&ipaddr)[rc])%10)+'0';
              }
              __domname[len] = '\0';
              rc = 0;
            }
          } /* if (rc != 0) */
        } /* if (_gethostname) */

        if (rc == 0)
        {
          if (__domname[0] == '\0')
            buffer[0] = '\0';
          else
          {
            __domname[sizeof(__domname)-1] = '\0';
            len = 0;
            while (__domname[len]=='.' || 
                   (__domname[len] >= '0' && __domname[len] <= '9'))
              len++;
            if (__domname[len] == '\0') /* IP address */
            {
              len = 0;
              while (__domname[len] && len<(bufflen-1))
                *buffer++ = __domname[len++];
              *buffer = __domname[0] = '\0';
            }
            else /* not IP address */
            {
              rc = -1; len = 0;
              while (__domname[len])
              {
                if (len < (bufflen-1))
                  *buffer++ = __domname[len];
                if (rc >= 0)
                  __domname[rc++] = __domname[len];
                else if (__domname[len] == '.')
                  rc = 0;
                len++;
              }
              if (rc < 0)
                rc = 0;
              *buffer = __domname[rc] = '\0';
              rc = 0;
            }  
          } /* domname[0] */
        } /* if rc == 0 */
      } /* we have an nlmhandle */

      if (rc != 0 && servername[0])
      {
        len = 0;
        while (servername[len] && len < (bufflen-1))
          *buffer++ = servername[len++];
        *buffer = '\0';
        rc = 0;
      }
    } /* if bufferlen > 1 */
  } /* if buffer */
  return rc;
}


