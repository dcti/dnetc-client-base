/*
 * gethostbyname(), gethostbyname2(), gethostbyname_r(), gethostbyname2_r(),
 * ['2' versions not for AF_INET6] and gethostbyaddr(), gethostbyaddr_r(), 
 * functions for linux (broken resolver library/routines)
 *
 * (this should work on pretty much any unix - indeed, it was written on 
 * freebsd, where 'static' actually means 'static').
 * The only requirement is that the machine that this is to run on
 * has 'host', and that too can be bypassed (described below).
 *
 * Created Aug 2 2000, by Cyrus Patel <cyp@fb14.uni-mainz.de>
 * $Id: resolv.c,v 1.1.2.2 2001/02/18 23:58:40 cyp Exp $
 *
 * The functions exported from here will first check if a bypass 
 * (described below) has been provided. 
 * If a bypass has been defined:
 *    - only the bypass is called/processed. 
 * If a bypass has NOT been provided, the functions will attempt to 
 *    resolve name/address in the following order: (the order list is
 *    not loaded from an external file. See __gethostbyXXYY__() below)
 *    - process /etc/hosts, 
 *    - process 'builtins' ("localhost" etc), 
 *    - if _nss_dns_gethostbyX or _gethostbydnsX is available, call it,
 *    - process the output of the equiv of 'host -t <A|PTR> <name|addr>`
 * With the exception of not using /etc/hosts.conf or /etc/nsswitch.conf to
 * select the resolution order, great care has been taken to mimic 
 * gethostbyname() behaviour as closely as possible.
 *
 * The bypass: 
 *   - If a text file called './nslookup.txt' is found, it will be read.
 *     This is probably only useful for debugging.
 *   - If a script called './nslookup.sh' is found, it will be run with
 *     the same arguments as those passed to nslookup, that is, 
 *     '/bin/sh ./nslookup.sh -type=<A|PTR> <name|addr>'
 *   - This module exports a function 
 *     void set_lookup_processor_appname( const char *appname );
 *     that may be used to define a lookup plug-in.
 *     The application will be called with a hostname as an argument and 
 *     must print output similiar to that of 'host' or 'nslookup'.
 *     Both old and new output formats are supported.
 *     The application/script need not print anything if resolution fails.
 *     NOTE: Unlike nslookup, the lookup processor is expected to process
 *     /etc/hosts itself, ie it should use or emulate gethostbyname().
 *
 * nb: the functions only grok AF_INET addresses. I don't know anything 
 * about AF_INET6
 *
 * Why this stuff is needed:
 *   - A binary linked dynamically against glibc won't work against
 *     older libcs (obviously), BUT linking it statically won't either,
 *     which of course negates the advantage of static linking in the 
 *     first place.
 *   - The reason a staticly linked executable will fail to resolve (and
 *     indeed may even crash the process), is because glibc's resolver is 
 *     NOT static even when the app was linked with -static. Instead, it 
 *     uses a dynamic linker interface. (glibc can be compiled to make it 
 *     static, this may not be feasable, and is in fact not recommended 
 *     by the glibc maintainer <drepper@cygnus.com>).
 *     The dynamic resolver isn't compatible to anything other than the 
 *     'current' versions of the external dependancies, which naturally
 *     may or may not be same version as the dynamic resolver itself.
 *   - Due to a mutex bug in glibc, using glibc's dynamic linker 
 *     interface will eventually hang the process when running 
 *     multithreaded and nice on an smp machine (even if only one 
 *     thread is doing lookups). The dynamic linker will also
 *     segfault on a non-glibc machine.
 *   - If an older libc is statically linked, it may, or may not 
 *     work on a glibc/"GNU Name Service Switch" configured machine.
 * All these wierd things are completely bypassed when using this module:
 *   - nslookup/host only does dns, so the NSS issues are never dealt with,
 *   - nslookup/host doesn't depend on anything in libc beyond the trivial,
 *   - nslookup/host is compiled 'locally', non-static, and without 
 *     (the need for) reentrancy guards.
 *   - the gethostbyX functions here are suitable for both static and
 *     dynamic builds since it only does DNS (and /etc/hosts of course).
 * When is this module not good for you?:
 *   - you need YP or IPv6 support
 *   - it bothers you that /etc/hosts is always processed first, and
 *     you don't want to adjust __gethostbyXXYY__() below.
 *   - you have reason to believe 'host' isn't available AND you
 *     link dynamically AND the target machine won't have _nss_dns_gethostbyX 
 *     or _gethostbydnsX AND you don't want to adjust do_res_xxx() below.
 *   - you need the non-reentrant versions of gethostby[name|name2|addr]() to
 *     be thread-safe (ie, to use thread-local-storage) and you don't want 
 *     to adjust __get_local_hostent_data() below.
*/
#if defined(__linux__) && defined(__STRICT_ANSI__)
#undef __STRICT_ANSI__ /* headers won't prototype popen()/usleep() otherwise */
#endif
#define __NO_STRING_INLINES /* work around glibc bug in bits/strings2.h */

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>       /* ERANGE when *_r function buffer is too small */
#include <ctype.h>       /* isalpha, isdigit */
#include <netdb.h>       /* struct hostent, gethostbyname() */
#include <netinet/in.h>  /* struct in_addr */
#include <arpa/inet.h>   /* inet_ntoa, inet_addr */
#include <sys/socket.h>  /* AF_INET define */

#ifndef _REENTRANT /* consolidate various implied forms of _REENTRANT */
#  if defined(_THREAD_SAFE) || defined(THREAD_SAFE) || \
      defined(REENTRANT) || defined(_POSIX_THREAD_SAFE_FUNCTIONS) || \
      defined(_MIT_POSIX_THREADS)
#   define _REENTRANT
#  endif
#endif

/* ---------------------------------------------------------------------- */

#if defined(_REENTRANT)
/* #define SHOW_THREAD_UNSAFE_FIXMES */
/* #define NON_REENTRANT_FUNCTIONS_USE_TLS */
#endif

#if defined(DEBUG) 
#  define DEBUG_OUT(x) printf x
#else  
#  define DEBUG_OUT(x) /* nothing */
#endif   

#if defined(CMPTEST) || defined(TRACE)  
  /* gethostbyXYZ() is exported as test_gethostbyXYZ() if CMPTEST is defined */  
#  define TRACE_OUT(x) printf x
#else
#  define TRACE_OUT(x) /* nothing */
#endif

/* ---------------------------------------------------------------------- */

/* set the (path+)name of an application that will take a hostname
 * as an argument and print an imitation of what host would print
 * to stdout. The lookup processor is expected to process
 * /etc/hosts itself, ie it should use or emulate gethostbyname().
 * The application/script need not print anything if resolution fails.
 *
 * Supported output formats:
 * 1) [nslookup output]
 *    Name: hostname
 *    Address: 123.45.67.89 (or "Addresses: 123.45.67.89, 98.76.54.32,...")
 *    Aliases: othername    (the "Alias:"/"Aliases:" line is optional)
 * 2) [BIND 4 host -t A, with optional CNAME]
 *    "ftp.uni-mainz.de  \tCNAME\tftp1.uni-mainz.de"
 *    "fb14.uni-mainz.de  \tA\t134.93.246.119"
 * 3) [BIND 8+ host -t A, with optional CNAME]
 *    "ftp.uni-mainz.de{.} is a nickname for ftp1.uni-mainz.de{.}"
 *    "ftp1.uni-mainz.de{.} has address 134.93.8.108"
 *    (hostnames have a trailing dot {.} only for BIND 9+)
 * 4) [BIND 4 host -t PTR 134.93.246.119]
 *    "119.246.93.134.in-addr.arpa\tname = fb14.uni-mainz.de"
 *    (without an explicit '-t PTR' it returns the same format as 1)
 * 5) [BIND 8+ host [-t PTR]]
 *   "10.39.161.130.in-addr.arpa{.} domain name pointer tiamat.et.tudelft.nl{.}"
 *    (hostnames have a trailing dot {.} only for BIND 9+)
*/       
#ifdef __cplusplus
extern "C" { 
#endif
extern void set_lookup_processor_appname( const char *appname );
#ifdef __cplusplus
} 
#endif
static char __lookup_processor_appname[256+1] = {0};
void set_lookup_processor_appname( const char *appname )
{
  if (!appname)
    __lookup_processor_appname[0] = '\0';
  else if (strlen(appname) < sizeof(__lookup_processor_appname))
    strcpy( __lookup_processor_appname, appname );
  TRACE_OUT(("set_proc: \"%s\"\n", __lookup_processor_appname));
  return;
}

/* ---------------------------------------------------------------------- */

struct gen_hostent_data
{                         /* when a *_r function was called, the members */
  struct hostent *result; /* are the same the arguments of that _r function, */
  char *buffer;           /* otherwise, for non-reentrant functions, they */
  int buflen;             /* point to static storage (or TLS space) */
  int *h_errnop;
};

/* apply h_errno and (optionally) an errno */
static void __apply_h_errno_to_hostent( struct gen_hostent_data *_r_buffer,
                                        int my_h_errno, int my_errno)
{
  if (_r_buffer)
  {
    if (_r_buffer->h_errnop)
      *(_r_buffer->h_errnop) = my_h_errno;
  }      
  if (my_h_errno == -1) /* NETDB_INTERNAL */
  {
    if (my_errno > 0) /* otherwise its already set */
      errno = my_errno;
  }    
  return;
}

static struct gen_hostent_data *__apply_r_func_params_to_hostent( 
        struct gen_hostent_data *_r_buffer, struct hostent *result, 
        char *buffer, int buflen, int *h_errnop )
{
  if (!buffer || !result)
    buflen = 0;
  else if (buflen >= ((int)sizeof(char *))) 
  {
    /* make sure the buffer is aligned */
    unsigned long aoff = ((unsigned long)buffer);
    if ( (aoff & (sizeof(char *)-1)) != 0)
    {
      aoff = (sizeof(char *)-((size_t)(aoff & (sizeof(char *)-1))));
      buffer += aoff;
      buflen -= aoff;
    }
  }
  if (!buffer || buflen < ((int)sizeof(char *)) || !result /*|| !h_errnop */) 
  {
    errno = EINVAL;
    if (h_errnop)
      *h_errnop = -1;
    return (struct gen_hostent_data *)0;
  }    
  _r_buffer->result = result;    
  _r_buffer->buffer = buffer;
  _r_buffer->buflen = buflen;
  _r_buffer->h_errnop = h_errnop;
  return _r_buffer;
}

/* generate a 'struct gen_hostent_data' from static data or thread local
 * storage for use with a non-reentrant gethostbyX() function.
*/
static struct gen_hostent_data *__get_local_hostent_data( 
                                struct gen_hostent_data *non_r_buffer )
{
  struct __tls_hostent_data {
    struct hostent result;
#define __MAX_ADDR_CACHE 32 /* max number of addresses to store */
    char buffer[ 256+1+ /* hostname */
               ((__MAX_ADDR_CACHE+1)*sizeof(char *))+ /* addrlist */
              ((__MAX_ADDR_CACHE)*sizeof(struct in_addr))+ /* addr */
              ((7)*sizeof(char *))+ /* alias list */
              256+1+ /* alias buffer */
              sizeof(char *) /* padding */];
#undef __MAX_ADDR_CACHE
  };  
#define NEED_STATIC_LOCAL_HOSTENT_DATA
#if defined(SHOW_THREAD_UNSAFE_FIXMES) && defined(_REENTRANT) && \
      defined(NON_REENTRANT_FUNCTIONS_USE_TLS)
#   undef NEED_STATIC_LOCAL_HOSTENT_DATA
#   error Code needs to be adjusted for use with threads such that
#   error calls to non-reentrant versions of gethostbyX use
#   error thread local storage as opposed to static storage. 
    /* On failure to allocate/obtain a tls pointer, return NULL */
    errno = ENOMEM;
    return (struct gen_hostent_data *)0;
#endif
#ifdef NEED_STATIC_LOCAL_HOSTENT_DATA
#   undef NEED_STATIC_LOCAL_HOSTENT_DATA
    static struct __tls_hostent_data static_storage;
    return __apply_r_func_params_to_hostent( non_r_buffer, 
           &(static_storage.result), &(static_storage.buffer[0]), 
           sizeof(static_storage.buffer), &(h_errno) );
#endif                          
}

/* ---------------------------------------------------------------------- */

/* split/parse/stuff the results from do_etc_hosts/do_nslookup etc
 * into the gen_hostent_data space
*/    
static struct hostent *do_gen_hostent( struct gen_hostent_data * _r_buffer,
                                       const char *hostname,
                                       const struct in_addr *addr_list, 
                                       unsigned int addr_count,
                                       const char **alias_list,
                                       unsigned int alias_count,
                                       int gen_errno )
{
  struct hostent *hp = (struct hostent *)0;
  if (!addr_count)
  {
    if (_r_buffer) /* should always be so (done in __gethostbyXXYY__) */
    {
      if (_r_buffer->h_errnop)
        *(_r_buffer->h_errnop) = gen_errno;
    }  
  }
  else if (_r_buffer) /* should always be so (done in __gethostbyXXYY__) */
  {
    static char *static_null_pointer = (char *)0; /* dummy */
    int min_needed = strlen(hostname)+1+sizeof(char *)+  /* h_name */
                     sizeof(char **)+(sizeof(char *)*2)+ /* h_addr_list */
                     sizeof(addr_list[0]); /* h_addr_list[0] */

    if (!_r_buffer->buffer || !_r_buffer->result)
      _r_buffer->buflen = 0;
    else if (_r_buffer->buflen > (int)(sizeof(char *)))
    {  
      /* make sure the buffer is aligned */
      unsigned long aoff = ((unsigned long)_r_buffer->buffer);
      if ((aoff & (sizeof(char *)-1)) != 0)
      {
        aoff = (sizeof(char *)-((size_t)(aoff & (sizeof(char *)-1))));
        _r_buffer->buffer += aoff;
        _r_buffer->buflen -= aoff;
      }        
    }
/* printf("buflen=%d, min_needed=%d\n", _r_buffer->buflen, min_needed ); */
    
    if ((min_needed >= _r_buffer->buflen) || !_r_buffer->result)
    {
#if defined(ERANGE)
      errno = ERANGE;
#endif
      if (_r_buffer->h_errnop)
        *(_r_buffer->h_errnop) = -1; /* NETDB_INTERNAL */
    }
    else
    {
      char *datap;
      unsigned int max_addrs = 0;
      unsigned int max_aliases = 0;
      unsigned int i; int addit_needed;
      int namelen, buflen = _r_buffer->buflen;

      for (i = 0; min_needed < buflen && i < addr_count; i++)
      {
        addit_needed = sizeof(char *)+sizeof(addr_list[0]);
        if (i == 0)
          addit_needed += sizeof(char **)+sizeof(char *);
        min_needed += addit_needed;
        if (min_needed < buflen)
          max_addrs++;
      }        
/* printf("max_addrs=%u\n", max_addrs); */
      for (i = 0; min_needed < buflen && i < alias_count; i++)
      {
        addit_needed = sizeof(char *)+strlen(alias_list[i])+1;
        if (i == 0)
          addit_needed += sizeof(char **)+sizeof(char *);
        min_needed += addit_needed;
        if (min_needed < buflen)
          max_aliases++;          
      }         
/* printf("max_aliases=%u\n", max_aliases); */

      hp = _r_buffer->result;
      datap = _r_buffer->buffer; /* already aligned */
      
      static_null_pointer = (char *)0;
      memset( hp, 0, sizeof(struct hostent));
      hp->h_addrtype = AF_INET;
      hp->h_length = sizeof(struct in_addr);
      hp->h_name = "";
      hp->h_aliases = &static_null_pointer;
      hp->h_addr_list = &static_null_pointer;
      
      if (hostname[0])
      {
        hp->h_name = datap;
        namelen = strlen( strcpy( datap, hostname ) );
        if (datap[namelen-1] == '.') /* canonical */
          datap[--namelen] = '\0';
        datap += 1 + namelen;
      }
      if (max_addrs)
      {
        datap+=(sizeof(char *)-((datap-(_r_buffer->buffer))&(sizeof(char *)-1)));
        hp->h_addr_list = (char **)datap;
        datap += (sizeof(char *) * (max_addrs+1));
        for (i = 0; i < max_addrs; i++)
        {
          hp->h_addr_list[i] = datap;
          memcpy( datap, &addr_list[i], sizeof(addr_list[i]) );
          datap += sizeof(addr_list[i]);
        }
        hp->h_addr_list[max_addrs] = (char *)0;
        hp->h_addr_list = &(hp->h_addr_list[0]);
      }
      if (max_aliases)
      {
        datap+=(sizeof(char *)-((datap-(_r_buffer->buffer))&(sizeof(char *)-1)));
        hp->h_aliases = (char **)datap;
        datap += ((max_aliases+1)*sizeof(char *));
        for (i = 0; i < max_aliases; i++)
        {
          hp->h_aliases[i] = datap;
          strcpy( datap, alias_list[i] );
          namelen = strlen( datap );
          if (namelen && datap[namelen-1] == '.') /* canonical */
            datap[--namelen] = '\0';
          datap += 1 + namelen;
        }
        hp->h_aliases[max_aliases] = (char *)0;
        hp->h_aliases = &(hp->h_aliases[0]);
      }
      if (_r_buffer->h_errnop)
        *(_r_buffer->h_errnop) = 0; /* NETDB_SUCCESS */
    }
  }
/* printf("return hp=%p\n", hp ); */
  return hp;
}

/* ---------------------------------------------------------------------- */

#if 0               /* resolv.conf parse works fine, but is no longer needed */
                    /* see match_hostnames() for more info */
struct resolv_conf
{
  int flags; /* currently unused, but needed to supress compiler warning */
#ifdef MAXNS
#  define __MAXNS MAXNS
#else
#  define __MAXNS 3
#endif
  struct in_addr ns_list[__MAXNS];
#undef __MAXNS
  char domain_name[256+1];
  const char *search_list[6+1];
  char search_buf[256+1];
  char resolve_order[8+1];
};

static struct resolv_conf *get_resolv_conf_data(void)
{
  static struct resolv_conf resconf = {0};
  static int need_init = 1;

#if defined(SHOW_THREAD_UNSAFE_FIXMES) && defined(_REENTRANT)
#  error FIXME: 'need_init' needs spinlock/mutex protection
#endif
  if (need_init)
  {
    FILE *file = fopen("/etc/resolv.conf", "r");
    int search_count = 0, ns_count = 0, have_domname = 0;
    
    if (file)
    {
      char tokbuf[256+1]; int kw = 0;
      size_t toklen = 0, search_used = 0;

      for (;;)
      {
        int i, c = fgetc(file);
        if (c == '#' || c == ' ' || c == '\t' || c == '\n' || c == EOF)
        {
          if (toklen > 0 && toklen < sizeof(tokbuf))
            tokbuf[toklen++] = '\0';
          if (toklen < 1)
            ; /* nothing */
          else if (toklen >= sizeof(tokbuf)) /* token too long */
          {
            if (c == ' ' || c == '\t')
              c = '#'; /* ignore to eol */
          }      
          else if (kw == 'd')                                 /* domain */
          {
            if (!have_domname && toklen < sizeof(resconf.domain_name))
            {
              strcpy(resconf.domain_name, tokbuf);
              have_domname = 1;
            }  
            if (c == ' ' || c == '\t')
              c = '#'; /* ignore to eol */
          }
          else if (kw == 'n')                                /* nameserver */
          {
            if (ns_count < 
               ((sizeof(resconf.ns_list)/sizeof(resconf.ns_list[0]))-1) )
            {
              resconf.ns_list[ns_count].s_addr = inet_addr(tokbuf);
              if (resconf.ns_list[ns_count].s_addr != 0xffffffff && 
                  resconf.ns_list[ns_count].s_addr != 0)
              {
                for (i = 0; i < ns_count; i++)
                {
                  if (resconf.ns_list[i].s_addr == 
                      resconf.ns_list[ns_count].s_addr)
                  {
                    toklen = 0;
                    break;
                  }
                }
                if (toklen != 0)      
                  ns_count++;
              }          
            } 
            if (c == ' ' || c == '\t')
              c = '#'; /* ignore to eol */
          }
          else if (kw == 's' || kw == 'S')                   /* search */
          {
            if ((kw == 'S' || search_count == 0) &&
              (toklen+search_used) < sizeof(resconf.search_buf) &&
              search_count < ((sizeof(resconf.search_list)/sizeof(resconf.search_list[0]))-1)) 
            {
              for (i = 0; i < search_count; i++)
              {
                if (strcasecmp(resconf.search_list[i], tokbuf) == 0)
                {
                  toklen = 0;
                  break;
                }
              }
              if (toklen != 0)
              {
                resconf.search_list[search_count++] = 
                    strcpy(&resconf.search_buf[search_used], tokbuf);
                search_used += toklen+1;
                kw = 'S'; 
              }
            }      
          }    
          else if (c == ' ' || c == '\t') /* not end of line/file */
          { 
            if (strcmp(tokbuf,"domain")==0)
              kw = 'd';
            else if (strcmp(tokbuf,"nameserver")==0)
              kw = 'n';
            else if (strcmp(tokbuf,"search")==0)
              kw = 's';
            else  
              c = '#'; /* ignore the rest of the line */        
          }        
          if (c == '#')
          {
            while (c != EOF && c != '\n')
              c = fgetc(file);
          }
          if (c == EOF)
            break;
          if (c == '\n')
            kw = 0;
          toklen = 0;
        }  
        else if (toklen < sizeof(tokbuf))
          tokbuf[toklen++] = (char)c;
        else 
          toklen = 0;    
      } /* for (;;) */
      fclose(file);
    } /* if file */

    if (!have_domname) /* no domain name determined */
    {
      if (gethostname(resconf.domain_name, sizeof(resconf.domain_name)) == 0)
      {
        resconf.domain_name[sizeof(resconf.domain_name)-1] = '\0';
        if (resconf.domain_name[0] && strlen(resconf.domain_name) < (sizeof(resconf.domain_name)-1))
        {
          char *p = strchr(resconf.domain_name, '.' );
          if (p)   
          {
            unsigned int i = strlen(p+1);
            if (i > 0 && i < (sizeof(resconf.domain_name)-1))
            {
              strcpy( resconf.domain_name, p+1 );
              have_domname = 1;
            }  
          }      
        }  
      }        
    }
    if (!have_domname)
      resconf.domain_name[0] = '\0';
    
    if (search_count == 0 && have_domname) /* no search list determined */
    {
      resconf.search_list[search_count++] = resconf.domain_name;
    }  
    resconf.search_list[search_count] = (const char *)0;
    
    if (ns_count == 0)  /* no namserver determined */
    {
      resconf.ns_list[ns_count++].s_addr = 0x0100007f; /* localhost */
    }
    resconf.ns_list[ns_count].s_addr = 0;

    if (file)    
      need_init = 0;

#if defined(DEBUG) || defined(TRACE1) || defined(TRACE2)
    {
      int q;
      printf( "resolv_conf: domainname=\"%s\"\n", resconf.domain_name );
      printf( "resolv_conf: %d search elements\n", search_count );
      for (q = 0; q <search_count; q++)
      printf( "             %d) \"%s\"\n", q+1, resconf.search_list[q] );
      printf( "resolv_conf: %d name servers\n", ns_count );
      for (q = 0; q <ns_count; q++)
      printf( "             %d) \"%s\"\n",q+1,inet_ntoa(resconf.ns_list[q]));
    }  
#endif
      
  } /* if need_init */
  
  return &resconf;    
}
#endif

/* ---------------------------------------------------------------------- */

static int do_nslookup( struct gen_hostent_data *_r_buffer, 
                        const char *hostname, 
                        const char *lookup_appname, int reverse, 
                        struct hostent **hpp )
{
  int proceed_to_next = -1;
#define MALLOC_STEP_SIZE 8192
  char *lubuf; size_t lubuflen = strlen(hostname)+(MALLOC_STEP_SIZE*2);
  lubuf = (char *)malloc( lubuflen );

  if (lubuf)
  {
    size_t readlen, toread, spoollen; 
    int do_debug = 0;
    FILE *file;

    if (access("./nslookup.out",0)==0)
    {
      strcpy(lubuf, "cat ./nslookup.out");
      do_debug = 1;
    }  
    else
    {
      const char *app = lookup_appname;
      const char *opt = " -type=";
      int need_sh = 1;
      if (access("./nslookup.sh",0)==0)
      {
        app = "./nslookup.sh";
        do_debug = 1;
      }  
      else if (!app)
      {
        need_sh = 0;
        app = "host";
        opt = " -t ";
        do_debug = 0;
      }  
      lubuf[0] = '\0';
      if (need_sh) /* double /bin/sh is to suppress stderr messages completely */
        strcpy(lubuf, "/bin/sh ");
      strcat(strcat(lubuf,app),opt);
      if (reverse)
        sprintf(&lubuf[strlen(lubuf)],"PTR %d.%d.%d.%d",
           ((int)(hostname[0]))&0xff, ((int)(hostname[1]))&0xff, 
           ((int)(hostname[2]))&0xff, ((int)(hostname[3]))&0xff);
      else
        strcat(strcat(lubuf, "A "),hostname);
      strcat( lubuf, " 2>/dev/null" );
    }        

    TRACE_OUT(("doing do_nslookup(\"%s\")\n",
                ((lookup_appname)?(lookup_appname):("nslookup"))));
    file = popen( lubuf, "r");
    if (do_debug)                
      printf("popen(\"%s\")=>%p %s\n", lubuf, file, (file)?(""):(strerror(errno)) );
    spoollen = 0;
        
    if (file)
    {
      int tries = 0, err = 0;
      for (;;)
      {
        if (!err && ((lubuflen-spoollen) < (MALLOC_STEP_SIZE/4)))
        {
          char *p = (char *)realloc( lubuf, lubuflen+MALLOC_STEP_SIZE );
          if (!p)
            err = 1;
          else  
          {
            lubuf = p;
            lubuflen += MALLOC_STEP_SIZE;
          }  
        }
        if (err)
          spoollen = 0;

        toread = (lubuflen-spoollen);
        readlen = fread( lubuf, 1, toread, file );
        spoollen += readlen;
        if (readlen < toread)
        {
          if (feof(file))
            break;
          if ((++tries) > 10)
          {
            err = 1;
            break;
          }  
          usleep(200000);
        }    
      } /* for (;;) */
        
      if (err)
        spoollen = 0;
      pclose(file);
    } /* if (popen()) */

    if (spoollen)
    {
      struct in_addr addr_list[32];
      unsigned int addr_count = 0;
      const char *aliaslist[8];
      unsigned int aliascount = 0;
      unsigned int readpos = 0;
      int innamesect = 0, oldstyleA = 0, newstyleA = 0;
      unsigned int linenum = 0, nblinenum = 0;

      lubuf[spoollen] = '\0';
      
      while (lubuf[readpos])
      {
        char *p, *q, *bufp = &lubuf[readpos];
        unsigned int linelen = 0;

        linenum++;
        while (bufp[linelen] && bufp[linelen]!='\n')
          linelen++;
        readpos += linelen;
        if (bufp[linelen] == '\n')
        {
          bufp[linelen] = '\0';  
          readpos++;
        }  
        while (linelen && (*bufp == ' ' || *bufp == '\t' || *bufp == '>'))
        { bufp++; linelen--; }
        if (linelen == 0)
          continue;
        nblinenum++;

        if (do_debug)                
          printf("line %u: (len=%u): \"%s\"\n", linenum, linelen, bufp );

        if (!innamesect && reverse)
        {
          /* newer `nslookup -type=PTR` does
          **   "119.246.93.134.in-addr.arpa\tname = fb14.uni-mainz.de"
          **   (different output than when not specifying -type)
          */
          char in_addr_arpa_sig[32+sizeof(".in-addr.arpa\tname = ")];
          sprintf( in_addr_arpa_sig,
                   "%d.%d.%d.%d.in-addr.arpa\tname = ",
              (((int)hostname[3])&0xff), (((int)hostname[2])&0xff),
              (((int)hostname[1])&0xff), (((int)hostname[0])&0xff) );
          q = &in_addr_arpa_sig[0];
          p = strstr( bufp, q );
          if (p <= bufp)
          {  
            /* newer `host [-t PTR] 130.181.38.10`
            ** "10.38.181.130.in-addr.arpa. domain name pointer tiamat.et.tudelft.nl"
            ** note that the .in-addr.arpa may not be lower case since 'host' 
            ** prints whatever the nameserver sent it.
            ** Also note that BIND 8+ 'host' puts a dot at the end of 'arpa'
            ** which older 'host' does not.
            */
            q = " domain name pointer ";
            p = strstr(bufp, q);
          }
          if (p > bufp)
          {     
            addr_list[addr_count++].s_addr = 
                              ((const struct in_addr *)hostname)->s_addr;
            p += (strlen(q)-1);
            while (*p == ' ' || *p == '\t')
              p++;
            q = (p + strlen(p)) - 1;
            while (q >= p && (*q == ' ' || *q == '\t'))
              q--;
            if (q > p)
            {
              *++q = '\0';  
              hostname = (const char *)p;
            }        
            break; /* while (lubuf[readpos]) */
          }  
        } /* if (!innamesect && reverse) */
        
        if (!innamesect && !reverse && nblinenum == 1)
        {
          int ostyle = 0;
          /* 
          ** "ftp.uni-mainz.de is a nickname for ftp1.uni-mainz.de"
          */
          q = " is a nickname for ";
          p = strstr(bufp, q);
          if (p <= bufp)
          {
            /*
            ** "ftp1.uni-mainz.de has address 134.93.8.108" 
            */
            q = " is an alias for ";
            p = strstr(bufp, q );
          }  
          if (p <= bufp)
          {
            /*
            **  "ftp.uni-mainz.de  \tCNAME\tftp1.uni-mainz.de"
            */
            q = "\tCNAME\t";
            p = strstr(bufp, q );
            ostyle = 1;
          }  
          if (p > bufp)
          {
            p += strlen(q);
            while (*p == ' ' || *p == '\t')
              p++;
            q = p;
            while (*p && *p != ' ' && *p != '\t')  
              p++;
            if (p == q)
              p = (char *)0;  
            else
            {
              *p = '\0';
              if (aliascount < (sizeof(aliaslist)/sizeof(aliaslist[0])-1))
                aliaslist[aliascount++] = hostname;
              hostname = q;
              if (do_debug)                
                printf("CNAME='%s'\n",q);
            }
            if (ostyle)
              oldstyleA = 1;
            else
              newstyleA = 1;
            continue;    
          } /* if (q > p) */
        } /* if (!innamesect && !reverse && nblinenum == 1) */

        if (!innamesect && !reverse && 
           (nblinenum == 1 || oldstyleA || newstyleA))
        {
          int ostyle = 0;
          /* "fb14.uni-mainz.de has address 134.93.246.119" */
          q = " has address ";
          p = strstr(bufp, q);
          if (!newstyleA && !p)
          {
            /* "fb14.uni-mainz.de   \tA\t134.93.246.119" */
            q = "\tA\t";
            p = strstr( bufp, q );
            ostyle = 1;
          }
          if (p <= bufp)
          {
            if (oldstyleA || newstyleA)
              break;  /* while (lubuf[readpos]) */
            /* else fallthrough and never come back */  
          }
          else
          {
            if ((ostyle && newstyleA) || (!ostyle && oldstyleA))
              break; /* while (lubuf[readpos]) */
            if (ostyle)
              oldstyleA = 1;
            else
              newstyleA = 1;
            p += strlen(q);
            while (*p == ' ' || *p == '\t')
              p++;
            bufp = p;
            while ( *p == '.' || isdigit(*p))
              p++;
            if (p == bufp || (*p!='\0' && *p!=' ' && *p!='\t'))
              break; /* zero length or bad termination */
            *p = '\0';  
            addr_list[addr_count].s_addr = inet_addr(bufp);
            if (addr_list[addr_count].s_addr != 0xffffffffu)
            {
              if (do_debug)
                printf("A: addr_list[%u]='%s'\n",addr_count,bufp);
              addr_count++;
              if (addr_count >= (sizeof(addr_list)/sizeof(addr_list[0])-1))
                break; /* while (lubuf[readpos]) */
            } 
          }
        }  
        
        if (!newstyleA && !oldstyleA) /* not handled just above this */
        {
          /* new 'A', old 'PTR' */
          /* 
          *  Example lines: [multiple lines in one 'section']
          *  [this format is used for both 'A' and 'PTR' records]
          *  "Name: ftp1.uni-mainz.de"
          *  "Address: 134.93.246.119" (or "Addresses: x.x.x.x, y.y.y.y")
          *  "Aliases: ftp.uni-mainz.de"
          */
          /* empty, null, blank lines have already been removed, old 'A' and 
             new 'PTR' lines have already been handled, so whats left should
             only be lines in the form "label: xxxxx"
          */     
          p = strchr( bufp, ':');
          if (!p)
            break; /* while (lubuf[readpos]) */

          *p++ = '\0';
          if (strcmp( bufp, "Name" )==0)
          {
            if (innamesect) /* can't have two name sections */
              break; /* while (lubuf[readpos]) */  
            innamesect = 1;
            while (*p == ' ' || *p == '\t')
              p++;
            q = p;
            while (*p && *p != ' ' && *p != '\t')
              p++;
            if (p != q)
            {
              *p = '\0';
              hostname = (const char *)q;
              if (do_debug)
                printf("innamesect-A/PTR: name='%s'\n",q);
            }  
          }
          else if (innamesect)
          {
            int isalias = (bufp[1] == 'l');
            if ( strcmp(bufp, "Address" ) && strcmp(bufp, "Addresses" ) &&
                 strcmp(bufp, "Aliases" ) && strcmp(bufp, "Alias" ) )
            {                                    
              break; /* while (lubuf[readpos]) - name section ended */
            }
            bufp = p;
            while (*bufp)
            {  
              DEBUG_OUT(("bufp1=%s\n",bufp));              
              while (*bufp == ' ' || *bufp == '\t' || *bufp == ',')
                bufp++;
              p = bufp;
              if (isalias)
              {
                if (aliascount >= (sizeof(aliaslist)/sizeof(aliaslist[0])-1))
                  break; /* while (*bufp) */
                while (*p && *p != ' ' && *p !='\t' && *p != ',')
                  p++;                                 
              }
              else
              {
                if (addr_count >= (sizeof(addr_list)/sizeof(addr_list[0])-1))
                  break; /* while (*bufp) */
                while ( *p == '.' || isdigit(*p))
                  p++;
              }          
              if (p == bufp || (*p!='\0' && *p!=' ' && *p!='\t' && *p!=','))
                break; /* zero length or bad termination */
              if (*p != '\0')
                *p++ = '\0';  
              DEBUG_OUT(("bufp2=%s\n",bufp));               
              if (isalias)
              {
                if (do_debug)
                  printf("innamesect-A/PTR: alias[%u]='%s'\n",aliascount,bufp);
                aliaslist[aliascount++] = bufp;
              }  
              else
              {
                addr_list[addr_count].s_addr = inet_addr(bufp);
                if (addr_list[addr_count].s_addr != 0xffffffffu)
                {
                  if (do_debug)
                    printf("innamesect-A/PTR: addr_list[%u]='%s'\n",addr_count,bufp);
                  addr_count++;
                }  
              }    
              bufp = p;        
            } /* while (*bufp) */        
          }  /* if (innamesect) */
        } /* new 'A', old 'PTR' */

      } /* while (lubuf[readpos]) */

      proceed_to_next = +1;
      if (addr_count)
      {      
        TRACE_OUT(("addr_count from do_nslookup() = %u\n", addr_count));
        *hpp = do_gen_hostent( _r_buffer, hostname, addr_list, addr_count, 
                              aliaslist, aliascount, 0 );
        proceed_to_next = 0;                              
      }
    } /* if spoollen */
    free( (void *)lubuf );
  } 
  if (proceed_to_next) /* -1=internal error, +1 = none found */
  {
    TRACE_OUT(("addr_count from do_nslookup() = 0, h_errno=%d\n",proceed_to_next));
    do_gen_hostent( _r_buffer, hostname, 0, 0, 0, 0, proceed_to_next );
    *hpp = (struct hostent *)0;
#if 0
    if (proceed_to_next < 0)
      proceed_to_next = 0;
    else 
#endif 
      proceed_to_next = 1;
  }
  return proceed_to_next;
}

/* ---------------------------------------------------------------------- */

static const char * do_validate_extern_procname(void)
{
  if (__lookup_processor_appname[0])
  {
    /* embedded white space handling is for macosx */
    int validated = 0, gotpath = 0;
    unsigned int maxlen = 0, cmdlen = 0; char quoted;
    char procnamebuf[sizeof(__lookup_processor_appname)+1];
    const char *appname = __lookup_processor_appname;    
    while (*appname == ' ' || *appname == '\t')
      appname++;
    quoted = 0;  
    if (*appname == '\"' || *appname == '\'')
    {
      unsigned int spcpos;
      quoted = (char)*appname++;  
      while (*appname == ' ' || *appname == '\t')
        appname++;
      while (appname[maxlen] && appname[maxlen]!=quoted)
        maxlen++;
      while (maxlen > 0 && (appname[maxlen-1]==' ' || appname[maxlen-1]=='\t'))
        maxlen--;
      spcpos = 0;
      while (spcpos<maxlen && appname[spcpos]!=' ' && appname[spcpos]!='\t')
        spcpos++;
      if (spcpos == maxlen)
      {
        quoted = 0;
        cmdlen = 0;
      }        
    }  
    while (appname[cmdlen])
    {
      if (quoted && cmdlen == maxlen)
        break;
      if (appname[cmdlen] == '/')
        gotpath = 1;
      else if (appname[cmdlen]== ' ' || appname[cmdlen]== '\t')
      {
        if (!quoted)
                break;  
      }          
      if (cmdlen >= (sizeof(procnamebuf)-2))
      {
        cmdlen = 0;
        break;
      }        
      procnamebuf[cmdlen] = (char)appname[cmdlen];
      cmdlen++;
    }  
    procnamebuf[cmdlen] = '\0';  
    if (cmdlen && gotpath)
    {
      /* DEBUG_OUT(("validate_app_name(simple): \"%s\"\n", procnamebuf )); */
      if (access(procnamebuf, 0 /*X_OK*/) == 0)
        validated = 1;
    }        
    else if (cmdlen)
    {
      char *dirlist = getenv("PATH");
      if (dirlist)
      {
        char fullpathbuf[sizeof(__lookup_processor_appname)+32];
        unsigned int dirlen = 0; char dirquoted = 0;
        int donecurrdir = 0, baddir = 0;
        TRACE_OUT(("search_path: \"%s\"\n", dirlist ));
        for (;;)
        {
          char c = *dirlist;
          if (c != '\0')
            dirlist++;
          if (c == ':' || c == '\0' || c == dirquoted)
          {
            if (!baddir)
            {
              if (dirlen == 0)
                fullpathbuf[dirlen++] = '.';
              if ( fullpathbuf[dirlen-1] != '/' )
                fullpathbuf[dirlen++] = '/';
              if (dirlen==2 && fullpathbuf[0]=='.' && fullpathbuf[1]=='/')
              {
                if (donecurrdir)
                  dirlen = 0;
                else
                  donecurrdir = 1;        
              }
              if (dirlen!=0 && (cmdlen + dirlen) < (sizeof(fullpathbuf)-1))
              {
                strcpy( &fullpathbuf[dirlen], procnamebuf );
                /* DEBUG_OUT(("validate_app_name(compound): \"%s\"\n",fullpathbuf)); */
                if (access( fullpathbuf, 0 /*X_OK*/) == 0)
                  validated = 1;
              }
            }  
            if (validated || *dirlist == '\0')
              break;
            dirlen = 0;        
            dirquoted = 0;
            baddir = 0;
          }
          else if (!baddir && (cmdlen + dirlen) < (sizeof(fullpathbuf)-1))
          {
            if (dirlen == 0 && (c == '\"' || c == '\''))
              dirquoted = c;
            else if (!dirquoted && (c == ' ' || c == '\t'))
              baddir = 1;
            else  
              fullpathbuf[dirlen++] = c;
          }
        } /* while (*dirlist && !validated) */
      }        /* if (dirlist) */
    }  /* if (!gotpath) */
    if (validated)
      return appname;
  } /* if (__lookup_processor_appname[0]) */
  return (const char *)0;
}  

static int do_ext_proc( struct gen_hostent_data *_r_buffer, 
                        const char *hostname, int reverse, 
                        struct hostent **hpp )
{                                
  int proceed_with_next = 1;
  const char *appname = do_validate_extern_procname();
  if (appname)
  {          
    do_nslookup(_r_buffer,hostname,appname,reverse,hpp);
    proceed_with_next = 0; /* never proceed */
  }        
  return proceed_with_next;
}

/* ---------------------------------------------------------------------- */

static int do_builtins( struct gen_hostent_data *_r_buffer,
                        const char *hostname, int reverse, 
                        struct hostent **hpp )
{
#ifndef INADDR_LOOPBACK
#define INADDR_LOOPBACK 0x0100007FUL
#endif
  static struct __builtins {   const char *name; unsigned long addr; } 
                builtins[]={ { "localhost"     , INADDR_LOOPBACK },
                             { "lb"            , INADDR_LOOPBACK }   };
  int proceed_to_next = 1;                           
  struct in_addr addr_list[(sizeof(builtins)/sizeof(builtins[0]))];
  const char *alias_list[(sizeof(builtins)/sizeof(builtins[0]))];
  unsigned int addr_count = 0, alias_count = 0;
  unsigned int idx; unsigned long addr; int found;

  TRACE_OUT(("doing do_builtins()\n"));

  found = 1;
  if (reverse)
  {
    found = 0;
    addr = ((const struct in_addr *)hostname)->s_addr;
    for (idx = 0; idx < (sizeof(builtins)/sizeof(builtins[0])); idx++)
    {
      if (addr == builtins[idx].addr)
      {
        hostname = builtins[idx].name;
        found = 1;
        break;
      }
    }
  }
  if (found)
  {
    for (idx = 0; idx < (sizeof(builtins)/sizeof(builtins[0])); idx++)
    {
      unsigned int i; 
      found = 0;  addr = builtins[idx].addr;
      for (i = 0; !found && i < addr_count; i++)
      {
        if ( (addr_list[i].s_addr & 0xffffffffu) == addr )
          found = 1;
      }
      if (found)
      {
        if (alias_count < ((sizeof(alias_list)/sizeof(alias_list[0])) ) )
          alias_list[alias_count++] = builtins[idx].name;
      }
      else if (strcasecmp( builtins[idx].name, hostname ) == 0)
      {
        if (addr_count < ((sizeof(addr_list)/sizeof(addr_list[0])) ) )
          addr_list[addr_count++].s_addr = builtins[idx].addr;
      }
    }
  }    
  TRACE_OUT(("addr_count from do_builtins() = %u\n", addr_count));
  *hpp = do_gen_hostent( _r_buffer, hostname, addr_list, addr_count, 
                                    alias_list, alias_count, 0 );
  proceed_to_next = 1;
  if (addr_count)                                    
    proceed_to_next = 0;                                    
  return proceed_to_next;
}

/* ---------------------------------------------------------------------- */

static int match_hostname(const char *hostname1, const char *hostname2)
{
  if (strcasecmp(hostname1, hostname2) == 0)
    return 1;
  /* I originally thought the domain name was also part of the
   * hostname matching when comparing two names (one from user, 
   * the other from /etc/hosts), but have since found that isn't so.
   * So, this function is now small and simple, and /etc/resolv.conf
   * doesn't need to be parsed.
  */ 
  return 0;
}

/* ---------------------------------------------------------------------- */

static int do_etc_hosts( struct gen_hostent_data *_r_buffer, 
                         const char *hostname, int reverse, 
                         struct hostent **hpp )
{
  int proceed_to_next = 1;
  struct in_addr addr_list[1]; /* addresses are unique */
  unsigned int addr_count = 0, alias_count = 0;
  const char *alias_list[6];
  char alias_buf[256+1];
  FILE *file = fopen("/etc/hosts", "r");
    
  TRACE_OUT(("doing do_etc_hosts() file=%p\n",file));
  if (file)
  {
    char tokbuf[256+1];
    size_t alias_used = 0, toklen = 0;
    int perline_alias_count = 0;
    struct in_addr addr;
    
    addr.s_addr = 0;  
    for (;;)
    {
      int c = fgetc(file);
/* printf("%c", c); fflush(stdout); */
      if (c == '#' || c == ' ' || c == '\t' || c == '\n' || c == EOF)
      {
        if (toklen > 0 && toklen < sizeof(tokbuf))
          tokbuf[toklen++] = '\0';
        if (toklen < 1)
          ; /* nothing */
        else if (toklen >= sizeof(tokbuf)) /* token too long */
        {
          if (c == ' ' || c == '\t')
            c = '#'; /* ignore to eol */
        }
        else if (addr.s_addr != 0) /* name. addr precedes this token */
        {
          DEBUG_OUT(("etc_hosts: name='%s'\n", tokbuf));
          if (!reverse &&
              addr_count < (sizeof(addr_list)/sizeof(addr_list[0])) &&
              match_hostname(hostname, tokbuf))
          {
            DEBUG_OUT(("etc_hosts: FOUND! name='%s'\n", tokbuf));
            addr_list[addr_count++].s_addr = addr.s_addr;
            if (alias_count > 0) /* follow ISC's example of setting hostname */
            {                    /* to the first name on the line */
              const char *swapit = hostname;
              hostname = alias_list[0];
              alias_list[0] = swapit;
            }
          }
          else if (alias_count < (sizeof(alias_list)/sizeof(alias_list[0]))
                && (alias_used+toklen+2) < sizeof(alias_buf))
          {
#if 0 /* ISC resolver doesn't do this, so why should we? */
            int i;
            for (i=0; i < alias_count; i++)
            {
              if (match_hostname( alias_list[i], tokbuf ))
              {
                toklen = 0;
                break;
              }
            }
#endif
            if (toklen)
            {          
              DEBUG_OUT(("etc_hosts: may be alias='%s'\n", tokbuf));
              alias_list[alias_count++]=strcpy(&alias_buf[alias_used],tokbuf);
              alias_used+=toklen+1;
              if (reverse && perline_alias_count == 0)
              {
                DEBUG_OUT(("etc_hosts: FOUND! addr='%s'\n", inet_ntoa(addr)));
                addr_list[addr_count++].s_addr = addr.s_addr;
              }
              perline_alias_count++;
            }  
            /* reverse switch of hostname and address is done later */
          }
        }  
        else if (c == ' ' || c == '\t') /* addr. name(s) follow this token */
        {
          alias_used = alias_count = 0;
          addr.s_addr = inet_addr(tokbuf);
          DEBUG_OUT(("etc_hosts: addr='%s'\n", inet_ntoa(addr)));
          if (addr.s_addr == 0xffffffffu)
            addr.s_addr = 0;
          if (addr.s_addr != 0)
          {
            if (reverse)
            {
              if (addr.s_addr != (((const struct in_addr *)hostname)->s_addr))
                addr.s_addr = 0;
            }
            else
            {
              unsigned int i;
              for (i=0; i < addr_count; i++)
              {
                if (addr_list[i].s_addr == addr.s_addr)
                {
                  addr.s_addr = 0;
                  break;
                }  
              }        
            }  
          }
          if (addr.s_addr == 0)
            c = '#';  
        }      
        if (c == '#')
        {
          while (c != EOF && c != '\n')
            c = fgetc(file);
        }
        if (c == EOF)
          break;
        if (c == '\n')
        {
          if (addr_count) /* addresses are unique */
            break;
          perline_alias_count = 0;
          addr.s_addr = 0;
        }  
        toklen = 0;
      }  
      else if (toklen < sizeof(tokbuf))
        tokbuf[toklen++] = (char)c;
      else 
        toklen = 0;    
    } /* for (;;) */
    fclose(file);
  } /* if file */

  if (addr_count && reverse)
  {
    if (!alias_count)
      addr_count = 0;
    else
    {
      unsigned int i;
      hostname = alias_list[0];
      for (i = 1; i < alias_count; i++)
        alias_list[i-1] = alias_list[i];
      alias_count--;    
    }  
  }
  TRACE_OUT(("addr_count from do_etc_hosts() = %u\n", addr_count));
  *hpp = do_gen_hostent( _r_buffer, hostname, addr_list, addr_count, 
                                    alias_list, alias_count, 0 );
  proceed_to_next = 1;
  if (addr_count)                                    
    proceed_to_next = 0;                                    
  return proceed_to_next;
}

/* ---------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif
extern struct hostent *_gethostbydnsaddr(const char *, int len, int af);
extern struct hostent *_gethostbydnsname(const char *, int af);
extern struct hostent *_nss_dns_gethostbyaddr_r(const char *, int len, int af,
                      struct hostent *result, 
                      char *buffer, int buflen, int *h_errnop );
extern struct hostent *_nss_dns_gethostbyname2_r(const char *, int af,
                      struct hostent *result, 
                      char *buffer, int buflen, int *h_errnop );
#ifdef __cplusplus
}
#endif

static int do_res_xxx( struct gen_hostent_data *_r_buffer,
                        const char *hostname, int reverse, 
                        struct hostent **hpp )
{
#if defined(NOWEAKS) || !defined(__GNUC__) /* we need '#pragma weak' here */
  TRACE_OUT(("skipping do_res_weaks()\n"));
  *hpp = do_gen_hostent( _r_buffer, hostname, 0, 0, 0, 0, 0 );
  reverse = reverse; /* unused */
  return 1;
#else
  int proceed_to_next = 1;
  int remap_static = 0;
  struct hostent *hp = (struct hostent *)0;

  TRACE_OUT(("doing do_res_weaks()\n"));
  if (reverse)
  {
#   pragma weak _nss_dns_gethostbyaddr_r
#   pragma weak _gethostbydnsaddr
    if (_nss_dns_gethostbyaddr_r) /* GLIBC - is this correct? */
    {
      TRACE_OUT(("doing _nss_dns_gethostbyaddr_r()\n"));
      *hpp = _nss_dns_gethostbyaddr_r( hostname, 4, AF_INET,
                                     _r_buffer->result, _r_buffer->buffer, 
                                     _r_buffer->buflen, _r_buffer->h_errnop );
      proceed_to_next = 1;
      remap_static = 0;
    }
    else if (_gethostbydnsaddr) /* BSD */
    {
      proceed_to_next = 0;
      TRACE_OUT(("doing _gethostbydnsaddr()\n"));
      hp = _gethostbydnsaddr(hostname, 4, AF_INET);
      proceed_to_next = 0; /* always works. no need to continue */
      remap_static = 1;
    }        
  }
  else /* if (!reverse) */
  {
#   pragma weak _nss_dns_gethostbyname2_r
#   pragma weak _gethostbydnsname
    if (_nss_dns_gethostbyname2_r) /* GLIBC - is this correct? */
    {
      TRACE_OUT(("doing _nss_dns_gethostbyname2_r()\n"));
      *hpp = _nss_dns_gethostbyname2_r( hostname, AF_INET,
                                     _r_buffer->result, _r_buffer->buffer, 
                                     _r_buffer->buflen, _r_buffer->h_errnop );
      proceed_to_next = 1;
      remap_static = 0;
    }
    else if (_gethostbydnsname) /* BSD */
    {
      TRACE_OUT(("doing _gethostbydnsname()\n"));
      hp = _gethostbydnsname(hostname, AF_INET);
      proceed_to_next = 0; /* always works. no need to continue */
      remap_static = 1;
    }        
  }
  if (hp && remap_static)
  {
    char *dummy = (char *)0;
    unsigned int addr_count = 0;
    unsigned int alias_count = 0;
    if (hp->h_name)
      hostname = hp->h_name;
    if (!hp->h_addr_list)
      hp->h_addr_list = &dummy;
    while (hp->h_addr_list[addr_count])
      addr_count++;
    if (!hp->h_aliases)
      hp->h_aliases = &dummy;
    while (hp->h_aliases[alias_count])
      alias_count++;
    TRACE_OUT(("addr_count from do_res_weaks() = %u\n", addr_count));
    *hpp = do_gen_hostent( _r_buffer, hostname, 
                      (const struct in_addr *)hp->h_addr_list[0], addr_count, 
                      (const char **)hp->h_aliases, alias_count, 0 );
  }
  TRACE_OUT(("done do_res_weaks()%s\n",proceed_to_next?" [No support]":""));
  return proceed_to_next;
#endif  /* defined(__GNUC__) */
}

/* ---------------------------------------------------------------------- */

static const char *validate_hostname(const char *hostname, int reverse)
{
  if (!hostname)
  {
    DEBUG_OUT(("hostname/addr rejected. (its NULL)\n"));
  }
  else if (reverse)
  {
    if ( (((const struct in_addr *)hostname)->s_addr) == 0xffffffffu ||
         (((const struct in_addr *)hostname)->s_addr) == 0 )
    {
      DEBUG_OUT(("addr rejected. (addr is ADDR_ANY or ADDR_BCAST)\n"));
      hostname = (const char *)0;
    }         
  }    
  else if (!*hostname)
  {
    DEBUG_OUT(("hostname rejected. (its empty)\n"));
    hostname = (const char *)0;
  }  
  else if (inet_addr(hostname) != 0xffffffffu) /* don't support IP addresses */
  {                                            /* has this policy changed? */
    DEBUG_OUT(("hostname rejected. (its an IP address)\n"));
    hostname = (const char *)0;
  }    
  else  
  {
    const char *q = hostname;
    while (*q)
    {
      int c = (char)*q++;    
      if (c != '.' && c != '_' && c != '-' && !isalpha(c) && !isdigit(c))
      {
        DEBUG_OUT(("hostname rejected. (has bad chars)\n"));
        hostname = (const char *)0;
        break;
      }
    }          
  }      
  return hostname;
}        

/* ---------------------------------------------------------------------- */

static struct hostent *__gethostbyXXYY__( int reverse,
              const char *hostname, int len, int af, int reentrant,
              struct hostent *result, char *buffer, int buflen, int *h_errnop )
{
  struct gen_hostent_data r_buf;
  struct hostent *hp = (struct hostent *)0;

#define STRINGIFY(x) STRINGIFY2(x)
#define STRINGIFY2(x) #x
  TRACE_OUT((__FILE__":"STRINGIFY(__LINE__)": gethostby%s%s%s() called.\n", 
                                          ((reverse)?("addr"):("name")),
                                          ((!reverse && af)?("2"):("")), 
                                                                        ((reentrant)?("_r"):("")) ));
  if (!reentrant)
  {
    if (!__get_local_hostent_data( &r_buf ))
      return (struct hostent *)0; /* errno/h_errno is already set */
  }    
  else if (!__apply_r_func_params_to_hostent( &r_buf, result, buffer, 
                                              buflen, h_errnop ))
  {                                              
    return (struct hostent *)0; /* errno/h_errno is already set */
  }    

  if (af == AF_INET && len == -1)
    len = sizeof(struct in_addr);
  else if (af == 0 /* AF_UNSPECIFIED */)
  {
    if (((unsigned int)len) == sizeof(struct in_addr))
      af = AF_INET;
  }      
  if (af != AF_INET || ((unsigned int)len) != sizeof(struct in_addr))
  {
#if defined(EAFNOSUPPORT)
    __apply_h_errno_to_hostent( &r_buf, -1, EAFNOSUPPORT );
#else
    __apply_h_errno_to_hostent( &r_buf, -1, EINVAL );
#endif
  }
  else if (!validate_hostname( hostname, reverse ))
  {
    __apply_h_errno_to_hostent( &r_buf, -1, EINVAL );
  }
  else
  {
    static const unsigned int run_order[] = {'h','b','v','n'}; 
    int proceed_to_next = 1;  /* /etc/hosts,builtins,vixie,nslookup */
    unsigned int order; /* ISC does builtins last, we do it after /etc/hosts */

    /* first try custom configured external lookup processor */
    /* if the proc exists, quit now, irrespective of success/failure to resolv */
    proceed_to_next = do_ext_proc( &r_buf, hostname, reverse, &hp );

    for (order = 0; proceed_to_next &&
          (((unsigned int)order) < (sizeof(run_order)/sizeof(run_order[0]))) ;
          order++)
    {
      TRACE_OUT(("run_order[%d]=%c\n", order, run_order[order]));
      switch (run_order[order])
      {
        case 'h': /* /etc/hosts */
          proceed_to_next = do_etc_hosts( &r_buf, hostname, reverse, &hp);
          break;
        case 'b': /* builtins */  
          proceed_to_next = do_builtins( &r_buf, hostname, reverse, &hp);
          break;
        case 'v': /* vixie/berkeley */
          proceed_to_next = do_res_xxx( &r_buf, hostname, reverse, &hp );
          break;
        case 'n': /* nslookup */
          proceed_to_next = do_nslookup( &r_buf, hostname, 0, reverse, &hp );
          break;
        default:
          hp = (struct hostent *)0;
          proceed_to_next = 0;
          break;
      }
    }      
  }    
  return hp;
}      

/* ---------------------------------------------------------------------- */

#if defined(__GNUC__) && !defined(CMPTEST)
  /* generate public symbols and simultaneously make the other 
     (library) ones weak, so that ld doesn't complain.
  */
#  if defined(__ELF__)
#    define _C_SYM(x) #x
#  else
#    define _C_SYM(x) "_"#x
#  endif
#  define weak_alias(original, alias) \
      __asm__ (".weak " _C_SYM(alias) "\n" _C_SYM(alias) " = " _C_SYM(original) "\n")
#  define weak_extern(extsymbol) \
    __asm__ (".weak " _C_SYM(extsymbol) "\n" ) /* gcc 2.8+ has .weakext */
#  if 1  
#    define strong_alias(original, alias)        \
      __asm__ (".globl " _C_SYM(alias) "\n .set " _C_SYM(alias) "," _C_SYM(original) "\n" )
#  else  
#    define strong_alias(original, alias)        \
      __asm__ (".globl " _C_SYM(alias) "\n " _C_SYM(alias) " = " _C_SYM(original) "\n" )
#  endif  
  /* don't be tempted to make the functions static. -O will optimize them away */
#  if defined(__cplusplus)
#    define __GETHOSTBY(x,y,z) \
            weak_extern( x ); strong_alias( y, x ); \
            extern "C" struct hostent * ##y ##z ; \
            struct hostent * ##y ##z
     extern "C" int __gen_h_errno;
#  else            
#    define __GETHOSTBY(x,y,z) \
            weak_extern( x ); strong_alias( y, x ); \
            extern struct hostent * ##y ##z ; \
            struct hostent * ##y ##z
#  endif            
   int __gen_h_errno;
#  if !defined(_REENTRANT) /* otherwise h_errno is redefined to be a pointer */
   weak_alias(__gen_h_errno, h_errno);
#  endif
#else /* !defined(__GNUC__) */
#  if defined(__cplusplus)
#     define __GETHOSTBY(x,y,z) \
             extern "C" struct hostent * ##x ##z ; \
             struct hostent * ##x ##z
#  if !defined(h_errno)             
   extern "C" int h_errno;  
#  endif
#  else            
#     define __GETHOSTBY(x,y,z) \
            extern struct hostent * ##x ##z ; \
            struct hostent * ##x ##z
#  endif            
#  if !defined(h_errno) /* redefined for thread safe code */
   int h_errno;
#  endif
#endif /* defined(__GNUC__) */ 

#ifdef CMPTEST /* if 'CMPTEST' is defined, call the functions 'test_*' */
#define _GETHOSTBY(x,z) __GETHOSTBY(test_gethostby##x , __gen_gethostby##x, ##z )
#else
#define _GETHOSTBY(x,z) __GETHOSTBY(gethostby##x , __gen_gethostby##x, ##z )
#endif

/* ---------------------------------------------------------------------- */

_GETHOSTBY(name2_r, ( const char *h, int af, struct hostent *result, 
                      char *buffer, int buflen, int *h_errnop ))
{
  return __gethostbyXXYY__( 0, h, -1, af, 1,result,buffer,buflen,h_errnop );
}
_GETHOSTBY(name_r,  ( const char *h, struct hostent *result, 
                      char *buffer, int buflen, int *h_errnop ))
{
  return __gethostbyXXYY__( 0, h, 4, 0,  1,result,buffer,buflen,h_errnop );
}
_GETHOSTBY(name2, ( const char *hostname, int af ))
{
  return __gethostbyXXYY__( 0, hostname, -1, af,    0,0,0,0,0 );
}
_GETHOSTBY(name,  ( const char *hostname ))
{
  return __gethostbyXXYY__( 0, hostname, 4, 0,   0,0,0,0,0 );
}
_GETHOSTBY(addr_r,  ( const char *a, int len, int af, struct hostent *result, 
                      char *buffer, int buflen, int *h_errnop ))
{
  return __gethostbyXXYY__( 1, a, len, af,  1,result,buffer,buflen,h_errnop);
}
_GETHOSTBY(addr,    ( const char *addr, int len, int af ))
{
  return __gethostbyXXYY__( 1, addr, len, af,   0,0,0,0,0 );
}

/* ---------------------------------------------------------------------- */

