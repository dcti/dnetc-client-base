#include <stdio.h>
#include <sys/types.h>
#include <netdb.h>       /* struct hostent */
#include <netinet/in.h>  /* struct in_addr */
#include <arpa/inet.h>   /* inet_ntoa, inet_addr */

extern void set_lookup_processor_appname(const char *);
/* if CMPTEST is defined, gethostbyXX() is vixie code and */
/*                      test_gethostbyXX() is exported by resolv.c */
/* otherwise, there is only the gethostbyXX() from resolv.c */
extern struct hostent *test_gethostbyname(const char *);
extern struct hostent *test_gethostbyaddr(const char *, int, int);

static void _set_lookup_proc(const char *argv0)
{
  #ifdef USE_EXTERN 
  char appname[128];
  char *app = &appname[strlen(strcpy( appname, argv0 ))];
  while (app > appname )
  {
    if (*--app == '/')
    {
      app++;
      break;
    }	
  }
  strcpy( app, "resolv.sh" );
  set_lookup_processor_appname(appname);
  #endif
  argv0 = argv0;
  return;
}

static void do_ghbn(const char *prefix, void *proc, int reverse,
                    const char *hostname )
{
  struct hostent *hp;
  int count;
  printf("----------- run %s ------------------\n", prefix );

  if (reverse)
    hp = (*((struct hostent *(*)(const char *,int, int))proc))(hostname,4,2);
  else
    hp = (*((struct hostent *(*)(const char *))proc))( hostname );    
    
  printf("------------eval %s -------\n  hostname: ", prefix );
  if (hp)
  {
    if (hp->h_name)
      printf("\"%s\"", hp->h_name );
  }
  printf("\n  alias(es): ");
  if (hp)
  {
    if (hp->h_aliases)
    {
      count = 0;
      while (hp->h_aliases[count])
      {
        printf("%s\"%s\"", ((count==0)?(""):(", ")), hp->h_aliases[count] );
        count++;
      }       
    }  
  }
  printf("\n  address(es): ");
  if (hp)
  {
    if (hp->h_addr_list)
    {
      count = 0;
      while (hp->h_addr_list[count])
      {
        printf("%s%s", ((count==0)?(""):(" ")),
	             inet_ntoa(*((struct in_addr *)hp->h_addr_list[count])) );
	count++;	     
      }
    }  
  }
  printf("\n-------------- end %s ----------------------\n", prefix );
  return;
}  


int main(int argc, char *argv[])
{
  struct in_addr addr;
  char *host = (char *)0;
  host = "fb14.uni-mainz.de";      /* single "A" test */
  host = "us.v27.distributed.net"; /* multi-"A" test */
  host = "ftp.uni-mainz.de";       /* "CNAME" test */
  host = "localhost";              /* /etc/hosts (or builtins) test */
  host = "127.0.0.1";              /* /etc/hosts (or builtins) test */
  
  _set_lookup_proc(argv[0]);
  if (argc > 1)
    host = argv[1];
  if (!host)
  {
    printf("Syntax: %s <hostname|hostaddr>\n",argv[0]);
    return 1;
  }        
  
  addr.s_addr = inet_addr(host);
  if (addr.s_addr == 0xffffffffu)
    addr.s_addr = 0;
  if (addr.s_addr == 0)
  {
    #if defined(CMPTEST)
    do_ghbn("vixie code", (void *)gethostbyname, 0, host );
    do_ghbn("resolv.c",   (void *)test_gethostbyname, 0, host );
    #else
    do_ghbn("resolv.c",   (void *)gethostbyname, 0, host );
    #endif
  }
  else
  {
    host = (char *)&(addr.s_addr);
    #if defined(CMPTEST)
    do_ghbn("vixie code", (void *)gethostbyaddr, 1, host );
    do_ghbn("resolv.c",   (void *)test_gethostbyaddr, 1, host );
    #else
    do_ghbn("resolv.c",   (void *)gethostbyaddr, 1, host );
    #endif
  }    

  return 0;
}


