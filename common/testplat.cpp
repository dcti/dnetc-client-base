/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This file contains functions used from ./configure
 * Specify 'build_dependancies' as argument 
 * (which is all this needs to do anymore)
 *
 * $Id: testplat.cpp,v 1.4.2.7 2001/04/07 16:15:39 cyp Exp $
*/ 
#include <stdio.h>   /* fopen()/fclose()/fread()/fwrite()/NULL */
#include <string.h>  /* strlen()/memmove() */
#include <stdlib.h>  /* malloc()/free()/atoi() */


static int fileexists( const char *filename ) /* not all plats have unistd.h */
{
  FILE *file = fopen( filename, "r" );
  if (file) fclose( file );
  return ( file != NULL );
}

static void __fixup_pathspec_for_locate(char *foundbuf)
{
  foundbuf = foundbuf;
  #if defined(__riscos)
  unsigned int n;
  for (n=0; foundbuf[n]; n++)
  {
    if (foundbuf[n] == '/') /* dirsep */
      foundbuf[n] = '.';
    else if (foundbuf[n] == '.') /* suffix */
      foundbuf[n] = '/';  
  }
  #endif
}

static void __fixup_pathspec_for_makefile(char *foundbuf)
{
  #if defined(__riscos)
  unsigned int n;
  for (n=0; foundbuf[n]; n++)
  {
    if (foundbuf[n] == '.')
      foundbuf[n] = '/';
    else if (foundbuf[n] == '/')
      foundbuf[n] = '.';  
  }
  #else
  if (foundbuf[0] == '.' && foundbuf[1] == '/')
    memmove( foundbuf, &foundbuf[2], strlen(&foundbuf[2])+1 );
  #endif
}

static int is_trace_checked(const char *filename)
{
  return 0;
  //return (strcmp(filename,"ogr/x86/ogr-a.cpp") == 0 
  //     || strcmp(filename,"ogr/x86/ogr-b.cpp") == 0);
}

static unsigned int build_dependancies( const char *cppname, /* ${TARGETSRC} */
                                        const char **include_dirs )
{
  char linebuf[512], pathbuf[64], foundbuf[64];
  char *p, *r;
  unsigned int l, count = 0;
  FILE *file = fopen( cppname, "r" );

  //fprintf(stderr,"cppname='%s', file=%p\n",cppname,file);

  if ( file )
  {
    int debug = is_trace_checked(cppname);

    strcpy( pathbuf, cppname );
    p = strrchr( pathbuf, '/' ); /* input specifiers are always unix paths */
    if ( p == NULL ) 
      pathbuf[0]=0;
    else *(++p)=0;

    while ( fgets( linebuf, sizeof( linebuf ), file ) != NULL )
    {
      p = linebuf;
      while ( *p == ' ' || *p == '\t' )
        p++;
      if ( *p == '#' && strncmp( p, "#include", 8 ) == 0 && 
                              (p[8]==' ' || p[8] =='\t'))
      {
        p+=8;
        while ( *p == ' ' || *p == '\t' )
          p++;
        if ( *p == '\"' /* || *p == '<' */ )
        {
          r = linebuf;
          foundbuf[0]= ((*p == '<') ? ('>') : ('\"'));
          while (*(++p) != foundbuf[0] )
            *r++ = *p;
          *r = 0;
          if (debug)
            fprintf(stderr, "'#include %s'\n", linebuf);
          if (linebuf[0])
          {
            strcpy( foundbuf, linebuf );
            /* include specifiers are always unix form */
            //if ( strchr( linebuf, '/' ) == NULL )
            {
              strcpy( foundbuf, pathbuf );
              strcat( foundbuf, linebuf );
              l = 0;
              for (;;)
              {
                char origbuf[sizeof(foundbuf)];
                strcpy(origbuf, foundbuf);
                __fixup_pathspec_for_locate(foundbuf);
                if (debug)
                  fprintf(stderr, "%d) '%s'\n", l, foundbuf);
                if (fileexists( foundbuf ))
                {
                  int namelen = strlen(origbuf);
                  if (namelen > 2 && strcmp(&origbuf[namelen-2],".h")!=0)
                    count += build_dependancies( origbuf, include_dirs );
                  break;
                }         
                if (!include_dirs)
                  break;
                if (!include_dirs[l])
                  break;
                strcpy( foundbuf, include_dirs[l] );
                if (foundbuf[strlen(foundbuf)-1] != '/') /* always unix */
                  strcat( foundbuf, "/" ); /* always unix */
                strcat( foundbuf, linebuf );
                l++;
              }
            }
            if ( fileexists( foundbuf ) )
            {
              __fixup_pathspec_for_makefile(foundbuf);
              printf( "%s%s", ((count!=0)?(" "):("")), foundbuf );
              count++;
            }  
          }
        }
      }
    }  
    fclose(file);
  }
  printf("\n");
  return count;
}      

static const char **get_include_dirs(int argc, char *argv[])
{
  char **idirs; int i;
  unsigned int bufsize = 0;
  for (i = 0; i < argc; i++)
    bufsize += strlen(argv[i])+5;
  idirs = (char **)malloc((argc*sizeof(char *)) + bufsize);
  if (idirs)
  {
    int numdirs = 0, in_i_loop = 0;
    char *buf = (char *)idirs;    
    buf += (argc * sizeof(char *));
    for (i = 1; i < argc; i++)
    {
      const char *d = argv[i];
      //fprintf(stderr,"d='%s'\n",d);
      int is_i = in_i_loop;
      in_i_loop = 0;
      if (*d == '-' && d[1] != 'I')
        is_i = 0;
      else if (*d == '-' && d[1] == 'I')
      {
        d += 2;
        is_i = (*d != '\0');
        in_i_loop = !is_i;
      }
      if (is_i)
      {
        while (*d)
        {
          const char *s = d;
          while (*d && *d != ':')
            d++;
          if (d != s)
          {
            idirs[numdirs++] = buf;  
            while (s < d)
              *buf++ = *s++;
            if ((*--s) != '/')  
              *buf++ = '/';
            *buf++ = '\0';
            //fprintf(stderr,"\"-I%s\"\n", idirs[numdirs-1]);
          }
          while (*d == ':')
            d++;
        } /* while (*d) */          
      }  /* if (is_i) */
    } /* for (i = 1; i < argc; i++) */
    idirs[numdirs] = (char *)0;
  } /* if (idirs) */
  return (const char **)idirs;
}  


int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    fprintf(stderr,"Specify 'build_dependancies' as argument.\n");
    return -1;
  }
  if (strcmp(argv[1], "build_dependancies" )== 0)
  {
    const char **idirs;
    //fprintf(stderr,"%s 1\n", argv[2] );
    idirs = get_include_dirs(argc,argv);
    //fprintf(stderr,"%s 2\n", argv[2] );
    build_dependancies( argv[2], idirs );    
    //fprintf(stderr,"%s 3\n", argv[2] );
    if (idirs) 
      free((void *)idirs);
  }  
  return 0;
}
