/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This file contains functions used from ./configure
 * Specify 'cpu', 'os', 'intsizes' or 'build_dependancies' as argument.
*/ 
const char *testplat_cpp(void) { 
return "@(#)$Id: testplat.cpp,v 1.4 1999/04/05 17:56:52 cyp Exp $"; } 

static const char *include_dirs[] = { "common", "rc5", "des", "ogr" };

#include <stdio.h>
#include <string.h>
#define IGNOREUNKNOWNCPUOS
#include "cputypes.h"

#if (CLIENT_OS == OS_RISCOS)
#define INCLUDE_SUBDIR_SEP "."
#else
#define INCLUDE_SUBDIR_SEP "/"
#endif

static int fileexists( const char *filename ) /* not all plats have unistd.h */
{
  FILE *file = fopen( filename, "r" );
  if (file) fclose( file );
  return ( file != NULL );
}

static unsigned int build_dependancies( char *cppname ) /* ${TARGETSRC} */
{
  char linebuf[512], pathbuf[64], foundbuf[64];
  char *p, *r;
  unsigned int l, count = 0;
  FILE *file = fopen( cppname, "r" );
  const char *subdir_sep = INCLUDE_SUBDIR_SEP;

  if ( file )
  {
    strcpy( pathbuf, cppname );
    p = strrchr( pathbuf, *subdir_sep );
    r = strrchr( pathbuf, '\\' );
    if ( r > p ) p = r;
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
        if ( *p == '\"' || *p == '<' )
        {
          r = linebuf;
	  foundbuf[0]= ((*p == '<') ? ('>') : ('\"'));
          while (*(++p) != foundbuf[0] )
            *r++ = *p;
          *r = 0;
          if (linebuf[0])
          {
            strcpy( foundbuf, linebuf );
            if ( strchr( linebuf, *subdir_sep ) == NULL )
            {
              strcpy( foundbuf, pathbuf );
              strcat( foundbuf, linebuf );
              l = 0;
              while ( !fileexists( foundbuf ) )
              {
                if (l >= (sizeof( include_dirs )/sizeof( include_dirs[0] )))
                  break;
                strcpy( foundbuf, include_dirs[0] );
		strcat( foundbuf, subdir_sep );
                strcat( foundbuf, linebuf );
                l++;
              }
            }
            if ( fileexists( foundbuf ) )
            {
              if ( count != 0 )
                putc( ' ', stdout );
              printf( "%s", foundbuf );
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

int main(int argc, char *argv[])
{
  if (argc < 2)
  {
    printf("Specify 'cpu', 'os', 'intsizes' or 'build_dependancies' as argument.\n");
    return -1;
  }
  if (strcmp(argv[1], "cpu") == 0)
    printf("%i\n", (int) CLIENT_CPU);
  else if (strcmp(argv[1], "os") == 0)
    printf("%i\n", (int) CLIENT_OS);
  else if (strcmp(argv[1], "intsizes") == 0)
    printf("%i%i%i\n", (int) sizeof(long), (int) sizeof(int), (int) sizeof(short));
  else if (strcmp(argv[1], "build_dependancies" )== 0)
    build_dependancies( argv[2] );    
  return 0;
}

