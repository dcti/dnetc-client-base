/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * This file contains functions used from ./configure
 * Specify 'build_dependancies' as argument 
 * (which is all this needs to do anymore)
 *
 * $Id: testplat.cpp,v 1.4.2.4 2001/01/28 00:57:05 cyp Exp $
*/ 
#include <stdio.h>
#include <string.h>

static const char *include_dirs[] = { "common", 
                                      "rc5", 
                                      "des", 
                                      "ogr/ansi"
                                     };

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
  foundbuf = foundbuf;
  #if defined(__riscos)
  unsigned int n;
  for (n=0; foundbuf[n]; n++)
  {
    if (foundbuf[n] == '.')
      foundbuf[n] = '/';
    else if (foundbuf[n] == '/')
      foundbuf[n] = '.';  
  }
  #endif
}

static unsigned int build_dependancies( char *cppname ) /* ${TARGETSRC} */
{
  char linebuf[512], pathbuf[64], foundbuf[64];
  char *p, *r;
  unsigned int l, count = 0;
  FILE *file = fopen( cppname, "r" );

  //fprintf(stderr,"cppname='%s', file=%p\n",cppname,file);

  if ( file )
  {
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
          //fprintf(stderr, "'#include %s'\n", linebuf);
          if (linebuf[0])
          {
            strcpy( foundbuf, linebuf );
            /* include specifiers are always unix form */
            if ( strchr( linebuf, '/' ) == NULL )
            {
              strcpy( foundbuf, pathbuf );
              strcat( foundbuf, linebuf );
              l = 0;
              //fprintf(stderr, "%d) '%s'\n", l, foundbuf);
              for (;;)
              {
                __fixup_pathspec_for_locate(foundbuf);
                if (fileexists( foundbuf ))
                  break;
                if (l >= (sizeof( include_dirs )/sizeof( include_dirs[0] )))
                  break;
                strcpy( foundbuf, include_dirs[l] );
                strcat( foundbuf, "/" ); /* always unix */
                strcat( foundbuf, linebuf );
                l++;
                //fprintf(stderr, "%d) '%s'\n", l, foundbuf);
              }
            }
            if ( !fileexists( foundbuf ) )
            {
              strcpy(foundbuf,linebuf);
            }
            __fixup_pathspec_for_makefile(foundbuf);
            printf( "%s%s", ((count!=0)?(" "):("")), foundbuf );
            count++;
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
    fprintf(stderr,"Specify 'build_dependancies' as argument.\n");
    return -1;
  }
  if (strcmp(argv[1], "build_dependancies" )== 0)
    build_dependancies( argv[2] );    
  return 0;
}
