/*
 * .ini (configuration file ala windows) file read/write[parse] routines
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>, no copyright, 
 * and no restrictions on usage in any form or function :)
 *
 * Unlike native windows' this version does not cache the file
 * separately, but uses buffered I/O and assumes the OS knows what 
 * file caching is. It does pass all windows compatibility tests I
 * could come up with (see table below).
 *
 * Unlike a cpp file of a similar name, this .ini file parser is pure
 * C, is portable, passes -pedantic tests, does not 'new' a bazillion 
 * and one times, will not die if mem is scarce, does not fragment 
 * memory, does not attempt to parse comments, will not choke on
 * crlf translation, can somewhat deal with no-disk space conditions, 
 * and does not assume that the calling function was written by a twit. 
 *
*/

const char *iniread_cpp(void) {
return "@(#)$Id: iniread.cpp,v 1.27.2.9 2000/06/14 20:05:35 cyp Exp $"; }

#include <stdio.h>   /* fopen()/fclose()/fread()/fwrite()/NULL */
#include <string.h>  /* strlen()/memmove() */
#include <ctype.h>   /* tolower()/isctrl(). do not use isspace()! */
#include <stdlib.h>  /* malloc()/free()/atoi() */
#include <limits.h>  /* UINT_MAX */
#ifndef _MSC_VER     /* geez, talk about compatibility */
#include <unistd.h>  /* access(), ms-c has this in stdlib.h */
#endif
#include "iniread.h"

#ifndef SEEK_SET   /* some OSs (sunos4) don't have SEEK_* */
#define SEEK_SET 0 /* Seek relative to start of file  */
#define SEEK_CUR 1 /* Seek relative to current positn */
#define SEEK_END 2 /* Seek relative to end of file    */
#endif

#if 0 /* embedded comment handling is not api conform */
#define ALLOW_EMBEDDED_COMMENTS 
#endif

/* ini_doit() functionality:
   section exists:  w create new key+value
                    w delete old key+value(+sect)
                    w replace key+value
                    w delete section
                    r get key+value
                    r get all key+value pairs for section
   does not exist:  w create new sect+key+value
                    w delete old key+value (nop - return ok)
                    w replace key+value  (create)
                    w delete section  (nop - return ok)
                    r get key+value   (nop - return default)
                    r get all key+value pairs for section (nop, return 0)
   null section:    w without key/value [ = flush] (nop - return ok)
                    w with key/value (as above)
*/                            


static unsigned long ini_doit( int dowrite, const char *sect, 
                               const char *key, const char *value,
                               char *buffer, unsigned long bufflen, 
                               const char *filename )
{
  char *data = NULL;
  long i,n,filelen = 0;
  unsigned long success = 0;

  if (dowrite && !sect && !key && !value ) /* flush */
  {
    success = 1;
  }
  else if ((!filename) || (!dowrite && (!buffer || bufflen <2)))
  {
    ; /* go return default (or if dowrite then return fail) */
  }
  else if (access(filename,0)!=0) /* file doesn't exist */
  {
    if (dowrite)
    {
      if (!sect || !key || !value) /* delete something */
      {
        success = 1;
      }
      else
      {
        long malloclen = 16;
        malloclen += (((sect)?(strlen(sect)):(0))+strlen(key)+strlen(value)+3);
        if (((unsigned long)malloclen) < ((unsigned long)(UINT_MAX-128)))
        {
          data = (char *)malloc((int)malloclen);
          if (data)
            memset(data,'\n',((int)malloclen));
        }
        filelen = 0;
      }
    }
  }
  else /* file exists, the fopen() is our write test */
  {
    FILE *file = fopen( filename, ((dowrite)?("r+"):("r")) );
    /* printf("fopen(\"%s\",\"%s\") => %p\n", filename, ((dowrite)?("r+"):("r")), file ); */
    if (file)
    {
      filelen = -1L;
      if ( fseek( file, 0, SEEK_END ) == 0 )
      {
        filelen = ftell( file );
        if (filelen > 0)
        {
          if ( fseek( file, 0, SEEK_SET ) != 0 )
            filelen = -1L;
        }
        /* printf("filelen 1 => %ld\n", filelen ); */
      }
      if (filelen != -1L)
      {
        long malloclen = filelen + 16;
        if (dowrite && key && value)
          malloclen += (((sect)?(strlen(sect)):(0))+strlen(key)+strlen(value)+3);
        /* printf("malloclen: %ld, max: %ld\n", malloclen, (((long)(UINT_MAX))-128) ); */
        if (((unsigned long)malloclen) < ((unsigned long)(UINT_MAX-128)))
        {
          data = (char *)malloc((int)malloclen);
          /* printf("havedata 1: %p\n", data ); */
          if (data)
          {
            memset(data,'\n',((int)malloclen));
            if (filelen > 0)
            {
              i = (long)fread( (void *)data, sizeof(char), filelen, file );
              /* printf("read len: %ld\n", i ); */
              if (i == 0)  /* can't do (i < filelen) because of crlf trans */
              {
                free(data);
                data = NULL;
              }
            }
          }
        }
      }
      fclose(file);
    }
  }

  /* printf("havedata: %p\n", data ); getchar(); */
      
  if (data)
  {
    char quotechar = 0;
    long offset = 0, sectoff = 0, sectoffend = 0, foundrecs = 0;
    int anysect = 0, changed = 0, foundsect = 0;

    if (sect == NULL && filelen)
      foundsect = 1;      

    if (dowrite && key && value && value[0]!='\'' && value[0]!='\"')
    {
      int qn=0;
      for (i=0;quotechar==0 && value[i];i++)
      {
        char c = (char)value[i];
        if (c=='\"' || c=='\'')
          quotechar = ((c=='\"')?('\''):('\"'));
        else if (c == ' ' || c=='\t')
          qn = 1;
        #ifdef ALLOW_EMBEDDED_COMMENTS
        else if (c==';' || c=='#')
          qn = 1;
        #endif
      }
      if (qn && !quotechar)
        quotechar = '\"';
    }

    while (offset < filelen)
    {
      long eollen = 0, keyoff = 0, leadspaces = 0, linelen = 0;
      int c;
      while (offset < filelen)
      {
        c = (int)(data[offset++]);
        if (c=='\r' || c=='\n')
        {
          if (linelen)
            eollen++;
        }
        else if (eollen)
        {
          offset--;
          break;
        }
        else if (linelen == 0 && (c==' ' || c== '\t')) 
        {
          leadspaces++;
        }
        else
        {
          if (linelen == 0)
            keyoff = offset-1;
          linelen++;
        }
      }
      if (linelen)
      {
        /* printf("line: \"%70.70s\"\n", &data[keyoff] ); */

        if (data[keyoff]=='[')
        {
          anysect = 1;
          if (foundsect)
          {
            sectoffend = keyoff;
            break;
          }
          else
          {
            i=1;
            n=0;
            while (i<linelen && (data[keyoff+i]=='\t' || data[keyoff+i]==' '))
              i++;
            while (sect[n]==' ' || sect[n]=='\t')
              n++;
            while (i<(linelen-1) && sect[n] && toupper(sect[n])==toupper(data[keyoff+i]))
            { i++; n++; }
            while (i<(linelen-1) && (data[keyoff+i]==' ' || data[keyoff+i]=='\t'))
              i++;
            while (sect[n]==' ' || sect[n]=='\t')
              n++;
            if (sect[n]==0 && data[keyoff+i]==']')
            {
              sectoff = keyoff - leadspaces;
              foundsect = 1;
            }
          }
        }
        else if (foundsect || (sect == NULL && !anysect))
        {
          long keylen = 0, valuepos = 0;
          int keyfound = 0;
          foundrecs++;

          if (!dowrite || key!=NULL /*!erase sect*/ || !changed /*done 0*/)
          {
            while (keylen<linelen && data[keyoff+keylen]!='=')
              keylen++;
            if (keylen<linelen && data[keyoff+keylen]=='=')
            {
              valuepos = keylen+1;
              while (keylen>0 && (data[keyoff+(keylen-1)]==' ' || 
                                  data[keyoff+(keylen-1)]=='\t'))
                keylen--;
            }
            if (keylen>0 && valuepos>0)
            {
              if (key == NULL)
              {
                if (!dowrite) /* erase section comes later */
                  keyfound = 1;
              }
              else
              {
                i=n=0;
                while (key[n]==' ' || key[n]=='\t')
                  n++;
                while (i<keylen && key[n] && toupper(key[n])==toupper(data[keyoff+i]))
                {i++; n++;}
                while (key[n]==' ' || key[n]=='\t')
                  n++;
                keyfound = (key[n]==0 && i==keylen);
              }
            }
          }
            
          if (keyfound)
          {
            long valuelen=0;
            if (dowrite)
            {
              linelen+=leadspaces+eollen;
              keyoff-=leadspaces;
              memmove( (void *)&data[keyoff], 
                       (void *)&data[keyoff+linelen],
                       filelen-(keyoff+linelen) );
              filelen-=linelen;
              offset-=linelen;
              if (value == NULL) /* delete key+value */
              {
                foundrecs--;     /* the record is gone */
                if (keyoff < filelen && data[keyoff]=='[')
                {                /* need to reinsert a blank line */
                  memmove( (void *)&data[keyoff+1], 
                           (void *)&data[keyoff], 
                           filelen-keyoff );
                  data[keyoff]='\n';
                  filelen++;
                  offset++;
                }
              }
              else               /* insert key+value */
              {
                valuelen = 0;
                while (value[valuelen])
                  valuelen++;
                linelen = keylen+1+valuelen+((quotechar)?(2):(0))+1;
                if (keyoff < filelen)
                {
                  int iseosec = (data[keyoff] == '[');
                  if (iseosec)
                    linelen++;
                  memmove( (void *)&data[keyoff+linelen], 
                           (void *)&data[keyoff],
                           filelen-keyoff );
                  if (iseosec)
                  {
                    linelen--;
                    data[keyoff+linelen]='\n';
                  }
                }
                filelen+=linelen;
                offset+=linelen;
                n=keyoff;
                for (i=0;i<keylen;i++)
                  data[n++]=key[i];
                data[n++]='=';
                if (quotechar)
                  data[n++]=quotechar;
                for (i=0;i<valuelen;i++)
                  data[n++]=value[i];
                if (quotechar)
                  data[n++]=quotechar;
                data[n++]='\n';
              }
              changed = 1;
            }
            else /* if (dowrite) else */
            {
              long valueoff;

              while (valuepos<linelen && 
                 (data[keyoff+valuepos]==' ' || data[keyoff+valuepos]=='\t'))
                valuepos++;
                
              valueoff = keyoff+valuepos;
              valuelen = linelen-valuepos;
              
              if (valuelen == 0)
                ;
              else if (data[valueoff]=='\'' || data[valueoff]=='\"')
              {
                char endquote=data[valueoff++];
                valuelen--;
                for (i=0;i<valuelen;i++)
                {
                  if (data[i+valueoff]==endquote)
                  {
                    valuelen=i;
                    break;
                  }
                }
              }
              else
              {
                #ifdef ALLOW_EMBEDDED_COMMENTS
                for (i=0;i<valuelen;i++)
                {
                  if (data[i+valueoff]==';' || data[i+valueoff]=='#')
                  {
                    valuelen=i;
                    break;
                  }
                }
                #endif
                while (valuelen>0 && (data[(valuelen-1)+valueoff]==' ' || 
                                      data[(valuelen-1)+valueoff]=='\t'))
                  valuelen--;
              }
              if (key == NULL) /* cat all keys */
              {
                if ((success+keylen+valuelen+3) > bufflen)
                  break;
                else
                {
                  for (i=0;i<keylen;i++)
                    buffer[success++]=data[keyoff+i];
                  buffer[success++]='=';
                  for (i=0;success<(bufflen-2) && i<valuelen;i++)
                    buffer[success++]=data[valueoff+i];
                  buffer[success++]=0;
                  buffer[success]=0;
                }
              }
              else
              {
                success=0;
                while (((unsigned int)success)<(bufflen-1) && 
                       ((long)success)<valuelen)
                  buffer[success++]=data[valueoff++];
                buffer[success++]=0;
                break;
              }
            } /* if dowrite else */
          } /* if keyfound */
        } /* if foundsect */
      } /* if linelen */
    } /* while (offset < filelen) */

        
    if (dowrite) 
    {
      if (!foundsect) /* no such section */
      {
        if (key == NULL) /* delete section */
          success = 1;
        else if (value == NULL) /* delete key+value */
          success = 0;
        else /* value != NULL */
        {
          /*can't use isspace(data[filelen-1])*/
          while (filelen > 0 && (data[filelen-1]=='\r' || data[filelen-1]=='\n'
                 || data[filelen-1]==' ' || data[filelen-1]=='\t'))
            filelen--;
          if (filelen > 0)
          {
            data[filelen++]='\n';  
            data[filelen++]='\n';
          }
          if (sect)
          {
            data[filelen++]='[';
            for (i=0;sect[i];i++)
              data[filelen++]=sect[i];
            data[filelen++]=']';
            data[filelen++]='\n';
          }
          for (i=0;key[i];i++)
            data[filelen++]=key[i];
          data[filelen++]='=';
          if (quotechar)
            data[filelen++]=quotechar;
          for (i=0;value[i];i++)
            data[filelen++]=value[i];
          if (quotechar)
            data[filelen++]=quotechar;
          data[filelen++]='\n';
          changed=1;
        }
      }
      else /* found section */
      {
        if (value == NULL && foundrecs == 0) /* no more records, */
          key = NULL;                        /* so delete section too */
              
        if (key == NULL)          /* delete section */
        {
          /*can't use isspace(data[sectoff-1])*/
          while (sectoff > 0 && (data[sectoff-1]=='\r' || data[sectoff-1]=='\n'
                 || data[sectoff-1]==' ' || data[sectoff-1]=='\t'))
             sectoff--;
          if (sectoff > 0)
          {
            data[sectoff++]='\n';  
            data[sectoff++]='\n';  
          }
          if (sectoffend == 0) /* no section after found section */
            filelen = sectoff; /* new filelen */
          else
          {
            memmove( (void *)&data[sectoff], 
                     (void *)&data[sectoffend],
                     filelen-sectoffend );
            filelen-=(sectoffend-sectoff);
          }
          changed = 1;
        }
        else if (value == NULL) /* found section, but didn't find key */
        {
          success = 1;
        }
        else if (!changed) /* no old key in section */
        {
          long i;
          if (sectoffend == 0)
            sectoffend = filelen;
          /*can't use isspace(data[sectoffend-1])*/
          while (sectoffend > 0 && (data[sectoffend-1]=='\r' || 
                 data[sectoffend-1]=='\n' || data[sectoffend-1]==' ' || 
                 data[sectoffend-1]=='\t'))
            sectoffend--;
          i = strlen(key)+strlen(value)+1+((quotechar)?(2):(0))+2;
          memmove( (void *)&data[sectoffend+i], (void *)&data[sectoffend],
                   filelen-sectoffend );
          filelen+=i;
          data[sectoffend++]='\n';
          for (i=0;key[i];i++)
            data[sectoffend++]=key[i];
          data[sectoffend++]='=';
          if (quotechar)
            data[sectoffend++]=quotechar;
          for (i=0;value[i];i++)
            data[sectoffend++]=value[i];
          if (quotechar)
            data[sectoffend++]=quotechar;
          data[sectoffend++]='\n';
          data[sectoffend++]='\n';
          changed = 1;
        }
      }
      if (changed)
      {
        FILE *file = fopen(filename,"r+");

        /* strip trailing triviality */
        while (filelen > 0 && (data[filelen-1]=='\r' ||
             data[filelen-1]=='\n' || data[filelen-1]==' ' ||
             data[filelen-1]=='\t' || iscntrl(data[filelen-1])))
          filelen--;
        if (filelen > 0)
          data[filelen++]='\n';

        i = -1L; 
        if (!file)
        {
          if (access(filename,0)!=0) /* only if file doesn't exist */
          {
            file = fopen(filename,"w");
            if (file)
              i = 0;
          }
        }
        else if (fseek(file,0,SEEK_END)==0)
        {
          i = ftell(file);
          if (i != -1L)
          {
            char nnn[64];
            memset(nnn, ' ', sizeof(nnn));
            nnn[0] = '\n';

            if (i != -1L && filelen >= i)
            {
              /* if new size is greater than old, gently extend the file */
              while (i != -1L && filelen > i)
              {
                unsigned int s = (filelen - i);
                if (s > sizeof(nnn))
                  s = sizeof(nnn);
                if (fwrite( (void *)&nnn[0], sizeof(char), s, file )!=s)
                  i = -1L;
                else if (fflush(file) != 0)
                  i = -1L;
                else if (nnn[0] == '\n')
                  i += s;
                else if (fseek(file,0,SEEK_END)!=0)
                  i = -1L;
                else
                  i = ftell(file);
                nnn[0] = ' ';
              }
              if (i != -1L)
              { 
                if (fseek(file,0,SEEK_SET)!=0)
                  i = -1L;
              }
            }
            else /* new size is less than old */
            {
              /* we should use ftruncate here, but thats not 
                 supported by MacOS, AmigaOS, VMS and perhaps others.
              */
              if (fwrite( (void *)&nnn[0], sizeof(char), 
                           sizeof(nnn), file )!=sizeof(nnn))
                i = -1L;
              else if (fflush(file) != 0)
                i = -1L;
              else if (fseek(file,0,SEEK_END)!=0)
                i = -1L;
              else
              {
                i = ftell(file);
                if (i != -1L)
                {
                  /* if the size difference was small, then the padding
                     maybe all we need. The chances are good that thats 
                     good enough to avoid the fopen(,"w") since .ini's 
                     rarely shrink by more than a few bytes.
                  */
                  if (filelen >= i)
                  {
                    if (fseek(file,0,SEEK_SET)!=0)
                      i = -1L;
                  }
                  else
                  {
                    fclose(file);
                    file = fopen( filename, "w" );
                  }
                }
              }
            }
          }
        }
        if (file)
        {
          if (i != -1L)
          {
            if (fwrite( (void *)data, sizeof(char), filelen, file )!=0)
              success = 1;
          }
          fclose(file);
        }

      } /* if changed */
    } /* if dowrite */
    
    free(data);
  } /* if data */

  
  if (!dowrite) 
  {
    if (success != 0)
      success--;
    else if (buffer && bufflen)
    {
      if (key == NULL) /* get all key+value pairs for sect */
      {
        buffer[0] = 0; 
        if (bufflen > 1) /* getsect is terminated by '\0\0' */
          buffer[1] = 0;
      }
      else if (value == NULL || bufflen < 2)
      {   /* no default or buffer is not long enough */
        buffer[0] = 0;
      }
      else
      {
        strncpy( buffer, value, bufflen);
        buffer[bufflen-1]=0;
        success = strlen(buffer);
      }
    }
  }
  
  return success;
}  

/* ------------------------------------------------------------------- */

unsigned long GetPrivateProfileStringB( const char *sect, const char *key, 
                      const char *defval, char *buffer, 
                      unsigned long buffsize, const char *filename )
{
  return ini_doit( 0, sect, key, defval, buffer, buffsize, filename );
}

int WritePrivateProfileStringB( const char *sect, const char *key, 
                        const char *value, const char *filename )
{
  char buf[2];
  return (int)ini_doit( 1, sect, key, value, buf, 0, filename );
}


unsigned int GetPrivateProfileIntB( const char *sect, const char *key, 
                          int defvalue, const char *filename )
{
  char buf[(sizeof(long)+1)*3];
  int n;  unsigned long i;
  i = GetPrivateProfileStringB( sect, key, "", buf, sizeof(buf), filename);
  if (i==0)
    return defvalue;
  if ((n = atoi( buf ))!=0)
    return n;
  if (i<2 || i>4)
    return 0;
  for (n=0;(i>((unsigned long)(n))) && n<4;n++)
    buf[n]=(char)tolower(buf[n]);
  if (i==2 && buf[0]=='o' && buf[1]=='n')
    return 1;
  if (i==3 && buf[0]=='y' && buf[1]=='e' && buf[2]=='s')
    return 1;
  if (i==4 && buf[0]=='t' && buf[1]=='r' && buf[2]=='u' && buf[3]=='e')
    return 1;
  return 0;
}

int WritePrivateProfileIntB( const char *sect, const char *key, 
                            int value, const char *filename )
{
  char buffer[(sizeof(long)+1)*3]; sprintf(buffer, "%lu", ((long)(value)) );
  return WritePrivateProfileStringB( sect, key, buffer, filename );
}

/* ------------------------------------------------------------------- */

