/*
 * .ini (configuration file ala windows) file read/write routines
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>, no copyright.
 *
 * Unlike native windows' this version does not cache the file
 * separately, but uses buffered I/O and assumes the OS knows what 
 * file caching is. It does pass all windows compatibility tests I
 * could come up with (see table below).
 *
 * Unlike a cpp file of a similar name, this .ini file parser is pure
 * C, is portable, passes -pedantic tests, does not 'new' a bazillion 
 * and one times, will not die if mem is scarce, does not fragment 
 * memory, does not attempt to parse comments, and does not assume 
 * that the calling function was written by a twit. 
 *
 * But,.. it ain't perfect (yet) and doesn't pretend to be.
 *
*/

const char *iniread_cpp(void) {
return "@(#)$Id: iniread.cpp,v 1.27.2.2 1999/11/08 02:48:37 cyp Exp $"; }

#include <stdio.h>   /* fopen()/fclose()/fread()/fwrite()/NULL */
#include <string.h>  /* strlen()/memmove() */
#include <ctype.h>   /* tolower(). do not use isxxx() functions! */
#include <stdlib.h>  /* malloc()/free()/atoi() */
#include <limits.h>  /* UINT_MAX */
#include "iniread.h"

//#define ALLOW_EMBEDDED_COMMENTS /* embedded comment handling is not api conform */

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
*/                            

static unsigned long ini_doit( int dowrite, const char *sect, 
                               const char *key, const char *value,
                               char *buffer, unsigned long bufflen, 
                               const char *filename )
{
  char *data = NULL;
  long i,n,filelen = 0;
  unsigned long success = 0;
  char *quotechar = "";

  if (dowrite && !sect && !key && !value ) /* flush */
  {
    success = 1;
  }
  else if (filename && (dowrite || (buffer && bufflen>1)))
  {
    FILE *file = fopen( filename, ((dowrite)?("r+"):("r")) );
    if (!file && dowrite)
    {
      if (!key || !value) /* delete section || key+value */
        success = 1;
      else 
        file = fopen( filename, "w+" );
    }
    data = NULL;

    if (file)
    {
      if (dowrite && key && value && value[0]!='\'' && value[0]!='\"')
      {
        int qn=0;
        for (i=0;*quotechar==0 && value[i];i++)
        {
          char c=value[i];
          if (c=='\"' || c=='\'')
            quotechar = (char *)((c=='\"')?("\'"):("\""));
          else if (c == ' ' || c=='\t')
            qn = 1;
          #ifdef ALLOW_EMBEDDED_COMMENTS
          else if (c==';' || c=='#')
            qn = 1;
          #endif
        }
        if (qn && !*quotechar)
          quotechar = "\"";
      }

      if ( fseek( file, 0, SEEK_END ) == 0 )
      {
        filelen = ftell( file );
//printf("filelen: %ld\n", filelen );
//getchar();
        if (filelen == 0)
        {
          if (dowrite)
          {
            if (!key || !value) /* delete section || key+value */
              success = 1;
            else
            {
              int ok = 1;
              if ( sect != NULL)
              {
                ok = (fputc('[', file ) != EOF);
                for (n=0;ok && sect[n];n++)
                  ok = (fputc(sect[n], file ) != EOF);
                if (ok && (fputc(']', file ) != EOF))
                  ok = (fputc('\n', file ) != EOF);
              }
              for (n=0;ok && key[n];n++)
                ok = (fputc(key[n], file ) != EOF);
              if (ok)
                ok = (fputc('=', file ) != EOF);
              if (ok && *quotechar)
                ok = (fputc(*quotechar, file ) != EOF);
              for (n=0;ok && value[n];n++)
                ok = (fputc(value[n], file ) != EOF);
              if (ok && *quotechar)
                ok = (fputc(*quotechar, file ) != EOF);
              if (ok)
                ok = (fputc('\n', file ) != EOF);
              if (!ok)
              {
                fclose(file);
                file=fopen(filename,"w+");
              }
              success=ok;
            }
          }
        }  
        else if (filelen > 0)
        {
          if (fseek( file, 0, SEEK_SET ) == 0)
          {
            long malloclen = filelen + 16;
            if (dowrite && key && value)
              malloclen += (((sect)?(strlen(sect)):(0))+strlen(key)+strlen(value)+3);
//printf("malloclen: %ld, max: %ld\n", malloclen, (((long)(UINT_MAX))-128) );
            if (((unsigned long)malloclen) < ((unsigned long)(UINT_MAX-128)))
            {
              data = (char *)malloc((int)malloclen);
//printf("havedata 1: %p\n", data );
              if (data)
              {
                memset(data,'\n',malloclen);
                filelen = (long)fread( (void *)data, sizeof(char), filelen, file );
//printf("fillen 2: %ld\n", filelen );
                if (filelen == 0)
                {
                  free(data);
                  data=NULL;
                }
              }
            }
          }
        }
      }
      if (file)
        fclose(file);
      file = NULL;
    }
  }

//printf("havedata: %p\n", data );
      
  if (data)
  {
    long offset = 0, sectoff = 0, sectoffend = 0, foundrecs = 0;
    int anysect = 0, changed = 0, foundsect = 0;

    if (sect == NULL && filelen)
      foundsect = 1;      

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
//printf("line: \"%70.70s\"\n", &data[keyoff] );

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
                foundrecs--;     /* the record is gone */
              else               /* insert key+value */
              {
                valuelen = 0;
                while (value[valuelen])
                  valuelen++;
                linelen = keylen+1+valuelen+((*quotechar)?(2):(0))+1;
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
                if (*quotechar)
                  data[n++]=*quotechar;
                for (i=0;i<valuelen;i++)
                  data[n++]=value[i];
                if (*quotechar)
                  data[n++]=*quotechar;
                data[n++]='\n';
              }
              changed = 1;
            }
            else //if (dowrite) else
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
          } //if keyfound */
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
          if (*quotechar)
            data[filelen++]=*quotechar;
          for (i=0;value[i];i++)
            data[filelen++]=value[i];
          if (*quotechar)
            data[filelen++]=*quotechar;
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
          i = strlen(key)+strlen(value)+1+((*quotechar)?(2):(0))+2;
          memmove( (void *)&data[sectoffend+i], (void *)&data[sectoffend],
                   filelen-sectoffend );
          filelen+=i;
          data[sectoffend++]='\n';
          for (i=0;key[i];i++)
            data[sectoffend++]=key[i];
          data[sectoffend++]='=';
          if (*quotechar)
            data[sectoffend++]=*quotechar;
          for (i=0;value[i];i++)
            data[sectoffend++]=value[i];
          if (*quotechar)
            data[sectoffend++]=*quotechar;
          data[sectoffend++]='\n';
          data[sectoffend++]='\n';
          changed = 1;
        }
      }
      if (changed)
      {
        FILE *file = fopen(filename,"w" );
        if (file)
        {
          long i = filelen;
          while (filelen > 0 && (data[filelen-1]=='\r' || 
                 data[filelen-1]=='\n' || data[filelen-1]==' ' || 
                 data[filelen-1]=='\t'))
            filelen--;
          if (i != filelen)
            data[filelen++]='\n';
          if (fwrite( (void *)data, sizeof(char), filelen, file )!=0)
            success = 1;
          if (i == filelen)
            fputc( '\n', file );
          fclose(file);
        }
      }
    } /* if dowrite */
    
    free(data);
  } //if data

  
  if (!dowrite) 
  {
    if (success != 0)
      success--;
    else if (buffer && bufflen)
    {
      buffer[0]=0;
      if (key == NULL) /* get all key+value pairs for sect */
      {
        if (bufflen>1)
          buffer[1]=0;
      }
      else if (value && bufflen>1)
      {
        strncpy( buffer, value, bufflen-1);
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
//  printf("GetStr: [%s]%s=%s\n", sect, key, defval);
//  getchar();
  unsigned long x=ini_doit( 0, sect, key, defval, buffer, buffsize, filename );
//printf("GetStr2: [%s]%s=%s\n", sect, key, buffer );
  return x;
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
  int n;
  unsigned long i;
  i = GetPrivateProfileStringB( sect, key, "", buf, sizeof(buf), filename);
  if (i==0)
    return defvalue;
  if ((n = atoi( buf ))!=0)
    return n;
  if (i<2)
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

