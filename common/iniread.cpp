/*
  INI file reading/processing class for C++

  $Log: iniread.cpp,v $
  Revision 1.14  1999/01/26 20:17:34  cyp
  new ini stuff from proxy

  Revision 1.10  1999/01/22 09:26:38  jlawson
  quoted values that have their closing quote in the middle with
  trailing garbage before the next separator/eol, will now have
  the garbage discarded.

  Revision 1.9  1999/01/22 08:56:30  jlawson
  added parsing for double-quotes around keyname.  fixed parsing of
  null values in equating lines.  fixed parsing of double-quoted
  values in equating lines.

  Revision 1.8  1998/12/27 22:17:50  jlawson
  fixed numerous code style and syntax weaknesses caught by lint checker.

  Revision 1.7  1998/12/25 02:04:38  jlawson
  changed usage of ltoa to only Win32

  Revision 1.6  1998/12/24 04:53:15  jlawson
  added handling for HAVE_SNPRINTF.  GetProfileString() functions renamed
  to GetProfileStringA() since Windows headers sometimes define them such.

  Revision 1.5  1998/09/06 20:08:45  jlawson
  corrected numerous compilation warnings under gcc

  Revision 1.4  1998/08/22 08:41:21  jlawson
  added new iniread code

  Revision 1.10  1998/07/07 21:55:41  cyruspatel
  client.h has been split into client.h and baseincs.h 
  
  Revision 1.9  1998/06/29 08:44:11  jlawson
  More OS_WIN32S/OS_WIN16 differences and long constants added.

  Revision 1.8  1998/06/29 06:58:02  jlawson
  added new platform OS_WIN32S to make code handling easier.

  Revision 1.7  1998/06/15 12:03:59  kbracey
  Lots of consts.

  Revision 1.6  1998/06/14 08:26:49  friedbait
  'Id' tags added in order to support 'ident' command to display a bill of
  material of the binary executable

  Revision 1.5  1998/06/14 08:12:53  friedbait
  'Log' keywords added to maintain automatic change history

*/

#if (!defined(lint) && defined(__showids__))
const char *iniread_cpp(void) {
return "@(#)$Id: iniread.cpp,v 1.14 1999/01/26 20:17:34 cyp Exp $"; }
#endif

#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>

#define IniNULL IniString()

#if defined(__TURBOC__)
#pragma warn -inl
#endif

/////////////////////////////////////////////////////////////////////////////
class IniString
{
  char *buffer;
public:
  // constructors and destructors
  inline IniString() : buffer(NULL) {};
  inline IniString(const IniString &that) : buffer(NULL) {*this = that;}
  inline IniString(const void *value) : buffer(NULL) {*this = (char *)value;}
  inline IniString(const char *value) : buffer(NULL) {*this = value;}
  inline IniString(char value) : buffer(NULL) {*this = value;}
  inline IniString(long value) : buffer(NULL) {*this = value;}
  inline IniString(int value) : buffer(NULL) {*this = (long) value;}
  inline ~IniString() {if (buffer) delete buffer;}

  // assignment
  IniString &operator= (const char *value);
  IniString &operator= (const IniString &that);
  IniString &operator= (char value);
  IniString &operator= (long value);
  IniString &operator= (int value);
    
  // output conversions
  inline const char *c_str(void) const
    { return (buffer ? buffer : ""); }
  inline operator const char *(void) const
    { return (buffer ? buffer : ""); }
  inline operator int (void) const
    { return (int) atol(c_str()); }
  inline operator long (void) const
    { return (long) atol(c_str()); }
  inline void copyto(char *target, int maxlen) const
    {
      strncpy(target, c_str(), maxlen);
      target[maxlen - 1] = 0;
    }


  // conditional tests
  inline int is_null(void) const
    { return (!buffer || !buffer[0]); }
  inline friend int operator== (const IniString &s1, const char *s2)
    { 
    const char *p = s1.c_str(); if (p==s2)  return 1; if (!p || !s2) return 0; 
    while (tolower(*p)==tolower(*s2)) {p++; s2++;}
    return (!*p && !*s2);
    }
  inline friend int operator== (const IniString &s1, const IniString &s2)
    { return (s1 == (s2.c_str())); }
  inline friend int operator!= (const IniString &s1, const IniString &s2)
    { return !(s1 == s2); }

  inline int need_quotes(void) const
    {return (buffer && (strchr(buffer, ' ') || strchr(buffer, ',')) &&
          *buffer != '"' && strlen(buffer) > 1 &&
          buffer[strlen(buffer) - 1] != '"' );}

  // appending and prepending
  friend IniString operator+ (const IniString &s1, const IniString &s2);  
  inline IniString &operator+= (const IniString &s2)
    {*this = *this + s2; return *this;}
  inline IniString &append (const IniString &s2)
    {return(*this += s2);}
  IniString &append (char ch);
  inline IniString &prepend (const IniString &s2)
    {return(*this = s2 + *this);}

  // length
  inline int length(void) const
    {return strlen(c_str());}
  inline char operator[] (int index)
    {return c_str()[index];}

  // substring extraction
  IniString left(int newlength) const;
  IniString right(int newlength) const;
  IniString mid(int offset, int newlength = -1) const;
  int instr(int offset, const IniString &match) const;
  inline int instr(const IniString &match) const
    { return instr(0, match); }

  // other manipulations
  IniString ucase(void) const;
  IniString lcase(void) const;
  inline void clear(void)
    { if (buffer) buffer[0] = 0; }
};
/////////////////////////////////////////////////////////////////////////////
template <class T> class IniList
{
  T **pointers;
  int count;
  int maxcount;
protected:
  void EnsureSpace(int additional = 1)
    {
      if (count + additional > maxcount)
        {
          T **oldptr = pointers;
          maxcount = count + additional + 10;
          pointers = new T* [maxcount];
          memmove(pointers, oldptr, sizeof(T*) * count);
          delete [] oldptr;
        }
    }
public:
  IniList() : count(0), maxcount(10)
    {
      pointers = new T*[maxcount];
    };
  IniList(const IniList &other) : 
    count(other.count), maxcount(other.maxcount)
    {
      pointers = new T*[maxcount];
      for (int i = 0; i < count; i++)
        pointers[i] = new T(*other.pointers[i]);
    }
  ~IniList()
    {
      for (int i = 0; i < count; i++)
        delete pointers[i];
      delete [] pointers;
    }
  T* AddNew(const T &that)
    {
      EnsureSpace(1);
      return pointers[count++] = new T(that);  
    }
  T* AddNew()
    {
      EnsureSpace(1);
      return pointers[count++] = (T*) (new T);
    }
  T* AddFrom(T *that)
    {
      EnsureSpace(1);
      return pointers[count++] = that;
    }
  inline T &operator[] (int index)
    {
      return *pointers[index];
    }
  inline int GetCount() const
    {
      return count;
    }
  inline IniList<T> &operator= (const IniList &other)
    {
      for (int i = 0; i < count; i++)
        delete pointers[i];
      delete [] pointers;
      
      count = other.count;
      maxcount = other.maxcount;
      pointers = new T*[maxcount];
      for (int j = 0; j < count; j++)
        pointers[j] = new T(*other.pointers[j]);

      return *this;
    }
  void Flush()
    {
      for (int i = 0; i < count; i++)
        delete pointers[i];
      count = 0;
    }
  void Detach(T *that)
    {
      for (int i = 0; i < count; i++)
        if (pointers[i] == that)
          {
            delete pointers[i];
            memmove(&pointers[i], &pointers[i+1], (count - i - 1) * sizeof(T*));
            count--;
            break;
          }
    }
};
/////////////////////////////////////////////////////////////////////////////
class IniStringList : public IniList<IniString>
{
public:
  // default constructor
  IniStringList() {};

  // string constructors
  inline IniStringList(const char *s1) { AddFrom(new IniString(s1)); }
  inline IniStringList(const IniString &v1) { AddNew(v1); }
  inline IniStringList(const IniString &v1, const IniString &v2) { AddNew(v1); AddNew(v2); }

  // integer constructors
  inline IniStringList(long v1) { AddFrom(new IniString(v1)); }
  inline IniStringList(int v1) { AddFrom(new IniString(v1)); }

  // file writing
  void fwrite(FILE *out);
};
/////////////////////////////////////////////////////////////////////////////
class IniRecord
{
public:
  int flags;              // user-defined data
  IniString key;
  IniStringList values;

  // constructors
  inline IniRecord() : flags(0) {};
  inline IniRecord(const char *Key, const IniStringList &Values) :
    flags(0), key(Key), values(Values) {};
  inline IniRecord(const char *Key, const char *Value) :
    flags(0), key(Key), values(Value) {};
  inline IniRecord(const char *Key, long v1) :
    flags(0), key(Key), values(v1) {};
  inline IniRecord(const char *Key, int v1) :
    flags(0), key(Key), values(v1) {};

  // file writing
  void fwrite(FILE *out);
};
/////////////////////////////////////////////////////////////////////////////
class IniSection
{
  IniList<IniRecord> records;

  // search iteration
  IniString lastsearch;
  int lastindex;
public:
  IniString section;

  // constructor
  inline IniSection() {};
  inline IniSection(const char *Section) : section(Section) {};

  // record addition
  inline void addrecord(const char *Key, const IniStringList &Values)
    { records.AddFrom(new IniRecord(Key, Values)); }
  inline void addrecord(const char *Key, const char *Value)
    { records.AddFrom(new IniRecord(Key, Value)); }
  inline void addrecord(const char *Key, long Value)
    { records.AddFrom(new IniRecord(Key, Value)); }
  inline void addrecord(const char *Key, int Value)
    { records.AddFrom(new IniRecord(Key, Value)); }

  // record modification
  inline void setkey(const char *Key, const IniStringList &Values)
    { IniRecord *that = findfirst(Key);if (that) that->values = Values;
      else addrecord(Key, Values); }
  inline void setkey(const char *Key, const char *v1)
    { IniRecord *that = findfirst(Key);if (that) { that->values.Flush(); 
      that->values.AddFrom(new IniString(v1)); } else addrecord(Key, v1); }
  inline void setkey(const char *Key, long v1)
    { char tmp[sizeof(long)*3]; sprintf(tmp,"%ld",v1); setkey( Key, tmp ); }
  inline void setkey(const char *Key, int v1) { setkey(Key, (long) v1); }

  // record retrieval
  inline const IniStringList &getkey(const char *Key, const IniStringList &DefValue)
    { IniRecord *that = findfirst(Key);
      return (that ? that->values : DefValue);}
  inline long getkey(const char *Key, long DefValue)
    { IniRecord *that = findfirst(Key); return (that && 
      that->values.GetCount() > 0 ? (long) that->values[0] : DefValue ); }
  inline int getkey(const char *Key, int DefValue)
    { return (int) getkey(Key, (long) DefValue); }
  inline IniString getkey(const char *Key, const char *DefValue = 0)
    { IniRecord *that = findfirst(Key); return IniString( (that && 
      that->values.GetCount() > 0) ? that->values[0].c_str() : DefValue ); }

  // efficient alternate record retrieval (similar to Win32 api)
  inline int GetProfileInt(const char *Key, int DefValue)
    { return (int) getkey(Key, (long) DefValue); }
  inline void GetProfileStringA(const char *Key, const char *DefValue, char *buffer, int buffsize)
    { IniRecord *that = findfirst(Key);
      if (that && that->values.GetCount() > 0) that->values[0].copyto(buffer, buffsize);
      else { strncpy(buffer, DefValue, buffsize); buffer[buffsize - 1] = 0; } }
  inline const char *GetProfileStringA(const char *Key, const char *DefValue = NULL)
    { IniRecord *that = findfirst(Key);
      if (that && that->values.GetCount() > 0) return that->values[0].c_str();
      else return DefValue; }

  // record searching
  inline IniRecord *findfirst() { lastindex = 0; lastsearch.clear(); return findnext(); }
  inline IniRecord *findfirst(const char *Key) { lastindex = 0; lastsearch = Key; return findnext(); }
  inline IniRecord *findfirst(const IniString &Key) { lastindex = 0; lastsearch = Key; return findnext(); }
  IniRecord *findnext();

  // record deletion
  inline void deleterecord(IniRecord *record) { this->records.Detach(record); }
  inline void deleterecord(const char *Key)
    { IniRecord *record = findfirst(Key); if (record) this->records.Detach(record); }

  // file writing
  void fwrite(FILE *out);
};
/////////////////////////////////////////////////////////////////////////////
class IniFile
{
  IniList<IniSection> sections;
  IniString lastfilename;
public:
  IniFile() {};
  IniFile(const char *Filename) : lastfilename(Filename) {};

  // clearing
  void clear()
    { sections.Flush(); }

  // reading and writing
  int ReadIniFile(const char *Filename = NULL, const char *Section = 0);
  int WriteIniFile(const char *Filename = NULL);
  void fwrite(FILE *out);

  // alternate record retrieval (similar to Win32 api).
  // (it is much more efficient to use findsection and call those methods)
  inline int GetProfileInt(const char *Section, const char *Key, int DefValue)
    { IniSection *section = findsection(Section);
      if (section) return section->GetProfileInt(Key, DefValue);
      else return DefValue; }
  inline void GetProfileStringA(const char *Section, const char *Key, const char *DefValue, char *buffer, int buffsize)
    { IniSection *section = findsection(Section);
      if (section) section->GetProfileStringA(Key, DefValue, buffer, buffsize);
      else { strncpy(buffer, DefValue, buffsize); buffer[buffsize - 1] = 0; } }
  inline const char *GetProfileStringA(const char *Section, const char *Key, const char *DefValue = NULL)
    { IniSection *section = findsection(Section);
      if (section) return section->GetProfileStringA(Key, DefValue);
      else return DefValue; }

  // section matching
  IniSection *findsection(const char *Section);
  inline IniSection *addsection(const char *Section)
    { IniSection *that = findsection(Section);
     return (that ? that : sections.AddFrom(new IniSection(Section))); }
};
/////////////////////////////////////////////////////////////////////////////


/* ------------------------------------------------------------------- */

/////////////////////////////////////////////////////////////////////////////
IniString &IniString::operator= (const char *value)
{
  if (buffer) delete buffer;
  if (value) {
    buffer = new char[strlen(value) + 1];
    strcpy(buffer, value);
  } else buffer = NULL;
  return *this;
}
/////////////////////////////////////////////////////////////////////////////
IniString &IniString::operator= (const IniString &that)
{
  if (buffer) delete buffer;
  if (that.buffer) {
    buffer = new char[strlen(that.buffer) + 1];
    strcpy(buffer, that.buffer);
  } else buffer = NULL;
    return *this;
}
/////////////////////////////////////////////////////////////////////////////
IniString &IniString::operator= (long value)
{
  char temp[(sizeof(long)+1)*3];
  sprintf(temp,"%ld",((long)(value)));
  (*this) = temp;
  return *this;
}  
/////////////////////////////////////////////////////////////////////////////
IniString &IniString::operator= (int value)
{
  char temp[(sizeof(long)+1)*3];
  sprintf(temp,"%ld",((long)(value)));
  (*this) = temp;
  return *this;
}  
/////////////////////////////////////////////////////////////////////////////
IniString &IniString::operator= (char value)
{
  if (buffer) delete buffer;
  buffer = new char[2];
  buffer[0] = value;
  buffer[1] = 0;
  return *this;
}
/////////////////////////////////////////////////////////////////////////////
IniString operator+ (const IniString &s1, const IniString &s2)
{
  if (s1.is_null()) return s2;
  else if (s2.is_null()) return s1;
  else {
    IniString temp;
    temp.buffer = new char[strlen(s1.c_str()) + strlen(s2.c_str()) + 1];
    strcat(strcpy(temp.buffer, s1.c_str()), s2.c_str());
    return temp;
  }
}
/////////////////////////////////////////////////////////////////////////////
IniString &IniString::append(char ch)
{
  char *oldbuffer = buffer;
  int oldlen = (oldbuffer ? strlen(buffer) : 0);
  buffer = new char[oldlen + 2];
  if (oldlen) memmove(buffer, oldbuffer, oldlen);
  if (oldbuffer) delete oldbuffer;
  buffer[oldlen] = ch;
  buffer[oldlen + 1] = 0;
  return(*this);
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::left(int newlength) const
{
  IniString output;
  if (newlength < 0) newlength = 0;
  int thislen = length();
  if (newlength > thislen) newlength = thislen;
  output.buffer = new char[newlength + 1];
  strncpy(output.buffer, c_str(), newlength);
  output.buffer[newlength] = 0;
  return output;
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::right(int newlength) const
{
  int thislen = length();
  if (newlength > thislen) newlength = thislen;
  return IniString(c_str() + thislen - newlength);
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::mid(int offset, int newlength) const
{
  IniString output;
  int thislen = length();
  if (offset > thislen || offset < 0) return output;
  if (newlength < 0 || offset + newlength > thislen) 
    newlength = thislen - offset;
  output.buffer = new char[newlength + 1];
  strncpy(output.buffer, c_str() + offset, newlength);
  output.buffer[newlength] = 0;
  return output;
}
/////////////////////////////////////////////////////////////////////////////
int IniString::instr(int offset, const IniString &match) const
{
  int thislen = length();
  if (offset > thislen) return -1;
  char *ptr = strstr((char*)c_str() + offset, (char*)match.c_str());
  if (ptr == NULL) return -1;
  return ptr - c_str();
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::ucase(void) const
{
  IniString output = *this;
  for (char *p = (char*) c_str(); *p; p++) 
    *p = (char) toupper(*p);
  return output;
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::lcase(void) const
{
  IniString output = *this;
  for (char *p = (char*) c_str(); *p; p++) 
    *p = (char) tolower(*p);
  return output;
}
/////////////////////////////////////////////////////////////////////////////
void IniStringList::fwrite(FILE *out)
{
  for (int i = 0; i < GetCount(); i++)
  {
    if (i) fprintf(out, ",");
    
    if ((*this)[i].need_quotes())
      fprintf(out, "\"%s\"", (*this)[i].c_str());
    else
      fprintf(out, "%s", (*this)[i].c_str());
  }
}
/////////////////////////////////////////////////////////////////////////////
void IniRecord::fwrite(FILE *out)
{
  if (key.is_null())
  {
    // this is a comment
    fprintf(out, ";%s\n", values[0].c_str());
  }
  else
  {
    if (key.need_quotes())
      fprintf(out, "\"%s\"=", key.c_str());
    else
      fprintf(out, "%s=", key.c_str());

    values.fwrite(out);
    fprintf(out, "\n");
  }
}
/////////////////////////////////////////////////////////////////////////////
void IniSection::fwrite(FILE *out)
{  
  fprintf(out, "[%s]\n", section.c_str());
  for (int i = 0; i < records.GetCount(); i++)
    records[i].fwrite(out);
}
/////////////////////////////////////////////////////////////////////////////
void IniFile::fwrite(FILE *out)
{
  for (int i = 0; i < sections.GetCount(); i++)
  {
    sections[i].fwrite(out);
    fprintf(out, "\n");
  }
}
/////////////////////////////////////////////////////////////////////////////
IniSection *IniFile::findsection(const char *Section)
{
  for (int i = 0; i < sections.GetCount(); i++)
    if (sections[i].section == Section) return &sections[i];
  return 0;
}
/////////////////////////////////////////////////////////////////////////////
IniRecord *IniSection::findnext()
{
  for (; lastindex < records.GetCount(); lastindex++)
  {
    if (lastsearch.is_null() || records[lastindex].key == lastsearch)
      return &records[lastindex++];
  }
  return NULL;
}
/////////////////////////////////////////////////////////////////////////////
// returns false on error
int IniFile::ReadIniFile(const char *Filename, const char *Section)
{
  // open up the file
  if (Filename) lastfilename = Filename;
  FILE *inf = fopen(lastfilename.c_str(), "r");
  if (inf == NULL) return false;             // open failed

  // start reading the file
  IniSection *section = 0;
  while (!feof(inf))
  {
    // eat leading whitespace
    int peekch = fgetc(inf);
    while (peekch != EOF && isspace(peekch)) 
      peekch = fgetc(inf);

    if (peekch == '[')
    {
      // []-------------------------[]
      // |  Handle section headings  |
      // []-------------------------[]
      IniString sectname;

      while ((peekch = fgetc(inf)) != EOF &&
        peekch != '\n' && peekch != ']')
          if (isprint(peekch)) sectname.append((char)peekch);

      if (peekch == ']' && (!Section || sectname == Section))
        section = addsection(sectname);
      else
        section = 0;

      // absorb to EOL or EOF
      while (peekch != EOF && peekch != '\n') peekch = fgetc(inf);

    }
    else if (peekch == '#' || peekch == ';')
    {
      // []----------------------[]
      // |  Handle comment lines  |
      // []----------------------[]
      IniString comment;
      while ((peekch = fgetc(inf)) != EOF && peekch != '\n')
      {
        if (isprint(peekch)) comment.append((char)peekch);
      }
      if (section) section->addrecord(NULL, IniStringList(comment));
    }
    else if (!section)
    {
      // []------------------------------------[]
      // |  ignore line (not in a section yet)  |
      // []------------------------------------[]
      while ((peekch = fgetc(inf)) != EOF && peekch != '\n') {};
    }
    else
    {
      // []-----------------------[]
      // |  Handle equating lines  |
      // []-----------------------[]

      // separate out the key name
      IniString key;
      if (peekch != EOF && peekch != '\n' && peekch != '=' &&
        isprint(peekch)) key.append((char)peekch);
      while ((peekch = fgetc(inf)) != EOF &&      
        peekch != '\n' && peekch != '=' && peekch != ';')
      {
        if (isprint(peekch)) key.append((char)peekch);
      }

      // chop trailing space and quote pairs.
      char *p = strchr((char*)key.c_str(), 0) - 1;
      if (*key.c_str() == '"' && p > key.c_str() && *p == '"')
      {
        *p = 0;
        strcpy((char*) key.c_str(), key.c_str() + 1);
        p = strchr((char*)key.c_str(), 0) - 1;
      }        
      while (isspace(*p) && p >= key.c_str()) *p-- = 0;

      // separate out all of the values
      if (peekch == '=')
      {
        IniStringList args;

        // strip out one argument
        while (peekch != EOF)
        {
          IniString value;

          // absorb leading white space
          while ((peekch = fgetc(inf)) != EOF &&
              peekch != '\n' && isspace(peekch)) {};
          if (peekch == EOF || peekch == '\n') 
            break;

          if (peekch == '"')
          {
            // quoted argument
            while ((peekch = fgetc(inf)) != EOF && peekch != '\n' && peekch != '"')
              if (isprint(peekch)) value.append((char)peekch);
            if (peekch == '"')
            {
              peekch = fgetc(inf);

              // skip past garbage to the next value.
              while (peekch != EOF && peekch != '\n' && peekch != ',')
                peekch = fgetc(inf);
            }
          }
          else
          {
            // unquoted argument
            while (peekch != '\n' &&
              peekch != ',' && peekch != ';' && peekch != '#')
            {
              if (isprint(peekch)) value.append((char)peekch);
              peekch = fgetc(inf);
              if (peekch == EOF) break;
            }

            // strip trailing whitespace
            char *p = strchr((char*)value.c_str(), 0) - 1;
            while (isspace(*p) && p >= value.c_str()) *p-- = 0;
          }


          // add it to our chain
          args.AddNew(value);

          // if we stopped because of a comment, then ignore to EOL
          if (peekch == ';' || peekch == '#')
            while ((peekch = fgetc(inf)) != EOF && peekch != '\n') {};

          // break if this was the end of the line
          if (peekch == '\n' || peekch == EOF) break;
        }

        // store this key/value pair
        section->addrecord(key, args);
      }
      else if (peekch != EOF && peekch != '\n')
      {
        // if we stopped because of a comment, then ignore to EOL
        while ((peekch = fgetc(inf)) != EOF && peekch != '\n') {};
      }
    }
  }
  fclose(inf);
  return 1; //true
}
/////////////////////////////////////////////////////////////////////////////
// returns false on error
int IniFile::WriteIniFile(const char *Filename)
{
  if (Filename) lastfilename = Filename;
  FILE *outf = fopen(lastfilename.c_str(), "w");
  if (outf == NULL) return 1;
  fwrite(outf);
  fclose(outf);
  return 1;
}
/////////////////////////////////////////////////////////////////////////////

unsigned long GetPrivateProfileStringA( const char *sect, const char *key, 
                      const char *defval, char *buffer, 
                      unsigned long buffsize, const char *filename )
{
  int foundentry;
  IniFile inifile;
  IniSection *inisect;
  IniRecord *inirec;

  if (sect == NULL)
    return 0;
  if (buffsize == 0 || buffer == NULL || filename == NULL)
    return 0;
  if (key == NULL)                 //we do not support section functions
    return 0;                      //ie return section if key is NULL
  if (defval == NULL)
    defval = "";
  buffer[0]=0;
  if (buffsize == 1)
    return 0;
  foundentry = 0;
  if ( inifile.ReadIniFile( filename ) == 0 )
    {
    if ((inisect = inifile.findsection( sect )) != NULL)
      {
      if ((inirec = inisect->findfirst( key )) != NULL)
        {
        buffer[0]=0;
        foundentry = 1;
        if (inirec->values.GetCount() > 0)
          strncpy( buffer, inirec->values[0].c_str(), buffsize-1 );
        }
      }
    }
  if (!foundentry && *defval && defval != buffer)
    strncpy( buffer, defval, buffsize-1 );
  buffer[buffsize-1]=0;
  return strlen(buffer);
}

int WritePrivateProfileStringA( const char *sect, const char *key, 
                        const char *value, const char *filename )
{
  IniFile inifile;
  IniSection *inisect;
  IniRecord *inirec;

  int changed =0;
  if (sect == NULL)
    return 0;
  if (key == NULL)                 //we do not support section functions
    return 0;                      //ie delete section if key is NULL
  inifile.ReadIniFile( filename );
  if ((inisect = inifile.findsection( sect )) == NULL)
    {
    if (value == NULL || key == NULL)
      return 1;
    if ((inisect = inifile.addsection( sect )) == NULL)
      return 0;
    }
  if (value == NULL)
    {
    if ((inirec = inisect->findfirst( key ))!=NULL)
      {
      inisect->deleterecord( inirec );
      changed = 1;
      }
    }
  else
    {
    inisect->setkey(key, value);
    changed = 1;
    }
  if (changed)
    {
    if ( inifile.WriteIniFile())
      return 0;
    }
  return 1; //success
}


unsigned int GetPrivateProfileIntA( const char *sect, const char *key, 
                          int defvalue, const char *filename )
{
  char buf[(sizeof(long)+1)*3];
  int n; unsigned long i;
  i=GetPrivateProfileStringA( sect, key, "", buf, sizeof(buf), filename);
  if (i==0)
    return defvalue;
  if ((n = atoi( buf ))!=0)
    return n;
  if (i<2 || i>4)
    return 0;
  for (n=0;n<4 && ((unsigned long)(n))<i;n++)
    buf[n]=(char)tolower(buf[n]);
  if ((i==2 && buf[0]=='o' && buf[1]=='n') ||
      (i==3 && buf[0]=='y' && buf[1]=='e' && buf[3]=='s') ||
      (i==4 && buf[0]=='t' && buf[1]=='r' && buf[3]=='u' && buf[4]=='e'))
    return 1;
  return 0;
}

int WritePrivateProfileIntA( const char *sect, const char *key, 
                            int value, const char *filename )
{
  char buf[(sizeof(long)+1)*3];
  sprintf(buf,"%ld",((long)(value)));
  return WritePrivateProfileStringA( sect, key, buf, filename );
}
