// Hey, Emacs, this a -*-C++-*- file !
//
// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//
// ----------------------------------------------------------------------
// INI file reading/processing class for C++.
// Created by Jeff Lawson.
// ----------------------------------------------------------------------
//
// $Id: iniread.h,v 1.28 1999/04/05 13:51:32 jlawson Exp $

#ifndef __INIREAD_H__
#define __INIREAD_H__

#ifdef PROXYTYPE      // compiling proxy
  // proxy needs automatic tokenizing of ini values.
  #undef INIREAD_SINGLEVALUE

  // proxy uses the full class interface.
  #undef INIREAD_WIN32_LIKE

#else                 // compiling client
  // define this value to not split ini values by commas.
  #define INIREAD_SINGLEVALUE

  // define this value to only provide a public win32-like interface.
  #define INIREAD_WIN32_LIKE
#endif

/////////////////////////////////////////////////////////////////////////////

#if !defined(INIREAD_WIN32_LIKE) || defined(COMPILING_INIREAD)

#if defined(__TURBOC__)
#pragma warn -inl       // disable "cannot inline" warning.
#endif

extern "C" {
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include <stdlib.h>
}

// convenient macro for an empty IniString.
#define IniNULL IniString()

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
      const char *z1 = s1.c_str(); 
      if (!z1 || !s2) return (z1 == s2);
      while (*z1 && *s2 && tolower(*z1) == tolower(*s2)) { z1++; s2++; }
      return (*z1 == 0 && *s2 == 0);
    }
  inline friend int operator== (const IniString &s1, const IniString &s2)
    { return (s1 == s2.c_str()); }
  inline friend int operator!= (const IniString &s1, const IniString &s2)
    { return !(s1 == s2); }
  inline int need_quotes(void) const
    {return (buffer && (strchr(buffer, ' ') || strchr(buffer, ',')) &&
          *buffer != '\"' && strlen(buffer) > 1 &&
          buffer[strlen(buffer) - 1] != '\"' );}

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
  IniList(const IniList<T> &other) :
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
  inline IniList<T> &operator= (const IniList<T> &other)
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

#ifndef INIREAD_SINGLEVALUE
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
#endif

/////////////////////////////////////////////////////////////////////////////

class IniRecord
{
public:
  int flags;              // user-defined data
  IniString key;
#ifdef INIREAD_SINGLEVALUE
  IniString values;
#else
  IniStringList values;
#endif

  // constructors
  inline IniRecord() : flags(0) {};
#ifdef INIREAD_SINGLEVALUE
  inline IniRecord(const char *Key, const IniString &Values) :
    flags(0), key(Key), values(Values) {};
#else
  inline IniRecord(const char *Key, const IniStringList &Values) :
    flags(0), key(Key), values(Values) {};
#endif
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

  // record searching
  IniRecord *findnext();
  inline IniRecord *findfirst() 
  { lastindex = 0; lastsearch.clear(); return findnext(); }
  inline IniRecord *findfirst(const char *Key) 
  { lastindex = 0; lastsearch = Key; return findnext(); }
  inline IniRecord *findfirst(const IniString &Key) 
  { lastindex = 0; lastsearch = Key; return findnext(); }

  // record deletion
  inline void deleterecord(IniRecord *record) { records.Detach(record); }
  inline void deleterecord(const char *Key)
    { IniRecord *record = findfirst(Key); if (record) records.Detach(record); }

  // file writing
  void fwrite(FILE *out);

  // record addition
#ifdef INIREAD_SINGLEVALUE
  inline void addrecord(const char *Key, const IniString &Values)
    { records.AddFrom(new IniRecord(Key, Values)); }
#else
  inline void addrecord(const char *Key, const IniStringList &Values)
    { records.AddFrom(new IniRecord(Key, Values)); }
#endif
  inline void addrecord(const char *Key, const char *Value)
    { records.AddFrom(new IniRecord(Key, Value)); }
  inline void addrecord(const char *Key, long Value)
    { records.AddFrom(new IniRecord(Key, Value)); }
  inline void addrecord(const char *Key, int Value)
    { records.AddFrom(new IniRecord(Key, Value)); }


  // record modification
#ifdef INIREAD_SINGLEVALUE
  inline void setkey(const char *Key, const IniString &Values)
#else
  inline void setkey(const char *Key, const IniStringList &Values)
#endif
  { IniRecord *that = findfirst(Key); 
    if (that) that->values = Values;  else addrecord(Key, Values); }
  inline void setkey(const char *Key, const char *v1)
    { IniRecord *that = findfirst(Key); 
#ifdef INIREAD_SINGLEVALUE
    if (that) { that->values = v1; }
#else
    if (that) { that->values.Flush(); that->values.AddFrom(new IniString(v1));}
#endif
    else addrecord(Key, v1);}
  inline void setkey(const char *Key, long v1)
    { IniRecord *that = findfirst(Key);
#ifdef INIREAD_SINGLEVALUE
    if (that) { that->values = v1; }
#else
    if (that) { that->values.Flush(); that->values.AddFrom(new IniString(v1)); }
#endif
    else addrecord(Key, v1);}
  inline void setkey(const char *Key, int v1) { setkey(Key, (long) v1); }


  // record retrieval
#ifdef INIREAD_SINGLEVALUE
  inline const IniString &getkey(const char *Key, const IniString &DefValue)
#else
  inline const IniStringList &getkey(const char *Key, const IniStringList &DefValue)
#endif
    { IniRecord *that = findfirst(Key); return (that ? that->values : DefValue); }
  inline long getkey(const char *Key, long DefValue)
    { IniRecord *that = findfirst(Key);
#ifdef INIREAD_SINGLEVALUE
    return (that ? (long) that->values : DefValue );
#else
    return (that && that->values.GetCount() > 0 ? (long) that->values[0] : DefValue );
#endif
    }
  inline int getkey(const char *Key, int DefValue)
    { return (int) getkey(Key, (long) DefValue); }
  inline IniString getkey(const char *Key, const char *DefValue = 0)
    { IniRecord *that = findfirst(Key);
#ifdef INIREAD_SINGLEVALUE
    return IniString((that ? that->values.c_str() : DefValue ));
#else
   return IniString((that && that->values.GetCount() > 0 ? that->values[0].c_str() : DefValue ));
#endif
    }


  // efficient alternate record retrieval (similar to Win32 api)
  inline int GetProfileInt(const char *Key, int DefValue)
    { return (int) getkey(Key, (long) DefValue); }
  inline void GetProfileStringA(const char *Key, const char *DefValue, char *buffer, int buffsize)
    {
      IniRecord *that = findfirst(Key);
#ifdef INIREAD_SINGLEVALUE
      if (that) that->values.copyto(buffer, buffsize);
#else
      if (that && that->values.GetCount() > 0) that->values[0].copyto(buffer, buffsize);
#endif
      else { strncpy(buffer, DefValue, buffsize); buffer[buffsize - 1] = 0; }
    }
  inline const char *GetProfileStringA(const char *Key, const char *DefValue = NULL)
    {
      IniRecord *that = findfirst(Key);
#ifdef INIREAD_SINGLEVALUE
      if (that) return that->values.c_str();
#else
      if (that && that->values.GetCount() > 0) return that->values[0].c_str();
#endif
      else return DefValue;
    }

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

  
  // section matching
  IniSection *findsection(const char *Section)
   { for (int i = 0; i < sections.GetCount(); i++)
       { 
       //printf("comparing %s and %s == %d\n", sections[i].section.c_str(), 
       //                  Section, (sections[i].section == Section));
       if (sections[i].section == Section) return &sections[i]; }
    return 0;}
  inline IniSection *addsection(const char *Section)
    { IniSection *that = findsection(Section);
      return (that ? that : sections.AddFrom(new IniSection(Section))); }

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

};
#endif    // COMPILING_INIREAD

/////////////////////////////////////////////////////////////////////////////

#ifdef INIREAD_WIN32_LIKE

unsigned long GetPrivateProfileStringB( const char *sect, const char *key, 
                                    const char *defval, char *buffer, 
                                    unsigned long buffsize, 
                                    const char *filename );
int WritePrivateProfileStringB( const char *sect, const char *key, 
                                    const char *value, const char *filename );
unsigned int GetPrivateProfileIntB( const char *sect, const char *key, 
                                    int defvalue, const char *filename );
int WritePrivateProfileIntB( const char *sect, const char *key, 
                                     int value, const char *filename );

#endif /* ifdef INIREAD_WIN32_LIKE */


/////////////////////////////////////////////////////////////////////////////

#endif /* ifndef __INIREAD_H__ */
