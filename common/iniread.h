// Hey, Emacs, this a -*-C++-*- file !
//
// INI file reading/processing class for C++
// 
// $Log: iniread.h,v $
// Revision 1.12  1998/06/29 06:58:04  jlawson
// added new platform OS_WIN32S to make code handling easier.
//
// Revision 1.11  1998/06/26 07:13:53  daa
// move strcmpi and strncmpi defs to cmpidefs.h
//
// Revision 1.10  1998/06/14 08:12:55  friedbait
// 'Log' keywords added to maintain automatic change history
//
// 

#ifndef __INIREAD_H__
#define __INIREAD_H__

#include "cputypes.h"

#if (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS)
extern "C" {
#endif

#include <stdio.h>
#include <ctype.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>

#if (CLIENT_OS == OS_RISCOS)
#include <sys/types.h>
#endif

#include <sys/stat.h>
#include "cmpidefs.h"

#if (CLIENT_OS == OS_AMIGAOS) || (CLIENT_OS == OS_RISCOS)
}
#endif

#define IniNULL IniString()

class ostream;

/////////////////////////////////////////////////////////////////////////////
class IniString
{
  char *buffer;
public:
  IniString() : buffer(NULL) {};
  IniString(const IniString &that) : buffer(NULL) {*this = that;}
  IniString(const void *value) : buffer(NULL) {*this = (char *)value;}
  IniString(const char *value) : buffer(NULL) {*this = value;}
  IniString(char value) : buffer(NULL) {*this = value;}
  IniString(s32 value);
  ~IniString() {if (buffer) delete buffer;}
  IniString &operator= (const char *value);
  IniString &operator= (const IniString &that);
  IniString &operator= (char value);
  IniString &assign(const char *value)
    {return (*this = value);}
  IniString &assign(const IniString &that)
    {return (*this = that);}
  const char *c_str(void) const
    {return (buffer ? buffer : "");}
  operator s32 (void) const;
  bool is_null(void) const
    {return (!buffer || !buffer[0] || strcmp(buffer, "0") == 0);}
  friend bool operator== (const IniString &s1, const IniString &s2);
  friend bool operator!= (const IniString &s1, const IniString &s2)
    {return !(s1 == s2);}
  friend ostream &operator<< (ostream &out, const IniString &that);
  bool need_quotes(void) const
    {return (buffer && (strchr(buffer, ' ') || strchr(buffer, ',')));}
  friend IniString operator+ (const IniString &s1, const IniString &s2);
  IniString &operator+= (const IniString &s2)
    {*this = *this + s2; return *this;}
  IniString &append (const IniString &s2)
    {return(*this += s2);}
  IniString &append (char ch);
  IniString &prepend (const IniString &s2)
    {return(*this = s2 + *this);}
  int length(void) const
    {return strlen(c_str());}
  char operator[] (int index)
    {return c_str()[index];}
  IniString left(int newlength) const;
  IniString right(int newlength) const;
  IniString mid(int offset, int newlength = -1) const;
  int instr(int offset, const IniString &match) const;
  int instr(const IniString &match) const
    {return instr(0, match);}
  IniString ucase(void) const;
  IniString lcase(void) const;
  void copyto(char *target, int maxlen) const;
};
/////////////////////////////////////////////////////////////////////////////
class IniStringList
{
  IniString *list;
  int count;
public:
  IniStringList() : list(NULL), count(0) {};
  IniStringList(const IniStringList &that) : list(NULL), count(0) {*this = that;}
  IniStringList(const IniString &v1)
    : list(NULL), count(0) {Add(v1);}
  IniStringList(const IniString &v1, const IniString &v2)
    : list(NULL), count(0) {Add(v1); Add(v2);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4); Add(v5);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4); Add(v5); Add(v6);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7) : list(NULL), count(0) {Add(v1); Add(v2);
    Add(v3); Add(v4); Add(v5); Add(v6); Add(v7);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6);
    Add(v7); Add(v8);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21,
    const IniString &v22 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21);
    Add(v22); }
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21,
    const IniString &v22, const IniString &v23 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21);
    Add(v22); Add(v23);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21,
    const IniString &v22, const IniString &v23, const IniString &v24 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21);
    Add(v22); Add(v23); Add(v24);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21,
    const IniString &v22, const IniString &v23, const IniString &v24,
    const IniString &v25 )
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21);
    Add(v22); Add(v23); Add(v24); Add(v25);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21,
    const IniString &v22, const IniString &v23, const IniString &v24,
    const IniString &v25, const IniString &v26)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21);
    Add(v22); Add(v23); Add(v24); Add(v25); Add(v26);}
  IniStringList(const IniString &v1, const IniString &v2, const IniString &v3,
    const IniString &v4, const IniString &v5, const IniString &v6,
    const IniString &v7, const IniString &v8, const IniString &v9,
    const IniString &v10, const IniString &v11, const IniString &v12,
    const IniString &v13, const IniString &v14, const IniString &v15,
    const IniString &v16, const IniString &v17, const IniString &v18,
    const IniString &v19, const IniString &v20, const IniString &v21,
    const IniString &v22, const IniString &v23, const IniString &v24,
    const IniString &v25, const IniString &v26, const IniString &v27)
    : list(NULL), count(0) {Add(v1); Add(v2); Add(v3); Add(v4);
    Add(v5); Add(v6); Add(v7); Add(v8); Add(v9); Add(v10); Add(v11);
    Add(v12); Add(v13); Add(v14); Add(v15);
    Add(v16); Add(v17); Add(v18); Add(v19); Add(v20); Add(v21);
    Add(v22); Add(v23); Add(v24); Add(v25); Add(v26); Add(v27);}


  ~IniStringList()
    {
      if (list) delete [] list;
      list = NULL;
      count = 0;
    }

  void Add(const IniString &value);
  void Erase(void)
    {
      if (list) delete [] list;
      list = NULL;
      count = -1;
    }
  IniString &operator[] (int index);
  IniStringList &operator= (const IniStringList &that);
  int length(void) const {return count;}
  friend ostream &operator<< (ostream &out, const IniStringList &that);
  void fwrite(FILE *out);
};
/////////////////////////////////////////////////////////////////////////////
class IniRecord
{
public:
  char flags;
  IniString key;
  IniStringList values;
  IniRecord *next;

  IniRecord() : flags(0), next(NULL) {};
  IniRecord(const IniString &Key, const IniStringList &Values, IniRecord *Next = NULL)
      {key = Key; values = Values; next = Next; flags = 0;}
  ~IniRecord() {if (next) delete next;}

  IniRecord *findfirst(const IniString &Key = IniNULL);
  IniRecord *findnext(const IniString &Key = IniNULL)
    {return (IniRecord*)(next ? next->findfirst(Key) : NULL);}
  friend ostream &operator<< (ostream &out, const IniRecord &that);
  void fwrite(FILE *out);
};
/////////////////////////////////////////////////////////////////////////////
class IniSection
{
  IniRecord *lastrecord;
public:
  IniString section;
  IniSection *next;
  IniRecord *record;

  IniSection() : lastrecord(NULL), next(NULL), record(NULL) {};
  ~IniSection()
    {
      if (record) delete record;
      if (next) delete next;
    }
  void clear(void)
    {
      if (record) {delete record; record = NULL;}
      if (next) {delete next; next = NULL;}
      section = IniNULL;
      lastrecord = NULL;
    }

  IniRecord *addrecord(const IniString &Section, const IniString &Key,
    const IniStringList &Values);
  IniRecord *setrecord(const IniString &Section, const IniString &Key,
    const IniStringList &Values);

  IniRecord *findfirst(const IniString &Section, const IniString &Key = IniNULL);
#if (CLIENT_CPU == CPU_ARM && CLIENT_OS != OS_RISCOS)
// get round a bug in gcc 2.7.2.2
  IniStringList &getkey(const IniString &Section, const IniString &Key,
      const IniStringList &DefValue = IniStringList(IniNULL), long AutoAdd = true);
  IniStringList &getkey(const IniString &Section, const IniString &Key,
      const IniString &DefValue, long AutoAdd = true)
      {return getkey(Section, Key, IniStringList(DefValue), AutoAdd);}
#else
  IniStringList &getkey(const IniString &Section, const IniString &Key,
      const IniStringList &DefValue = IniStringList(IniNULL), bool AutoAdd = true);
  IniStringList &getkey(const IniString &Section, const IniString &Key,
      const IniString &DefValue, bool AutoAdd = true)
      {return getkey(Section, Key, IniStringList(DefValue), AutoAdd);}
#endif
  friend ostream &operator<< (ostream &out, const IniSection &that);
  void fwrite(FILE *out);

  bool ReadIniFile(const char *filename, const IniString &Section = IniNULL, long offset = 0);
  bool UpdateIniFile(const char *Filename);
  bool WriteIniFile(const char *Filename);
};
/////////////////////////////////////////////////////////////////////////////

#endif

