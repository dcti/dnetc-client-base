/*
  INI file reading/processing class for C++

  
  $Log: iniread.cpp,v $
  Revision 1.6  1998/06/14 08:26:49  friedbait
  'Id' tags added in order to support 'ident' command to display a bill of
  material of the binary executable

  Revision 1.5  1998/06/14 08:12:53  friedbait
  'Log' keywords added to maintain automatic change history

  

  version 2.0 (May 25, 1997)
  by Jeff Lawson
  jlawson@hmc.edu     or    JeffLawson@aol.com
  http://members.aol.com/JeffLawson/


  changes in 1.1: (Mar 2, 1997)
    read function handles equating lines with nothing after equal sign.
    update will delete a key if the value array is NULL;
  changes in 1.2: (Apr 20, 1997)
    addrecord() now has an alternate protoype allowing a single string.
    files are closed before swapping in updateini.
    updateini now handles situation when ini file does not exist.
    updateini no longer copies multiple blank lines.
  changes in 2.0: (May 25, 1997)
    complete rewrite to not depend on the Borland string class.
*/

static char *id="@(#)$Id: iniread.cpp,v 1.6 1998/06/14 08:26:49 friedbait Exp $";

#include "iniread.h"


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
IniString::IniString(s32 value) : buffer(NULL)
{
  char temp[30];
  sprintf(temp, "%d", (int) value);
  assign(temp);
}
/////////////////////////////////////////////////////////////////////////////

IniString::operator s32 (void) const {return (s32) (buffer ? atoi(buffer) : 0);}

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
IniString &IniString::operator= (char value)
{
  if (buffer) delete buffer;
  buffer = new char[2];
  buffer[0] = value;
  buffer[1] = 0;
  return *this;
}
/////////////////////////////////////////////////////////////////////////////
#ifdef NO_STRCASECMP
int strcasecmp(const char *s1, const char *s2)
{
  while (*s1 && *s2) {
    if (toupper(*s1) != toupper(*s2)) {
      return (*s1 < *s2) ? -1 : 1;
    }
    s1++;
    s2++;
  }
  return *s2 ? -1 : (*s1 ? 1 : 0);
}
#endif
/////////////////////////////////////////////////////////////////////////////
bool operator== (const IniString &s1, const IniString &s2)
{
  if (s1.is_null() && s2.is_null()) return true;
  else if (s1.buffer && s2.buffer && strcmpi(s1.buffer, s2.buffer) == 0) return true;
  else return false;
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
  IniString output;
  int thislen = length();
  if (newlength > thislen) newlength = thislen;
  output.assign(c_str() + thislen - newlength);
  return output;
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::mid(int offset, int newlength) const
{
  IniString output;
  int thislen = length();
  if (offset > thislen || offset < 0) return output;
  if (newlength < 0 || offset + newlength > thislen) newlength = thislen - offset;
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
#ifdef __TURBOC__
  strupr((char*)output.c_str());
#else
  for (char *p = (char*) c_str(); *p; p++) *p = toupper(*p);
#endif
  return output;
}
/////////////////////////////////////////////////////////////////////////////
IniString IniString::lcase(void) const
{
  IniString output = *this;
#ifdef __TURBOC__
  strlwr((char*)output.c_str());
#else
  for (char *p = (char*) c_str(); *p; p++) *p = tolower(*p);
#endif
  return output;
}
/////////////////////////////////////////////////////////////////////////////
void IniString::copyto(char *target, int maxlen) const
{
  strncpy(target, c_str(), maxlen);
  target[maxlen - 1] = 0;
}
/////////////////////////////////////////////////////////////////////////////
void IniStringList::Add(const IniString &value)
{
  if (count < 0) count = 0;
  IniString *newlist = new IniString[count + 1];
  if (list) {
    for (int i = 0; i < count; i++) newlist[i] = list[i];
    delete [] list;
  }
  newlist[count++] = value;
  list = newlist;
}
/////////////////////////////////////////////////////////////////////////////
IniString &IniStringList::operator[] (int index)
{
  if (index >= count || index < 0) {
    // out of range, but return something to play with
    static IniString fake;
    fake = IniNULL;
    return fake;
  } else {
    return list[index];
  }
}
/////////////////////////////////////////////////////////////////////////////
IniStringList &IniStringList::operator= (const IniStringList &that)
{
  if (list) delete [] list;
  if ((count = that.count) != 0) {
    list = new IniString[count];
    for (int i = 0; i < count; i++) list[i] = that.list[i];
  } else list = NULL;
  return *this;
}
/////////////////////////////////////////////////////////////////////////////
void IniStringList::fwrite(FILE *out)
{
  for (int i = 0; i < length(); i++)
  {
    if (i) fprintf(out, ",");

    IniStringList &That = (IniStringList &)(*this);
    if (That[i].need_quotes())
      fprintf(out, "\"%s\"", That[i].c_str());
    else
      fprintf(out, "%s", That[i].c_str());
  }
}
/////////////////////////////////////////////////////////////////////////////
IniRecord *IniRecord::findfirst(const IniString &Key)
{
  if (Key.is_null()) return this;
  IniRecord *ptr = this;
  while (ptr) {
    if (ptr->key == Key) return ptr;
    ptr = ptr->next;
  }
  return NULL;
}
/////////////////////////////////////////////////////////////////////////////
void IniRecord::fwrite(FILE *out)
{
  if (values.length() < 0) return;

  if (key.need_quotes())
    fprintf(out, "\"%s\"=", key.c_str());
  else
    fprintf(out, "%s=", key.c_str());
  values.fwrite(out);
  fprintf(out, "\n");
}
/////////////////////////////////////////////////////////////////////////////
IniRecord *IniSection::addrecord(const IniString &Section, const IniString &Key,
  const IniStringList &Values)
{
  if (section == Section || (!record && section.is_null())) {
    section = Section;
    if (record && lastrecord) {
      lastrecord->next = new IniRecord(Key, Values);
      lastrecord = lastrecord->next;
    } else {
      lastrecord = record = new IniRecord(Key, Values);
    }
    return lastrecord;
  } else if (next == NULL) {
    next = new IniSection;
  }
  return next->addrecord(Section, Key, Values);
}
/////////////////////////////////////////////////////////////////////////////
IniRecord *IniSection::setrecord(const IniString &Section, const IniString &Key,
  const IniStringList &Values)
{
  if (section == Section || (!record && section.is_null())) {
    if (section != Section) section = Section;
    IniRecord *ptr = record->findfirst(Key);
    if (ptr) {
      ptr->values = Values;
      return ptr;
    } else {
      if (record && lastrecord) {
        lastrecord->next = new IniRecord(Key, Values);
        lastrecord = lastrecord->next;
      } else {
        lastrecord = record = new IniRecord(Key, Values);
      }
      return lastrecord;
    }
  } else if (next == NULL) {
    next = new IniSection;
  }
  return next->setrecord(Section, Key, Values);
}
/////////////////////////////////////////////////////////////////////////////
IniRecord *IniSection::findfirst(const IniString &Section, const IniString &Key)
{
  IniSection *ptr = this;
  while (ptr) {
    if (ptr->section == Section)
      if (ptr->record) return ptr->record->findfirst(Key);
      else return NULL;
    else if (Section.is_null() && ptr->record) {
      IniRecord *that = ptr->record->findfirst(Key);
      if (that) return that;
    }
    ptr = ptr->next;
  }
  return NULL;
}
/////////////////////////////////////////////////////////////////////////////
#if (CLIENT_CPU == CPU_ARM && CLIENT_OS != OS_RISCOS)
IniStringList &IniSection::getkey(const IniString &Section, const IniString &Key,
  const IniStringList &DefValue, long AutoAdd)
#else
IniStringList &IniSection::getkey(const IniString &Section, const IniString &Key,
  const IniStringList &DefValue, bool AutoAdd)
#endif
{
  IniRecord *r = findfirst(Section, Key);
  if (r) return r->values;
  else {
    if (AutoAdd) {
      return addrecord(Section, Key, DefValue)->values;
    } else {
      static IniStringList dummy;
      dummy = DefValue;
      return dummy;
    }
  }
}
/////////////////////////////////////////////////////////////////////////////
void IniSection::fwrite(FILE *out)
{
  if (!section.is_null() || record)
  fprintf(out, "[%s]\n", section.c_str());

  IniRecord *r = record;
  while (r)
  {
    r->fwrite(out);
    r = r->next;
  }
  fprintf(out, "\n");
  if (next) next->fwrite(out);
}
/////////////////////////////////////////////////////////////////////////////
// returns true on error
bool IniSection::ReadIniFile(const char *Filename, const IniString &Section, long offset)
{
  // open up the file
  FILE *inf = fopen(Filename, "rb");
  if (inf == NULL) return true;             // open failed
  if (offset < 0) fseek(inf, offset, 2);
  else if (offset > 0) fseek(inf, offset, 0);

  // start reading the file
  IniString sect;
  while (!feof(inf))
  {
    // eat leading whitespace
    int peekch = fgetc(inf);
    while (peekch != EOF && isspace(peekch)) peekch = fgetc(inf);

    if (peekch == '[')
    {
      // []-------------------------[]
      // |  Handle section headings  |
      // []-------------------------[]
      IniString h;

      while ((peekch = fgetc(inf)) != EOF &&
        peekch != '\n' && peekch != ']' && peekch != ';')
          if (isprint(peekch)) h.append((char)peekch);

      if (peekch == ']') sect = h;

      // absorb to EOL or EOF
      while (peekch != EOF && peekch != '\n') peekch = fgetc(inf);

    } else {
      // []-----------------------[]
      // |  Handle equating lines  |
      // []-----------------------[]

      // separate out the key name
      IniString key;
#if (CLIENT_OS != OS_NETWARE)
      ungetc(peekch, inf);
#else
      if (peekch != EOF && peekch != '\n' && peekch != '=' && peekch != ';')
      if (isprint(peekch)) key.append((char)peekch);
#endif
      while ((peekch = fgetc(inf)) != EOF &&
        peekch != '\n' && peekch != '=' && peekch != ';')
          if (isprint(peekch)) key.append((char)peekch);

#ifndef __WATCOMC__
      if (!Section.is_null() && sect != Section) continue;
#endif

      // chop trailing space
      char *p = strchr((char*)key.c_str(), 0) - 1;
      while (isspace(*p) && p >= key.c_str()) *p-- = 0;

      // separate out all of the values
      if (peekch == '=')
      {
        IniStringList args;

        // strip out one argument
        while (!feof(inf))
        {
          IniString value;
          while ((peekch = fgetc(inf)) != EOF)
            if (peekch == '\n' || !isspace(peekch))
            {
#if (CLIENT_OS != OS_NETWARE)
                ungetc(peekch, inf);
#endif
                break;
            }

          int quoted = (peekch == '"');
#if (CLIENT_OS != OS_NETWARE)
          if (quoted) fgetc(inf);
#else
          if (peekch != EOF)
          {
            if (quoted && peekch == '"')
            {
              while ((peekch = fgetc(inf)) != EOF &&
                  peekch != ',' && peekch != '\n' && peekch != ';');
              goto ValueReady;
            } else if (quoted && peekch == '\n') {
              goto ValueReady;
            } else if (!quoted && (peekch == ',' || peekch == ';' || peekch == '\n')) {
              // strip trailing whitespace
              char *p = strchr((char*)value.c_str(), 0) - 1;
              while (isspace(*p) && p >= value.c_str()) *p-- = 0;
              goto ValueReady;
            } else {
              if (isprint(peekch)) value.append((char)peekch);
            }
          }
#endif

          while ((peekch = fgetc(inf)) != EOF)
          {
            if (quoted && peekch == '"')
            {
              while ((peekch = fgetc(inf)) != EOF &&
                  peekch != ',' && peekch != '\n' && peekch != ';');
              break;
            } else if (quoted && peekch == '\n') {
              break;
            } else if (!quoted && (peekch == ',' || peekch == ';' || peekch == '\n')) {
              // strip trailing whitespace
              char *p = strchr((char*)value.c_str(), 0) - 1;
              while (isspace(*p) && p >= value.c_str()) *p-- = 0;
              break;
            } else {
              if (isprint(peekch)) value.append((char)peekch);
            }
          }

#if (CLIENT_OS == OS_NETWARE)
ValueReady:
#endif
          // add it to our chain
          args.Add(value);

          // if we stopped because of a comment, then ignore to EOL
          if (peekch == ';') while ((peekch = fgetc(inf)) != EOF && peekch != '\n');

          // break if this was the end of the line
          if (peekch == '\n') break;
        }

        // store this key/value pair
        this->addrecord(sect, key, args);
      } else if (peekch == ';') {
        // if we stopped because of a comment, then ignore to EOL
        while ((peekch = fgetc(inf)) != EOF && peekch != '\n');
      }
    }
  }
  fclose(inf);
  return false;
}
/////////////////////////////////////////////////////////////////////////////
// returns TRUE on error
bool IniSection::WriteIniFile(const char *Filename)
{
#if ((CLIENT_OS == OS_VMS) || (CLIENT_OS == OS_HPUX))
  FILE *outf = fopen(Filename, "w");
#else
  FILE *outf = fopen(Filename, "wt");
#endif
  if (outf == NULL) return true;
  this->fwrite(outf);
  fclose(outf);
  return false;
}
/////////////////////////////////////////////////////////////////////////////

