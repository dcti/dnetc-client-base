/* 
 * This module contains general utility stuff including some functions
 * that are called from the installer.
 * 
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * $Id: w32exe.cpp,v 1.1.2.1 2001/01/21 15:10:25 cyp Exp $
*/
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <io.h>
#include <string.h>
#include <sys\types.h>
#include <sys\stat.h>
#include <fcntl.h>
#ifdef __BORLANDC__
  #include <utime.h>
#else
  #include <sys/utime.h>
#endif
#ifndef SH_DENYNO
  #include <share.h>
#endif

typedef short WORD;
typedef long DWORD;
typedef unsigned char BYTE;

#pragma pack(1)
struct _ifh /* size 20 */
 {
   WORD    Machine;                /* 0 */
   WORD    NumberOfSections;       /* 2 */
   DWORD   TimeDateStamp;          /* 4 */
   DWORD   PointerToSymbolTable;   /* 8 */
   DWORD   NumberOfSymbols;        /* 12 */
   WORD    SizeOfOptionalHeader;   /* 16 */
   WORD    Characteristics;        /* 18 */
};
struct _ofh /* size 224 */
{
   WORD    Magic;                  /*  0 */
   BYTE    MajorLinkerVersion;     /*  2 */
   BYTE    MinorLinkerVersion;     /*  3 */
   DWORD   SizeOfCode;             /*  4 */
   DWORD   SizeOfInitializedData;  /*  8 */
   DWORD   SizeOfUninitializedData;/* 12 */
   DWORD   AddressOfEntryPoint;    /* 16 */
   DWORD   BaseOfCode;             /* 20 */
   DWORD   BaseOfData;             /* 24 */
   DWORD   ImageBase;              /* 28 */
   /* --- end of std header, begin NT specific part */
   DWORD   SectionAlignment;       /* 32 */
   DWORD   FileAlignment;          /* 36 */
   WORD    MajorOSVersion;         /* 40 */
   WORD    MinorOSVersion;         /* 42 */
   WORD    MajorImageVersion;      /* 44 */
   WORD    MinorImageVersion;      /* 46 */
   WORD    MajorSubsystemVersion;  /* 48 */
   WORD    MinorSubsystemVersion;  /* 50 */
   DWORD   Win32VersionValue;      /* 52 always 0? */
   DWORD   SizeOfImage;            /* 56 */
   DWORD   SizeOfHeaders;          /* 60 */
   DWORD   CheckSum;               /* 64 */
   WORD    Subsystem;              /* 68 */
   WORD    DllCharacteristics;     /* 70 */
   DWORD   SizeOfStackReserve;     /* 72 */
   DWORD   SizeOfStackCommit;      /* 76 */
   DWORD   SizeOfHeapReserve;      /* 80 */
   DWORD   SizeOfHeapCommit;       /* 84 */
   DWORD   LoaderFlags;            /* 88 */
   DWORD   NumberOfRvaAndSizes;    /* 92 */
   struct { DWORD VirtualAddress, Size; } DataDirectory[16]; /* 96-224 */
   //IMAGE_DATA_DIRECTORY DataDirectory[IMAGE_NUMBEROF_DIRECTORY_ENTRIES];
};
/* first section begins here */
struct _ish /* size 40 */
{
  BYTE    Name[8];
  union 
  { DWORD   PhysicalAddress;
    DWORD   VirtualSize;
  } Misc;
  DWORD   VirtualAddress;
  DWORD   SizeOfRawData;
  DWORD   PointerToRawData;
  DWORD   PointerToRelocations;
  DWORD   PointerToLinenumbers;
  WORD    NumberOfRelocations;
  WORD    NumberOfLinenumbers;
  DWORD   Characteristics;
};
#pragma pack()

#define IMAGE_SUBSYSTEM_UNKNOWN              0   // Unknown subsystem.
#define IMAGE_SUBSYSTEM_NATIVE               1   // Image doesn't require a subsystem.
#define IMAGE_SUBSYSTEM_WINDOWS_GUI          2   // Image runs in the Windows GI subsystem.
#define IMAGE_SUBSYSTEM_WINDOWS_CUI          3   // Image runs in the Windows character subsystem.
#define IMAGE_SUBSYSTEM_OS2_CUI              5   // image runs in the OS/2 character subsystem.
#define IMAGE_SUBSYSTEM_POSIX_CUI            7   // image run  in the Posix character subsystem.
#define IMAGE_SUBSYSTEM_RESERVED8            8   // image run  in the 8 subsystem.

/* ---------------------------------------------------- */

#if 0
static const char *Subsys2Name(WORD uitype)
{
  switch (uitype)
  {
    case IMAGE_SUBSYSTEM_UNKNOWN: return "Unknown";
    case IMAGE_SUBSYSTEM_NATIVE: return "Native";
    case IMAGE_SUBSYSTEM_WINDOWS_GUI: return "GUI";
    case IMAGE_SUBSYSTEM_WINDOWS_CUI: return "CUI";
    case IMAGE_SUBSYSTEM_OS2_GUI: return "OS/2 CUI";
    case IMAGE_SUBSYSTEM_POSIX_CUI: return "Posix CUI";
    case IMAGE_SUBSYSTEM_RESERVED8: return "Reserved";
  }
  return "*bad*";
}
#endif

/* ---------------------------------------------------- */

static int __GetSetSubsysOrVer(const char *filename, int newver, int newsubsys,
                                              int *oldverP, int *oldsubsysP)
{
  WORD oldver = 0, oldsubsys = 0, binbits = 0;
  int handle, madechange = 0;
  struct stat statblk;
  
  if ( stat( filename, &statblk ) != 0)
  {
    //printf("EXEVER: Unable to stat %s\n", filename );
    return -1;
  }
  else if (( handle = sopen( filename,
      (((newver || newsubsys) ? O_RDWR: O_RDONLY)|O_BINARY), SH_DENYNO ))==-1)
  {
    //printf("can't open file %s\n", filename );
    return -1;
  }
  else
  {
    long offset = 0;
    struct utimbuf filetimes;
    char rbuf[0x100]; char *p;
      
    filetimes.actime = statblk.st_atime;
    filetimes.modtime = statblk.st_mtime;
      
    if (read( handle, rbuf, sizeof(rbuf) ) != sizeof(rbuf))
    {
      rbuf[0] = rbuf[1] = '\0'; 
      madechange = -1;
    }
    else if ((rbuf[0]=='M' && rbuf[1]=='Z') || (rbuf[1]=='M' && rbuf[2]=='Z'))
    {
      p = &rbuf[0x3C]; /*off from start of file to NE/LE image */
      if ((offset = *((long *)p)) != 0)
      {
        if (lseek( handle, offset, 0 ) != offset)
          madechange = -1;
        else if (read( handle, rbuf, sizeof(rbuf) ) != sizeof(rbuf))
        { /* overwrite in case read was partial */
          rbuf[0] = 'M'; rbuf[1] = 'Z'; 
          madechange = -1;
        }
      }
    }
    if ((rbuf[0]=='N' && rbuf[1]=='E' && 
         rbuf[0x36] == 0x02 /* win */ || rbuf[0x36]==0x04 /* win386 */))
    {
      oldver = (WORD) (((WORD) rbuf[0x3F]) * 100 + ((WORD) rbuf[0x3E]));
      oldsubsys = IMAGE_SUBSYSTEM_WINDOWS_GUI;
      binbits = 16;
        
      if (newsubsys != 0 && newsubsys != oldsubsys)
      {
        //printf("You can't set CUI/GUI for Win16 apps\n");
      }
      if (newver != 0 && newver != oldver)
      {
        offset += 0x3F;
        if (lseek(handle,offset,0) != offset)
          madechange = -1;
        else
        {
          rbuf[0] = (char) (newver / 100);
          rbuf[1] = (char) (newver % 100);
          if (write(handle,rbuf,2)!=2)
            madechange = -1;
          else
          {
            //printf("New version (win16) %d.%d\n",newver/100,newver%100);
            madechange = 1;
          }
        }
      }
    }
    else if (rbuf[0]=='P' && rbuf[1]=='E' && !rbuf[2] && !rbuf[3])
    {
      struct _ifh *ifh;
      struct _ofh *ofh;
      p = &rbuf[4];
      ifh = (struct _ifh *)p;
      p += sizeof(struct _ifh);
      ofh = (struct _ofh *)p;
      if (ifh->SizeOfOptionalHeader >= sizeof(struct _ofh) &&
         ofh->Magic == 0x010B /* file == 0x10B, rom == 0x107 */ )
      {
        oldver = (WORD) ((ofh->MajorSubsystemVersion * 100) +
                 (ofh->MinorSubsystemVersion));
        oldsubsys = ofh->Subsystem;
        binbits = 32;

        if (newsubsys == oldsubsys)
          newsubsys = 0;
        if (newver == oldver)
          newver = 0;

        if (newsubsys != 0 || newver != 0)
        {
          if (newsubsys != 0)
            ofh->Subsystem = (WORD)(newsubsys);
          if (newver != 0)
          {
            ofh->MajorSubsystemVersion = (WORD) (newver / 100);
            ofh->MinorSubsystemVersion = (WORD) (newver % 100);
          }   
          if ( lseek( handle, offset, 0 ) != offset )
            madechange = -1;
          else if (write( handle, rbuf, sizeof(rbuf) ) != sizeof(rbuf))
            madechange = -1;
          else
            madechange = 1;
        }
      }    
    }  
    close(handle);
    if (madechange)
      utime( filename, &filetimes );
  }
  if (madechange < 0)
  {
    //printf("Write/seek error. File unchanged.\n");
    return -1;
  }
  if (binbits != 0)
  {
    if (oldverP)
      *oldverP = oldver;
    if (oldsubsysP)
      *oldsubsysP = oldsubsys;
    return binbits;
  }
  return 0; /* not a windows executable */
}    

/* ---------------------------------------------------- */

/* returns <0 on error, else 0 == cui, >0 == gui */
int winIsGUIExecutable(const char *filename)
{
  int subsys = 0;
  if (!filename)
    return -1;
  if (__GetSetSubsysOrVer(filename, 0, 0, NULL, &subsys ) <= 0)
    return -1;
  if (subsys == IMAGE_SUBSYSTEM_WINDOWS_GUI)
    return 1;
  return 0;
}

/* ---------------------------------------------------- */

int install_cmd_exever(int argc, char *argv[])
{
  int newver = 0, newsubsys = 0;
  if (argc >= 3)
  {
    if (strcmpi(argv[2],"cui")==0)
      newsubsys = IMAGE_SUBSYSTEM_WINDOWS_CUI;
    else if (strcmpi(argv[2],"gui")==0)
      newsubsys = IMAGE_SUBSYSTEM_WINDOWS_GUI;
    else if (strcmpi(argv[2],"native")==0)
      newsubsys = IMAGE_SUBSYSTEM_NATIVE;
    else if (strcmpi(argv[2],"posix")==0)
      newsubsys = IMAGE_SUBSYSTEM_POSIX_CUI;
    else
    {
      newver = -2;
      if (*argv[2]>='0' && *argv[2]<='9')
      {
        char *zz = strchr(argv[2],'.');
        if (zz && zz[1]>='0' && zz[1]<='9')
        {
          int minor = atoi(zz+1);
          newver = atoi(argv[2]);
          if (newver > 255 || minor > 255)
            newver = -2;
          else
            newver = (newver*100)+minor;
        }
      }
    }
  }
  if (argc >= 2 && newver != -2)
  {
    if (__GetSetSubsysOrVer(argv[1], newver, newsubsys, NULL, NULL) > 0)
      return 0;
  }
  return -1;
}

/* ---------------------------------------------------- */

int install_cmd_copyfile(int argc, char *argv[])
{
  if (argc >= 3)
  {
    if ( strcmp( argv[1], ".self" ) == 0 )
      argv[1] = argv[0];
  #if defined(_WINNT_)
    chmod( argv[2], 0 );
    if (CopyFile( argv[1], argv[2], FALSE ) != 0)
      return 0;
  #else
    struct stat statblk;
    int inhandle, outhandle;
    if ( stat( argv[1], &statblk ) == 0)
    {
      if ((inhandle = sopen ( argv[1], O_RDONLY, SH_DENYNO, 0 )) != -1)
      {
        int success = 0;
        chmod( argv[2], 0 );
        if ((outhandle = open( argv[2], O_RDWR|O_TRUNC|O_CREAT|O_BINARY))!=-1)
        {
          char buffer[512];
          int bytesread;
          success = 1;
          while ((bytesread = read(inhandle, buffer, sizeof(buffer))) > 0)
          {
            int byteswrite = write(outhandle, buffer, bytesread);
            if (byteswrite != bytesread)
            {
              success = 0;
              break;
            }
          }
          close(outhandle);
          if (!success)
            unlink( argv[2] );
          else
          {
            struct utimbuf filetimes;
            filetimes.actime = statblk.st_atime;
            filetimes.modtime = statblk.st_mtime;
            utime( argv[2], &filetimes );
            chmod( argv[2], statblk.st_mode );
          }
        }
        close(inhandle);
        if (success)
          return 0;
      }
    }
  #endif
  }
  return -1;
}

/* ---------------------------------------------------- */

