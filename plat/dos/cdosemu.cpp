/* Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * ----------------------------------------------------------------
 * get informational message used by the client (from PrintBanner)
 * used nowhere else, and is now quite useless :) 
 *
 * [it used to be the function that detected which yield method to use, 
 * but that is now in cdosidle]
 * ----------------------------------------------------------------
 *
*/
const char *cdosemu_cpp(void) {
return "@(#)$Id: cdosemu.cpp,v 1.1.2.1 2001/01/21 15:10:19 cyp Exp $"; }

#include <stdlib.h>   /* getenv() */
#include <stdio.h>    /* sprintf() */
#include <string.h>   /* strcmp() */
#include "cdosemu.h"  /* keep prototypes in sync */

const char *dosCliGetEmulationDescription(void)
{
  static char emudesc[40];
  unsigned short sversion;
  unsigned int vermajor, verminor;
  unsigned char inrom, oemid, dosmajor, dosminor;


  _asm  push bx
  _asm  mov ax,3000h
  _asm  int 21h
  _asm  mov oemid,bh
  _asm  mov dosmajor,al
  _asm  mov dosminor,ah
  _asm  pop bx

  _asm  push bx
  _asm  mov  ax, 3306h
  _asm  xor  bx, bx
  _asm  dec  bx
  _asm  cmp  bx, 1           /* clear carry flag */
  _asm  int  21h
  _asm  mov  dx,bx
  _asm  pop  bx
  _asm  sbb  cx,cx
  _asm  or   dx,cx
  _asm  cmp  al,255
  _asm  cmc 
  _asm  sbb  cx,cx
  _asm  or   dx,cx
  _asm  mov  sversion,dx

  inrom = 0;
  if (((sversion>>8) < 100) && ((sversion & 0xff) >= 5))
  {
    dosminor = (unsigned char)((sversion>>8)&0xff);
    dosmajor = (unsigned char)(sversion&0xff);
    _asm push bx
    _asm mov ax,3001h
    _asm int 21h
    _asm and bh,4
    _asm mov inrom,bh
    _asm mov sversion,ax
    _asm pop bx

    if (!inrom && oemid == 0x00 && dosmajor == 6 && dosminor == 0)
    {                 /* Novell DOS 7.0 or IBM DOS 6.1 (no IBM DOS 6.0)) */
      if ((sversion>>8)!=0xff)
      {
        _asm mov ax,4452h /* 'DR' */
        _asm cmp ax,4453h /* set carry */
        _asm int 21h
        _asm sbb ax,ax
        _asm inc al
        _asm add dosmajor,al
      }
      else
      {
        _asm mov ax,6D00h
        _asm int 21h
        _asm cmp ax,1
        _asm sbb ax,ax
        _asm inc al
        _asm add dosmajor,al
      }
      if (dosmajor == 7)
        oemid = 0xef;  /* Novell DOS 7.0 */
      else
        dosminor = 10; /* IBM DOS 6.10 */
    }
  }
  else if (dosmajor == 10 && oemid == 0)
    return "OS/2 1.x VDM";

  /* -------------------------------------------- */

  {
    char *envp;
    sversion = 0;
    if (dosmajor == 5 && dosminor == 50) /* DOS ver 5.50 */
      sversion = 1;
    else if ((envp = getenv("OS"))!=((char *)0)) 
    {
      if ( strcmp( envp, "Windows_NT" )==0 )
        sversion = 1;
    }
    if (sversion)
      return "Windows NT/NTAS VDM"; /* 19 chars */
  }

  /* -------------------------------------------- */

  if (dosmajor==20) 
  {
    if (dosminor==0 || dosminor==10 || dosminor==20)
      sversion = (short)(2 + (((short)(dosminor%10))<<8));
    else
    {
      _asm push ds
      _asm push es
      _asm push bx
      _asm push bp
      _asm push di
      _asm push si
      _asm mov  ax,4010h
      _asm int  2fh
      _asm cmp  ax,1
      _asm sbb  ax,ax
      _asm and  ax,bx
      _asm mov  sversion,ax
      _asm pop  si
      _asm pop  di
      _asm pop  bp
      _asm pop  bx
      _asm pop  es
      _asm pop  ds
    }
    if (sversion)
    {
      sprintf( emudesc, "OS/2 %d.%d VDM", sversion&0xff, sversion>>8);
      return emudesc;
    }
  }

  /* -------------------------------------------- */

  _asm push bx
  _asm mov ax,2B01h  /* DESQview installation check */
  _asm mov cx,4445h  /* 'DE' */
  _asm mov dx,5351h  /* 'SQ' */
  _asm mov bx,0      /* clear version # */
  _asm int 21h
  _asm cmp al,0FFh   /* set carry flag if supported (<0xFF) */
  _asm sbb ax,ax     /* make zero if not supported, 0xFFFF otherwise */
  _asm and bx,ax     /* clear version if not supported */
  _asm mov sversion,bx
  _asm pop bx
  vermajor = sversion & 0xff;
  verminor = (sversion >> 8) & 0xff;
  if (sversion)
  {
    sprintf( emudesc, "DesqView v%d.%d VM", vermajor, verminor ); //18 chars
    return emudesc;
  }

  /* -------------------------------------------- */

  _asm push bx             /* save bx */
  _asm mov  bx,0f000h      /* this is the segment we want a selector for */
  _asm mov  ax,2           /* DPMI: MAP SEGMENT TO DESCRIPTOR */
  _asm int  31h            /* <-ax=2, bx=seg -> ax=sel and cflag clear */
  _asm pop  bx             /* restore bx */
  _asm cmc                 /* flip carry */
  _asm sbb  cx,cx          /* make 0xffff if not error */
  _asm and  ax,cx          /* make 0 if error */
  _asm mov  sversion,ax    /* save descriptor */
  if (sversion)
  {
    _asm push es
    _asm push bx
    _asm mov  ax, sversion
    _asm mov  es, ax
    _asm mov  bx, 0ffe0h
    _asm mov  ax, es:[bx+0]
    _asm mov  dx, es:[bx+2]
    _asm mov  cx, es:[bx+4]
    _asm mov  bx, es:[bx+6]
    _asm xor  ax, 'D$'
    _asm xor  dx, 'SO'
    _asm xor  cx, 'ME'
    _asm xor  bx, '$U'
    _asm or   ax, dx
    _asm or   ax, cx
    _asm or   ax, bx
    _asm mov  sversion, ax
    _asm pop  bx
    _asm pop  es
    if (sversion == 0)
    {
      _asm push bx
      _asm xor ax,ax
      _asm xor bx,bx
      _asm int 0xe6
      _asm sub ax,0xaa55
      _asm cmp ax,1
      _asm sbb ax,ax
      _asm and bx,ax
      _asm mov sversion,bx
      _asm pop bx
      if (sversion)
      {
        sprintf( emudesc, "DOSEMU v%d.%d VM", sversion>>8, sversion&0xff );
        return emudesc;
      }
    }
  }

  /* -------------------------------------------- */

  if (oemid == 0xEE || oemid == 0xef)  /* Digital Research or Novell */
  {
    _asm mov ax,4452h  /* "DR */
    _asm cmp ax,4453h  /* set carry */
    _asm int 21h
    _asm cmc
    _asm sbb cx,cx
    _asm and ax,cx
    _asm mov sversion,ax
    if (!sversion || ((sversion >> 8)&4)!=0)
    {
      _asm mov ax,4451h  /* Concurrent DOS */
      _asm cmp ax,4452h  /* set carry */
      _asm int 21h
      _asm cmc
      _asm sbb cx,cx
      _asm and ax,cx
      _asm mov sversion,ax
    }
    vermajor = verminor = 0;
    if (sversion)
    {
      vermajor = ((sversion>>8) & 0xff); /* actually CP/M type */
      verminor = sversion & 0x00ff; /* version */
    }
    if ((vermajor & 4)!=0) /* multi user */
    {
      if (verminor == 0x66)
        return "DR Multiuser DOS 5.1";
      strcpy(emudesc,"Concurrent DOS");
      if (verminor == 0x32)
        return strcat(emudesc, " 3.2");
      else if (verminor == 0x41)
        return strcat(emudesc, " 4.1");
      else if (verminor == 0x67)
        return strcat(emudesc, " 5.1");
      sprintf(&emudesc[strlen(emudesc)],"/XM %d.%d", 
                      verminor>>4, verminor&0x0f );
      return emudesc;
    }
    else if (verminor)
    {
      char *p = NULL;
      if (verminor < 0x63) 
        return strcat( emudesc, "DR-DOS+" );
      switch( verminor )
      {
        case 0x63: p = "3.41"; break;
        case 0x64: p = "3.42"; break;
        case 0x65: p = "5.0";  break;
        case 0x66: p = "5.06"; break;  /* made up */
        case 0x67: p = "6.0";  break;
        case 0x68: p = "6.08"; break; /* made up */
        case 0x69: p = "6.09"; break; /* made up */
        case 0x70: return "PalmDOS";
        case 0x71: p = "6.0" ; break; /* 6.0 business update*/
        case 0x72: return "Novell DOS 7.0";
      }
      if (p)  
        return strcat( strcpy( emudesc, "DR-DOS " ), p );
      if (dosmajor == 7 && dosminor == 1) /* "Caldera OpenDOS 7.01" */
        return "OpenDOS 7.01";
      if (dosmajor == 7 && dosminor == 2)
      { /* two versions: "Caldera OpenDOS 7.02" and "Caldera DR-DOS 7.02" */
        return "Caldera DOS 7.02";
      }
      /* 7.03 and above are all "Caldera DR-DOS" */
      /* FYI: SET shows "OS=DRDOS" and "VER=7" */
      sprintf(emudesc, "DR-DOS %d.%02d", dosmajor, dosminor );
      return emudesc; 
    }
  }

  /* -------------------------------------------- */
  
#if 0 /* not needed, caught in oemid table below */
  _asm push ds
  _asm push es
  _asm push bx
  _asm push bp
  _asm push di
  _asm push si
  _asm mov  ax,12ffh
  _asm mov  bl,0
  _asm int  2fh
  _asm add  al,22h
  _asm cmp  al,255
  _asm cmc
  _asm sbb  ax,ax
  _asm mov  sversion,ax
  _asm pop  si
  _asm pop  di
  _asm pop  bp
  _asm pop  bx
  _asm pop  es
  _asm pop  ds
  
  if (sversion)
  {
    sprintf(emudesc, "FreeDOS E%d.%d", dosmajor, dosminor );
    return emudesc;
  }
#endif

  /* -------------------------------------------- */

  _asm push ds
  _asm push es
  _asm push bx
  _asm push bp
  _asm push di
  _asm push si
  _asm mov  ax,0xAF00
  _asm int  2Fh
  _asm cmp  al,255
  _asm cmc
  _asm sbb  ax,ax
  _asm mov  sversion,ax
  _asm pop  si
  _asm pop  di
  _asm pop  bp
  _asm pop  bx
  _asm pop  es
  _asm pop  ds
  if (sversion)
    return "WinDOS 5.x";

  /* -------------------------------------------- */

  _asm push ds
  _asm push es
  _asm push bx
  _asm push bp
  _asm push di
  _asm push si
  _asm mov  ax,0xF400
  _asm int  2Fh
  _asm mov  cx,ax
  _asm sub  cx,0xF400
  _asm cmp  cx,1
  _asm sbb  cx,cx
  _asm xor  ah,ah
  _asm and  ax,cx
  _asm mov  sversion,ax
  _asm pop  si
  _asm pop  di
  _asm pop  bp
  _asm pop  bx
  _asm pop  es
  _asm pop  ds
  if (sversion)
  {
    unsigned short ddversion;
    _asm push ds
    _asm push es
    _asm push bx
    _asm push bp
    _asm push di
    _asm push si
    _asm mov  ax,0xE400
    _asm int  2Fh
    _asm mov  cx,ax
    _asm sub  cx,0xE400
    _asm cmp  cx,1
    _asm sbb  cx,cx
    _asm xor  ah,ah
    _asm and  ax,cx
    _asm mov  ddversion,ax
    _asm pop  si
    _asm pop  di
    _asm pop  bp
    _asm pop  bx
    _asm pop  es
    _asm pop  ds
    if (ddversion == sversion)
    {
      sprintf(emudesc, "DoubleDOS E%d.%d", dosmajor, dosminor );
      return emudesc;
    }
  }

  /* -------------------------------------------- */

  _asm mov ax,3000h
  _asm mov bx,ax
  _asm mov cx,ax
  _asm mov dx,ax
  _asm int 21h
  _asm sub cx,3000h
  _asm cmp cx,1
  _asm cmc
  _asm sbb cx,cx
  _asm and ax,cx
  _asm sub dx,3000h
  _asm cmp dx,1
  _asm cmc
  _asm sbb dx,dx
  _asm and ax,dx
  _asm mov sversion,ax
  if (sversion)
  {
    sprintf(emudesc,"PC-MOS/386 v%d.%d", sversion&0xff, sversion>>8);
    return emudesc;
  }

  /* -------------------------------------------- */

  if (dosminor < 10)
    dosminor *= 10;
  if (oemid != 0xff)
  {
    static struct { char id; const char *name; } id2name[] = {
      { 0x00, "IBM-" }, { 0x01, "Compaq-" }, { 0x02, "MS-" }, 
      { 0x04, "AT&T-" }, { 0x05, "Zenith-" }, { 0x06, "HP-" },
      { 0x07, "Bull-" }, { 0x0D, "Packard Bell " }, { 0x16, "DEC-" },
      { 0x23, "Olivetti-" }, { 0x28, "TI-" }, { 0x29, "Toshiba-" },
      { 0x33, "Novell-" }, { 0x34, "MSMS-" }, { 0x35, "MSMS-" },
      { 0x4D, "HP-" }, { 0x5E, "Rx" }, { 0x66, "PTS-" }, 
      { 0x99, "GS Embedded " }, { 0xEE, "DR-" }, { 0xEF, "Novell-" }, 
      { 0xFD, "Free" }, { 0xFF, "MS-" /* also Phoenix- */ } };
    unsigned int i;
    for (i=0;i<(sizeof(id2name)/sizeof(id2name[0]));i++)
    {
      if (id2name[i].id == oemid)
      {
        sprintf(emudesc,"%sDOS %d.%02d", id2name[i].name, dosmajor, dosminor );
        return emudesc;
      }
    }
    sprintf(emudesc,"???-DOS %d.%02d (oem=0x%02x)", dosmajor, dosminor, oemid);
    return emudesc;
  }

  /* -------------------------------------------- */

  _asm  push es
  _asm  push ds
  _asm  push bx
  _asm  push bp
  _asm  push di
  _asm  push si
  _asm  mov  ax, 1600h
  _asm  int  2fh  /* ->al=major, ah=minor */
  _asm  mov  sversion, ax
  _asm  pop  si
  _asm  pop  di
  _asm  pop  bp
  _asm  pop  bx
  _asm  pop  ds
  _asm  pop  es
  vermajor = sversion & 0xff;
  verminor = (sversion >> 8) & 0xff;
  if (((vermajor)!=0) && ((vermajor & 0x7F)!=0x0) && (vermajor&0x80)==0)
  {
    sprintf(emudesc, "Win%s v%d.%d DOS Box", ((vermajor>=4)?("32"):("dows")),
                     vermajor, verminor ); /* 19 or 21 chars */
    return emudesc;
  }

  /* -------------------------------------------- */

  sprintf(emudesc,"MS-DOS %d.%02d", dosmajor, dosminor );
  return emudesc;
}      

