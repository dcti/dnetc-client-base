/* 
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * ------------------------------------------------------------------
 * real sleep() and usleep() for DOS.
 * also calls (the equivalent of) kbhit() while sleeping
 * ------------------------------------------------------------------
 *
*/
const char *cdosidle_cpp(void) {
return "@(#)$Id: cdosidle.cpp,v 1.1.2.1 2001/01/21 15:10:19 cyp Exp $"; }

#include <stdlib.h>   /* getenv() */
#include <string.h>   /* strcmp() */
#define  INCLUDING_CDOSIDLE_CPP
#include "cdosidle.h" /* keep prototypes in sync */

/* ----------------------------------------------------------------- */

static void __yield(void)
{
  static int yieldtype = -1;
  #define YIELDTYPE_DESQVIEW 0x01
  #define YIELDTYPE_WINOS2   0x02
  #define YIELDTYPE_DPMI     0x03
  #define YIELDTYPE_BIOSWAIT 0x04
  #define YIELDTYPE_DOSIDLE  0x05

  if (yieldtype == -1)
  {
    unsigned short sversion;
    int verlsb, vermsb;
    char *p;

    if ((p = getenv("OS")) != ((char *)0))
    {
      if (strcmp( p, "Windows_NT" )==0)
        yieldtype = YIELDTYPE_WINOS2;
    }  
    if (yieldtype == -1)
    {
      _asm  mov ax, 3306h
      _asm  mov bx, 0ffffh
      _asm  int 21h
      _asm  mov sversion, bx
      verlsb = sversion & 0xff;
      vermsb = (sversion >> 8) & 0xff;

      if (vermsb <=100 && verlsb >= 10 && (verlsb%10) == 0)
        yieldtype = YIELDTYPE_WINOS2;
    }
    if (yieldtype == -1)
    {
      _asm  mov ax, 1600h
      _asm  int 2fh
      _asm  mov sversion, ax
      verlsb = sversion & 0xff;
      vermsb = (sversion >> 8) & 0xff;
    
      if (((verlsb)!=0) && ((verlsb & 0x7F)!=0x0))
        yieldtype = YIELDTYPE_WINOS2;
    }
    if (yieldtype == -1)
    {
      _asm mov ax,2B01h  /* DESQview installation check */
      _asm mov cx,4445h  /* 'DE' */
      _asm mov dx,5351h  /* 'SQ' */
      _asm mov bx,0      /* clear version # */
      _asm int 21h
      _asm cmp al,0FFh   /* set carry flag if supported (<0xFF) */
      _asm sbb ax,ax     /* make zero if not supported, 0xFFFF otherwise */
      _asm and bx,ax     /* clear version if not supported */
      _asm mov sversion,bx

      if (sversion != 0)
        yieldtype = YIELDTYPE_DESQVIEW;
    }
    if (yieldtype == -1)
    {
      _asm mov ax,0x1680
      _asm int 0x2F
      _asm cmp al,1
      _asm sbb ax,ax
      _asm mov sversion,ax
        
      if (sversion != 0) /* something that supports winos2 type yield */
        yieldtype = YIELDTYPE_WINOS2;
    }
    if (yieldtype == -1)
    {
      _asm mov ax,1680h
      _asm int 2fh
      _asm mov sversion,ax
      if ((sversion & 0xff) != 0x80)
        yieldtype = YIELDTYPE_DPMI;
    }
  }

  if (yieldtype == YIELDTYPE_DPMI || yieldtype == YIELDTYPE_WINOS2)
  {
    _asm mov ax,1680h  
    _asm int 2fh
  }
  else if (yieldtype == YIELDTYPE_DESQVIEW)
  {
    _asm mov ax,1000h                         /* DesqView idle */
    _asm int 15h
  }
  else if (yieldtype == YIELDTYPE_BIOSWAIT) /* cx:dx == time to wait */
  {
    _asm mov ah, 86h
    _asm mov cx, 0
    _asm mov dx, 1
    _asm int 15h
  }
  else if (yieldtype == YIELDTYPE_DOSIDLE)
  { 
    _asm int 28h
  }
  _asm mov  ah,0x0b /* benign dos call (kbhit()) */
  _asm int  0x21    /* to keep ^C handling alive */
  return;
}      

/* ----------------------------------------------------------------- */

static unsigned long __getuclock(void) /* usecs in the last hour */
{
  unsigned long ticks,uticks;
  _asm 
  {
    push es
    mov  ax,40h
    mov  es,ax
    mov  ecx,dword ptr es:[6Ch]
    reread:
    xor  eax,eax
    out  043h,al
    in   al,040h
    mov  ah,al
    in   al,040h
    mov  edx,dword ptr es:[6Ch]
    cmp  edx,ecx
    mov  ecx,edx
    jnz  reread
    pop  es
    xchg ah,al
    not  ax           /* timer counts down from 65535 so we flip it */
    mov  ticks,ecx
    mov  uticks,eax
  }
  ticks%= 65543ul;    /* make sure we only have ticks in the last hour */
  ticks*= 54925;      /* 54935 == micro sec per BIOS clock tick. */
  return ( ticks + (( (uticks * 8381ul) + (ticks % 1000) ) / 1000) );
}


/* ----------------------------------------------------------------- */

static void __xxsleep(unsigned long secs, unsigned long usecs )
{
  if (secs == 0 && usecs == 0)
    __yield();
  else
  {
    unsigned long uclock = 0xfffffffful; 
    if (usecs > 1000000ul)
    {
      secs += usecs / 1000000ul;
      usecs %= 1000000ul;
    }
    do
    {
      unsigned long nuclock, e_secs, e_usecs;
      __yield();
      nuclock = __getuclock(); /* usecs in the last hour */
      e_secs = e_usecs = 0;
      if (uclock != 0xfffffffful)
      {
        if (nuclock < uclock)
          e_usecs = 60000000ul;
        e_usecs += nuclock;
        e_usecs -= uclock;
        if (e_usecs > 1000000ul)
        {
          e_secs = e_usecs/1000000ul;
          e_usecs %= 1000000ul;
        }
      }
      uclock = nuclock;
      if (secs < e_secs || (secs == e_secs && usecs < e_usecs))
        break;
      if (usecs < e_usecs)
      {
        secs--;
        usecs += 1000000ul;
      }
      secs -= e_secs;
      usecs -= e_usecs;
    } while (secs || usecs);
  }
  return;  
}  
    
/* ----------------------------------------------------------------- */

unsigned int delay(unsigned int msecs)
{
  __xxsleep( (msecs/1000), (1000ul * ((unsigned long)(msecs%1000))));
  return 0;
}

/* ----------------------------------------------------------------- */

unsigned int usleep(unsigned int usecs)
{
  __xxsleep( 0, usecs );
  return 0;
}

/* ----------------------------------------------------------------- */

unsigned int sleep(unsigned int secs)
{
  __xxsleep( secs, 0 );
  return 0;
}
