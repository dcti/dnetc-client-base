/*
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * Written by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * -----------------------------------------------------------------
 * these are console i/o functions not defined in the C spec.
 *
 * dosCliConIsScreen() - isatty(fileno(stdout)) but also determines
 *                       various other console parameters
 * the others: dosCliConGetPos(), dosCliConSetPos(), dosCliConGetSize()
 * and dosCliConClear(). 
 * -----------------------------------------------------------------
 *
*/
const char *cdoscon_cpp(void) {
return "@(#)$Id: cdoscon.cpp,v 1.1.2.1 2001/01/21 15:10:18 cyp Exp $"; }

#include "cdoscon.h" /* keep prototypes in sync */

static struct
{
  int scrheight;
  int scrwidth;
  int conisatty;
  int scrpage;
  int scrmode;
} cdoscon = {-1,-1,-1,-1,-1};

/* ----------------------------------------------------------------- */

int dosCliConIsScreen(void)
{
  if (cdoscon.conisatty == -1)
  {
    char isacon = 0;
    cdoscon.conisatty = 0;
    _asm mov ax,4400h
    _asm mov bx,1
    _asm xor dl,dl
    _asm int 21h
    _asm cmc
    _asm sbb ax,ax
    _asm and dx,ax
    _asm mov al,dl
    _asm and dl,82h             /* 0x80=isdevice, 0x02=isstdout */
    _asm and al,12              /* 0x04=nuldev 0x08=clockdev */
    _asm cmp al,1
    _asm sbb al,al
    _asm and dl,al
    _asm cmp dl,82h
    _asm sbb dl,dl
    _asm inc dl
    _asm mov isacon,dl
    if (isacon)
    {
      char rows, cols, mode, page;
      _asm push bx
      _asm mov  ah,0Fh                 /* get page/mode/maxcol */
      _asm int  10h                    /* ->ah=# of cols,al=mode,bh=page */
      _asm mov  cols, ah
      _asm mov  mode, al
      _asm mov  page, bh
      _asm mov  bl, ah
      _asm xor  bh, bh
      _asm mov  dx, es
      _asm mov  ax, 40h
      _asm mov  es, ax
      _asm mov  ax, es:[4Ch]          /* size of regen buffer in bytes */
      _asm mov  es, dx
      _asm xor  dx, dx
      _asm shr  ax, 1                 /* 2 bytes per cell */
      _asm div  bx
      _asm mov  rows,al               
      _asm pop  bx
      cdoscon.scrwidth = cols;
      cdoscon.scrheight = rows;
      cdoscon.scrpage = page;
      cdoscon.scrmode = mode;
      cdoscon.conisatty = 1;
    }
  }
  return cdoscon.conisatty;
}

/* ----------------------------------------------------------------- */

int dosCliConGetPos( int *colP, int *rowP )
{
  if (dosCliConIsScreen())
  {
    char row, col, page = (char)cdoscon.scrpage;
    _asm push bx
    _asm mov  bh,page
    _asm mov  ah,3               /* get cursor size and pos */
    _asm int  10h                /* <-bh=page, -> cx=curs, dh=row, dl=col */
    _asm pop  bx                     
    _asm mov  row,dh
    _asm mov  col,dl
    if (colP) *colP = (int)col;
    if (rowP) *rowP = (int)row;
    return 0;
  }
  return -1;
}  

/* ----------------------------------------------------------------- */

int dosCliConSetPos( int acol, int arow )
{
  if (dosCliConIsScreen())
  {
    char row = (char)arow, col=(char)acol, page = (char)cdoscon.scrpage;
    _asm push bx
    _asm mov  bh,page
    _asm mov  dh,row
    _asm mov  dl,col
    _asm mov  ah,02h   /* set cursor pos */
    _asm int  10h      /* <-dh=row, dl=col, bh=page# */
    _asm pop  bx
    return 0;
  }
  return -1;
}  

/* ----------------------------------------------------------------- */

int dosCliConGetSize( int *cols, int *rows )
{
  if (dosCliConIsScreen())
  {
    if (rows) *rows = cdoscon.scrheight;
    if (cols) *cols = cdoscon.scrwidth;
    return 0;
  }
  return -1;
}
    
/* ----------------------------------------------------------------- */

#include <stdio.h>

int dosCliConClear(void)
{
  int rows, cols;
  if (dosCliConGetSize(&rows,&cols) == 0)
  {
    char buffer[256];
    size_t todo, done, totaldone=0, scrsize = (rows*cols);
    for (done=0;done<sizeof(buffer);done++)
      buffer[done]=' ';
    fflush(stdout);
    dosCliConSetPos(0,0);
    do
    {
      todo = sizeof(buffer);
      if (todo > (scrsize-totaldone))
        todo = (scrsize-totaldone);
      if ((done = fwrite(buffer,sizeof(char),todo,stdout)) == 0)
        break;
      totaldone+=done;
    } while (totaldone<scrsize);
    fflush(stdout);
    dosCliConSetPos(0,0);
    return 0;
  }
  return -1;
}

