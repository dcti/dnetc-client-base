/*
 * Created by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * Copyright distributed.net 1997-1998 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 * 
 * -----------------------------------------------------------------
 * This module is for hooking IRQ1 (int 9h) and clearing the keyboard
 * buffer when a ^C is seen.
 * This is needed because ^C's sitting in the keyboard buffer behind
 * another key don't get seen until the first key is cleared (or worse,
 * they are discarded if the keyboard buffer is full).
 *
 * (note: to keep dos ^C checking alive, usleep() [cdosidle.cpp] polls
 * int 21h with the equivalent of a kbhit() check)
 * -----------------------------------------------------------------
*/
const char *cdoskeyb_cpp(void) {
return "@(#)$Id: cdoskeyb.cpp,v 1.1.2.1 2001/01/21 15:10:22 cyp Exp $"; }

#if !defined(__WATCOMC__)
#error This is watcom C specific
#else

#include <dos.h>

static struct 
{ void (__interrupt __far *prev_irq_01)(); /* 6 bytes */
  short seg40_sel;                         /* 2 bytes */
  char ctrl;                               /* 1 byte */
} _hdata;

#define HANDLER_CODESIZE 512
#define HANDLER_DATASIZE 10

static void __interrupt __far _pmkeybhandler()
{
  char ctrl = _hdata.ctrl;    /* we need to make local copies because the */
  short seg40_sel = _hdata.seg40_sel; /* inline assembler can't reference */
                              /* structure members */
  _asm                       
  {
    in  al,60h                /* get the scan code */

    cmp al,1dh                /* is it the ctrl key make code */
    mov ah,4                  /* bios keyflags equivalent of 'ctrl down' */
    jz  _isctrl               /* go set the static flag if so */
    cmp al,9dh                /* is it the ctrl key break code */
    jnz _notctrl              /* go check other keystrokes if not */
    xor ah,ah                 /* what we set the static flag to */
    _isctrl:                  
    mov ctrl,ah               /* save the ctrl key state (0x4 or 0x0) */
    jmp short _nochange       /* nothing more to do */

    _notctrl:                
    mov ah,ctrl               /* get the bios keyflags state */
    cmp ax,042Eh              /* is it the ^C code */
    jnz _nochange             /* nothing more to do if not */

    mov  ax,seg40_sel         /* get the selector for the bios data area */
    or   ax,ax                /* have it already? */
    jnz  _clearbuf            /* go ahead with buffer clear if so */

    push bx                   /* save bx */
    mov  bx,40h               /* this is the segment we want a selector for */
    mov  ax,2                 /* DPMI: MAP SEGMENT TO DESCRIPTOR */
    int  31h                  /* <-ax=2, bx=seg -> ax=sel and cflag clear */
    pop  bx                   /* restore bx */
    jc   _nochange            /* error, can't clear the buffer */
    mov  seg40_sel,ax         /* save the selector */

    _clearbuf:                
    push bx
    push es                   /* save es for use as the selector to seg 40h */
    mov  es,ax                /* load the selector to the bios data area */
    mov  bx,1Ch               /* ptr to offset to tail */
    mov  ax,word ptr es:[bx]  /* load tail ptr */
    sub  bx,2                 /* ptr to offset to head */
    mov  word ptr es:[bx],ax  /* make head and tail the same */
    pop  es                   /* restore es */
    pop  bx

    _nochange:
  }
  _hdata.ctrl = ctrl;
  _hdata.seg40_sel = seg40_sel;
  _chain_intr( _hdata.prev_irq_01 ); /* never returns */
}  


static void _installhandler(void)
{
  char islocked = 0;

  _hdata.prev_irq_01 = 0;
  _hdata.seg40_sel = 0;
  _hdata.ctrl = 0;
  
  _asm {
  push esi
  push edi
  push ebx
  mov  eax,600h
  lea  ebx,cs:_pmkeybhandler
  xor  ecx,ecx
  mov  cx,bx
  shr  ebx,16
  xor  esi,esi
  mov  edi,HANDLER_CODESIZE
  int  31h
  jc   _endlock
  mov  eax,600h
  lea  ebx,ds:_hdata
  xor  ecx,ecx
  mov  cx,bx
  shr  ebx,16
  xor  esi,esi
  mov  edi,HANDLER_DATASIZE
  int  31h
  mov  al,1
  mov  islocked,al
  jnc  _endlock
  dec  al
  mov  islocked,al
  mov  eax,601h
  lea  ebx,cs:_pmkeybhandler
  xor  ecx,ecx
  mov  cx,bx
  shr  ebx,16
  xor  esi,esi
  mov  edi,HANDLER_CODESIZE
  int  31h
  _endlock:
  pop  ebx
  pop  edi
  pop  esi
  }
  if (islocked)
  {
    _hdata.prev_irq_01 = _dos_getvect( 0x09 );
    if (_hdata.prev_irq_01)
    {
      _dos_setvect( 0x09, _pmkeybhandler );
      _asm mov dl,1
      _asm mov ax,3301h
      _asm int 21h
    }
  }
}

static void _deinstallhandler(void)
{
  if (_hdata.prev_irq_01)
  {
    _dos_setvect( 0x09, _hdata.prev_irq_01 );
    _asm {
    push esi
    push edi
    push ebx
    mov  eax,601h
    lea  ebx,cs:_pmkeybhandler
    xor  ecx,ecx
    mov  cx,bx
    shr  ebx,16
    xor  esi,esi
    mov  edi,HANDLER_CODESIZE
    int  31h
    mov  eax,601h
    lea  ebx,ds:_hdata
    xor  ecx,ecx
    mov  cx,bx
    shr  ebx,16
    xor  esi,esi
    mov  edi,HANDLER_DATASIZE
    int  31h
    pop  ebx
    pop  edi
    pop  esi
    }
  }
}

#pragma pack(1)
struct ib_data { char resfield; char level; void (*proc)(void); };
#pragma pack()

#pragma data_seg ( "XIB" );
#pragma data_seg ( "XI" );
struct ib_data _cdoskeyb_irq01_init = { 0, 255, _installhandler };
#pragma data_seg ( "XIE" );
#pragma data_seg ( "YIB" );
#pragma data_seg ( "YI" );
struct ib_data _cdoskeyb_irq01_fini = { 0, 0, _deinstallhandler };
#pragma data_seg ( "YIE" );
#pragma data_seg ( "_DATA" );
  
#endif /* Watcom C only */
