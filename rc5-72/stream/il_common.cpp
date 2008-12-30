/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev
 * PanAm
 * Alexei Chupyatov
 *
 * $Id: il_common.cpp,v 1.3 2008/12/30 17:01:33 andreasb Exp $
*/

#include "il_common.h"

//Key increment
void key_incr(unsigned *hi, unsigned *mid, unsigned *lo, unsigned incr)
{
  _asm {
    mov esi,[hi]
    mov eax,[esi]
    add eax,incr
    mov byte ptr [esi],al
    shr eax,8
    mov esi,[mid]
    mov ebx,[esi]
    mov edi,[lo]
    mov edx,[edi]
    bswap ebx
    bswap edx
    add ebx,eax
    adc edx,0
    bswap ebx
    bswap edx
    mov     [esi],ebx
    mov     [edi],edx
  }
}

//Subtract two keys, use only mid & hi, since the result is always less than 40 bits
u32 sub72(u32 t_hi, u32 t_mid, u32 s_hi, u32 s_mid)
{
  u32 res;

  _asm {
    mov al,byte ptr [t_hi]
    sub al, byte ptr [s_hi]     ; hi
    mov edx,[t_mid]
    bswap edx
    mov edi,[s_mid]
    bswap edi
    sbb edx,edi
    shl edx,8
    mov dl,al
    mov res,edx
  }
  return res;
}

//Compare two keys
u32 cmp72(u32 o1h, u32 o1m, u32 o1l, u32 o2h, u32 o2m, u32 o2l)
{
  u32 _o1l,_o2l,_o1m,_o2m;
  _asm {
    mov eax,o1l
    mov ebx,o2l
    bswap eax
    bswap ebx
    mov _o1l,eax
    mov _o2l,ebx

    mov eax,o1m
    mov ebx,o2m
    bswap eax
    bswap ebx
    mov _o1m,eax
    mov _o2m,ebx
  }
  if(_o2l>_o1l)
    return 1;
  else
  if(_o2l==_o1l) {
    if(_o2m>_o1m)
      return 1;
    else
    if((_o2m==_o1m)&&(o2h>o1h))
      return 1;
  }
  return 0;
}
