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
 * $Id: r72stream-common.cpp,v 1.6 2009/02/19 09:38:26 andreasb Exp $
*/

#include "r72stream-common.h"

static u32 swap32(u32 a)
{
  u32 t=(a>>24)|(a<<24);
  t|=(a&0x00ff0000)>>8;
  t|=(a&0x0000ff00)<<8;
  return t;
}

//Key increment

void key_incr(u32 *hi, u32 *mid, u32 *lo, u32 incr)
{
  *hi+=incr;
  u32 ad=*hi>>8;
  if(*hi<incr)
    ad+=0x01000000;
  u32 t_m=swap32(*mid)+ad;
  u32 t_l=swap32(*lo);
  if(t_m<ad)
    t_l++;

  *hi=*hi&0xff;
  *mid=swap32(t_m);
  *lo=swap32(t_l);
}

//Subtract two keys, use only mid & hi, since the result is always less than 40 bits
u32 sub72(u32 t_hi, u32 t_mid, u32 s_hi, u32 s_mid)
{
  u32 res_h=t_hi-s_hi;
  u32 res_m=swap32(t_mid)-swap32(s_mid);
  if(res_h>t_hi)
    res_m--;
  res_m=(res_m<<8)+(res_h&0xff);
  return res_m;
}

//Compare two keys
u32 cmp72(u32 o1h, u32 o1m, u32 o1l, u32 o2h, u32 o2m, u32 o2l)
{
  u32 _o1l,_o2l,_o1m,_o2m;

  _o1l=swap32(o1l);
  _o2l=swap32(o2l);

  _o1m=swap32(o1m);
  _o2m=swap32(o2m);

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
