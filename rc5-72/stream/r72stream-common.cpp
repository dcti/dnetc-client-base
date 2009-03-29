/*
 * Copyright 2008 Vyacheslav Chupyatov <goteam@mail.ru>
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * Special thanks for help in testing this core to:
 * Alexander Kamashev, PanAm, Alexei Chupyatov
 *
 * $Id: r72stream-common.cpp,v 1.7 2009/03/29 20:02:27 andreasb Exp $
*/

#include "r72stream-common.h"

static inline u32 swap32(u32 a)
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
