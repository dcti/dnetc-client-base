; Copyright distributed.net 1997 - All Rights Reserved
; For use in distributed.net projects only.
; Any other distribution or use of this source violates copyright.
;
; MMX bitslice including CSC transformation tables
; based on 6bit-called core (GCC 2.95 generated): converted 1999/12/10 (cyp)
;
; WARNING: NASM has a 'bt' bug, it assembles 'bt edx,eax' to 'bt eax,edx'
;
%include "csc-mac.inc"

    global    csc_unit_func_6b_mmx
    global    _csc_unit_func_6b_mmx

    extern    convert_key_from_inc_to_csc, convert_key_from_csc_to_inc
    extern    csc_tabp,csc_bit_order

__DATASECT__
    db  "@(#)$Id: csc-mmx.asm,v 1.1.2.2 1999/12/12 07:27:00 jlawson Exp $",0

__CODESECT__

;__DATASECT__
csc_tabc_mmx:
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

csc_tabe_mmx:
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
;csc_transP2_data_1:
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff

;.new_section .rodata, "dr2"
;rodata: db    0xaa, 0xaa, 0xaa, 0xaa
;X$59: db    0xaa, 0xaa, 0xaa, 0xaa
;X$60: db    0xcc, 0xcc, 0xcc, 0xcc
;X$61: db    0xcc, 0xcc, 0xcc, 0xcc
;X$62: db    0xf0, 0xf0, 0xf0, 0xf0
;X$63: db    0xf0, 0xf0, 0xf0, 0xf0
;X$64: db    0x00, 0xff, 0x00, 0xff
;X$65: db    0x00, 0xff, 0x00, 0xff
;X$66: db    0x00, 0x00, 0xff, 0xff
;X$67: db    0x00, 0x00, 0xff, 0xff
;X$68: db    0x00, 0x00, 0x00, 0x00
;X$69: db    0xff, 0xff, 0xff, 0xff

csc_transP2_data_1:
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
    align 64

mmNOT:
    db  0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff 
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
    db  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00

    align 1<<8

csc_transP2:
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [csc_transP2_data_1]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [edx+0x48]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [csc_transP2_data_1]
    por       mm7,mm4
    mov       eax,dword ptr [edx+0x40]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [csc_transP2_data_1]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [edx+0x4c]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [csc_transP2_data_1]
    por       mm3,mm2
    mov       eax,dword ptr [edx+0x44]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [edx+0x50]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [edx+0x5c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [edx+0x58]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [edx+0x54]
    movq      qword ptr [eax],mm4
    ret       

    align 1<<8

cscipher_bitslicer_6b_mmx:
    sub       esp,0x000000dc
    push      ebp
    push      edi
    push      esi
    push      ebx
    mov       eax,dword ptr [esp+0x100]
    mov       dword ptr [esp+0xdc],eax
    add       eax,0x00000200
    mov       esi,eax
    add       esi,0x00001ac0
    mov       dword ptr [esp+0xc0],esi
    add       esi,0x00000060
    mov       dword ptr [esp+0x100],esi
    mov       esi,dword ptr [esp+0xf0]
    mov       dword ptr [esp+0xc8],eax
    mov       edx,eax
    add       edx,0x00001600
    mov       dword ptr [esp+0xc4],edx
    mov       ecx,eax
    add       ecx,0x000018c0
    mov       dword ptr [esp+0xd8],ecx
    mov       ebx,eax
    add       ebx,0x000019c0
    mov       dword ptr [esp+0xd4],ebx
    mov       edi,eax
    add       esi,0x00000200
    cld       
    mov       ecx,0x00000080
    repe movsd 
    mov       edi,dword ptr [esp+0xc8]
    mov       esi,dword ptr [esp+0xf0]
    add       edi,0x00000200
    cld       
    mov       ecx,0x00000080
    repe movsd 
    mov       ebx,dword ptr [esp+0xc8]
    mov       dword ptr [esp+0xb8],0x00000000
    mov       dword ptr [ebx+0x4c0],0x00000000
    mov       dword ptr [ebx+0x4c4],0x00000000
    mov       esi,dword ptr [esp+0xc8]
    mov       dword ptr [esi+0x500],0x00000000
    mov       dword ptr [esi+0x504],0x00000000
    mov       edi,dword ptr [esp+0xc8]
    mov       dword ptr [edi+0x540],0x00000000
    mov       dword ptr [edi+0x544],0x00000000
    mov       eax,dword ptr [esp+0xc8]
    mov       dword ptr [eax+0x580],0x00000000
    mov       dword ptr [eax+0x584],0x00000000
    mov       edx,dword ptr [esp+0xc8]
    mov       dword ptr [edx+0x5c0],0x00000000
    mov       dword ptr [edx+0x5c4],0x00000000
    mov       ecx,dword ptr [esp+0xc8]
    mov       dword ptr [ecx+0x400],0xffffffff
    mov       dword ptr [ecx+0x404],0xffffffff
    mov       ebx,dword ptr [esp+0xc8]
    mov       dword ptr [ebx+0x440],0xffffffff
    mov       dword ptr [ebx+0x444],0xffffffff
    mov       esi,dword ptr [esp+0xc8]
    mov       dword ptr [esi+0x480],0xffffffff
    mov       dword ptr [esi+0x484],0xffffffff
    mov       dword ptr [esp+0xbc],csc_tabc_mmx+0x40
    mov       ebx,dword ptr [esp+0xbc]
    mov       edx,dword ptr [esp+0xc0]
    mov       ebp,esi
    add       ebp,0x00000408
    add       esi,0x00000240
    mov       dword ptr [esp+0x24],esi
    mov       edi,ebp
    mov       eax,esi
    mov       esi,0x00000007
    mov       ebp,eax

    db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]

.loop0:
    movq      mm7,qword ptr [ebp+0x38]
    mov       dword ptr [edx+0x40],edi
    pxor      mm7,qword ptr [ebx+0x38]
    add       edi,0x00000040
    mov       dword ptr [edx+0x44],edi
    movq      mm6,qword ptr [ebp+0x30]
    add       edi,0x00000040
    pxor      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x48],edi
    movq      mm5,qword ptr [ebp+0x28]
    add       edi,0x00000040
    pxor      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x4c],edi
    movq      mm4,qword ptr [ebp+0x20]
    add       edi,0x00000040
    pxor      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],edi
    movq      mm3,qword ptr [ebp+0x18]
    add       edi,0x00000040
    pxor      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x54],edi
    movq      mm2,qword ptr [ebp+0x10]
    add       edi,0x00000040
    pxor      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x58],edi
    movq      mm1,qword ptr [ebp+0x8]
    add       edi,0x00000040
    pxor      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x5c],edi
    movq      mm0,qword ptr [ebp]
    sub       edi,0x000001c0
    pxor      mm0,qword ptr [ebx]
    call      near csc_transP2
    add       ebp,0x00000040
    add       ebx,0x00000040
    add       edi,0x00000008
    dec       esi
    jg        near .loop0
    mov       eax,ebp
    mov       eax,dword ptr [esp+0xc4]
    mov       edx,dword ptr [esp+0xc8]
    mov       ecx,dword ptr [esp+0xc8]
    mov       ebx,dword ptr [esp+0xc8]
    mov       esi,dword ptr [esp+0xd4]
    mov       dword ptr [esp+0xb4],0x00000002
    add       eax,0xffffff18
    mov       dword ptr [esp+0xa8],eax
    add       edx,0x00000200
    mov       dword ptr [esp+0x48],edx
    add       ecx,0x00000400
    mov       dword ptr [esp+0x40],ecx
    add       ebx,0x00000600
    mov       dword ptr [esp+0x38],ebx
    mov       dword ptr [esp+0x4c],esi

    db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
    db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi

X$1:
    mov       edx,dword ptr [esp+0xf4]
    mov       edi,dword ptr [esp+0xa8]
    add       edx,0x00000007
    sub       edx,dword ptr [esp+0xb4]
    mov       eax,0x00000007
    sub       eax,dword ptr [esp+0xb4]
    mov       ebx,dword ptr [esp+0xc8]
    mov       dl,byte ptr [edx]
    xor       dl,byte ptr [eax+csc_tabp]
    mov       byte ptr [esp+0xb3],dl
    xor       byte ptr [esp+0xb3],0x40
    movzx     eax,byte ptr [esp+0xb3]
    movzx     edx,dl
    mov       dl,byte ptr [edx+csc_tabp]
    xor       dl,byte ptr [eax+csc_tabp]
    mov       byte ptr [esp+0xb3],dl
    imul      edx,dword ptr [esp+0xb4],0x00000074
    mov       eax,dword ptr [esp+0xb4]
    lea       ecx,[edi+edx]
    shl       eax,0x00000006
    lea       eax,[ebx+eax+0x230]
    mov       dword ptr [ecx],eax
    mov       esi,dword ptr [esp+0xb4]
    mov       edi,dword ptr [esp+0xb4]
    mov       dword ptr [esp+0xac],0x00000001
    mov       dword ptr [esp+0xa4],0x00000000
    mov       dword ptr [esp+0x44],edx
    inc       esi
    mov       dword ptr [esp+0x3c],esi
    mov       eax,ebx
    add       eax,0x00000400
    lea       eax,[eax+edi*8]
    mov       dword ptr [esp+0x28],eax
    add       ecx,0x00000004
    mov       esi,ecx

    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;lea       edi,[0x0+edi*1]
    db 0x89, 0xf6           ; mov esi, esi

X$2:
    mov       edx,dword ptr [esp+0xa4]
    movzx     eax,byte ptr [esp+0xb3]
    db 0x0f, 0xa3, 0xd0  ; bt edx,eax (nasm bug reads that 'bt eax,edx')
    jae       near X$5
    mov       ebx,dword ptr [esp+0xb4]
    mov       eax,dword ptr [esp+0x28]
    lea       edi,[ebx+edx*8]
    mov       dword ptr [esi],eax
    add       esi,0x00000004
    mov       ebx,edi
    shr       ebx,0x00000004
    shl       ebx,0x00000006
    mov       ebp,ebx
    add       ebp,dword ptr [esp+0xd8]
    mov       eax,edi
    and       eax,0x00000007
    lea       eax,[ebp+eax*8]
    mov       dword ptr [esi],eax
    add       ecx,0x00000008
    add       esi,0x00000004
    add       dword ptr [esp+0xac],0x00000002
    mov       edx,edi
    and       edx,0x0000000f
    cmp       edx,0x00000007
    ja        X$3
    mov       eax,dword ptr [esp+0x4c]
    add       eax,ebx
    lea       eax,[eax+edx*8]
    mov       dword ptr [ecx],eax
    jmp       short X$4
    align 16  ;lea       esi,[esi]
X$3:
    mov       edx,dword ptr [esp+0x4c]
    add       edx,ebx
    lea       eax,[edi+0x1]
    and       eax,0x00000007
    shl       eax,0x00000003
    add       edx,eax
    mov       dword ptr [ecx],edx
    add       ecx,0x00000004
    add       esi,0x00000004
    inc       dword ptr [esp+0xac]
    test      edi,0x00000001
    je        X$5
    add       ebp,eax
    mov       dword ptr [ecx],ebp

    db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
    db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi

X$4:
    add       ecx,0x00000004
    add       esi,0x00000004
    inc       dword ptr [esp+0xac]

    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;lea       edi,[0x0+edi*1]
    db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi

X$5:
    add       dword ptr [esp+0x28],0x00000040
    inc       dword ptr [esp+0xa4]
    cmp       dword ptr [esp+0xa4],0x00000007
    jle       near X$2
    mov       edx,dword ptr [esp+0xac]
    mov       ecx,dword ptr [esp+0x44]
    mov       ebx,dword ptr [esp+0xa8]
    lea       eax,[ecx+edx*4]
    mov       dword ptr [ebx+eax],0x00000000
    mov       esi,dword ptr [esp+0x3c]
    mov       dword ptr [esp+0xb4],esi
    cmp       esi,0x00000007
    jle       near X$1
    mov       ebp,dword ptr [esp+0x40]
    mov       edi,dword ptr [esp+0xd8]
    mov       eax,dword ptr [esp+0xd4]
    mov       edx,dword ptr [esp+0xf8]
    mov       dword ptr [esp+0xa0],0x00000000
    mov       dword ptr [esp+0x34],edi
    mov       dword ptr [esp+0x30],eax
    mov       dword ptr [esp+0x2c],edx

    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;lea       edi,[0x0+edi*1]
    db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi

X$6:
    mov       edi,dword ptr [esp+0x2c]
    mov       ecx,edi
    mov       ebx,dword ptr [edi+0x40]
    mov       esi,dword ptr [edi+0x44]
    xor       ebx,dword ptr [ebp+0x40]
    xor       esi,dword ptr [ebp+0x44]
    mov       dword ptr [esp+0x98],ebx
    mov       dword ptr [esp+0x9c],esi
    mov       eax,dword ptr [ecx+0x48]
    mov       edx,dword ptr [ecx+0x4c]
    xor       eax,dword ptr [ebp+0x48]
    xor       edx,dword ptr [ebp+0x4c]
    mov       dword ptr [esp+0x90],eax
    mov       dword ptr [esp+0x94],edx
    mov       ebx,dword ptr [edi+0x50]
    mov       esi,dword ptr [edi+0x54]
    xor       ebx,dword ptr [ebp+0x50]
    xor       esi,dword ptr [ebp+0x54]
    mov       dword ptr [esp+0x88],ebx
    mov       dword ptr [esp+0x8c],esi
    mov       eax,dword ptr [ecx+0x58]
    mov       edx,dword ptr [ecx+0x5c]
    xor       eax,dword ptr [ebp+0x58]
    xor       edx,dword ptr [ebp+0x5c]
    mov       dword ptr [esp+0x80],eax
    mov       dword ptr [esp+0x84],edx
    mov       ebx,dword ptr [edi+0x60]
    mov       esi,dword ptr [edi+0x64]
    xor       ebx,dword ptr [ebp+0x60]
    xor       esi,dword ptr [ebp+0x64]
    mov       dword ptr [esp+0x78],ebx
    mov       dword ptr [esp+0x7c],esi
    mov       eax,dword ptr [ecx+0x68]
    mov       edx,dword ptr [ecx+0x6c]
    xor       eax,dword ptr [ebp+0x68]
    xor       edx,dword ptr [ebp+0x6c]
    mov       dword ptr [esp+0x1c],eax
    mov       dword ptr [esp+0x20],edx
    mov       ebx,dword ptr [edi+0x70]
    mov       esi,dword ptr [edi+0x74]
    xor       ebx,dword ptr [ebp+0x70]
    xor       esi,dword ptr [ebp+0x74]
    mov       dword ptr [esp+0x70],ebx
    mov       dword ptr [esp+0x74],esi
    mov       eax,dword ptr [ecx+0x78]
    mov       edx,dword ptr [ecx+0x7c]
    mov       ecx,dword ptr [esp+0xa0]
    mov       ebx,dword ptr [esp+0x34]
    shl       ecx,0x00000006
    lea       edi,[ebx+ecx]
    mov       ebx,dword ptr [esp+0x2c]
    xor       eax,dword ptr [ebp+0x78]
    xor       edx,dword ptr [ebp+0x7c]
    mov       dword ptr [esp+0x6c],edx
    mov       esi,dword ptr [esp+0x6c]
    mov       dword ptr [esp+0x68],eax
    mov       eax,dword ptr [ebx+0x38]
    mov       edx,dword ptr [ebx+0x3c]
    mov       ebx,dword ptr [esp+0x68]
    xor       eax,dword ptr [ebp+0x38]
    xor       edx,dword ptr [ebp+0x3c]
    mov       dword ptr [esp+0x50],eax
    mov       dword ptr [esp+0x54],edx
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x38],ebx
    mov       dword ptr [edi+0x3c],esi
    mov       eax,dword ptr [esp+0x1c]
    mov       esi,dword ptr [esp+0x2c]
    mov       edx,dword ptr [esp+0x20]
    add       ecx,dword ptr [esp+0x30]
    xor       eax,dword ptr [esi+0x30]
    xor       edx,dword ptr [esi+0x34]
    xor       eax,dword ptr [ebp+0x30]
    xor       edx,dword ptr [ebp+0x34]
    mov       dword ptr [ecx+0x30],eax
    mov       dword ptr [ecx+0x34],edx
    mov       ebx,dword ptr [esp+0x70]
    mov       esi,dword ptr [esp+0x74]
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x30],ebx
    mov       dword ptr [edi+0x34],esi
    mov       ebx,dword ptr [esp+0x2c]
    mov       esi,dword ptr [esp+0x20]
    mov       eax,dword ptr [ebx+0x28]
    mov       edx,dword ptr [ebx+0x2c]
    mov       ebx,dword ptr [esp+0x1c]
    xor       eax,dword ptr [ebp+0x28]
    xor       edx,dword ptr [ebp+0x2c]
    mov       dword ptr [esp+0x58],eax
    mov       dword ptr [esp+0x5c],edx
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x28],ebx
    mov       dword ptr [edi+0x2c],esi
    mov       esi,dword ptr [esp+0x2c]
    mov       eax,dword ptr [esp+0x80]
    mov       edx,dword ptr [esp+0x84]
    xor       eax,dword ptr [esi+0x20]
    xor       edx,dword ptr [esi+0x24]
    xor       eax,dword ptr [ebp+0x20]
    xor       edx,dword ptr [ebp+0x24]
    mov       dword ptr [ecx+0x20],eax
    mov       dword ptr [ecx+0x24],edx
    mov       ebx,dword ptr [esp+0x78]
    mov       esi,dword ptr [esp+0x7c]
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x20],ebx
    mov       dword ptr [edi+0x24],esi
    mov       ebx,dword ptr [esp+0x2c]
    mov       eax,dword ptr [ebx+0x18]
    mov       edx,dword ptr [ebx+0x1c]
    xor       eax,dword ptr [ebp+0x18]
    xor       edx,dword ptr [ebp+0x1c]
    mov       dword ptr [esp+0x60],eax
    mov       ebx,dword ptr [esp+0x80]
    mov       esi,dword ptr [esp+0x84]
    mov       dword ptr [esp+0x64],edx
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x18],ebx
    mov       dword ptr [edi+0x1c],esi
    mov       esi,dword ptr [esp+0x2c]
    mov       eax,dword ptr [esp+0x90]
    mov       edx,dword ptr [esp+0x94]
    xor       eax,dword ptr [esi+0x10]
    xor       edx,dword ptr [esi+0x14]
    xor       eax,dword ptr [ebp+0x10]
    xor       edx,dword ptr [ebp+0x14]
    mov       dword ptr [ecx+0x10],eax
    mov       dword ptr [ecx+0x14],edx
    mov       ebx,dword ptr [esp+0x88]
    mov       esi,dword ptr [esp+0x8c]
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x10],ebx
    mov       dword ptr [edi+0x14],esi
    mov       ebx,dword ptr [esp+0x2c]
    mov       esi,dword ptr [esp+0x94]
    mov       eax,dword ptr [ebx+0x8]
    mov       edx,dword ptr [ebx+0xc]
    mov       ebx,dword ptr [esp+0x90]
    xor       eax,dword ptr [ebp+0x8]
    xor       edx,dword ptr [ebp+0xc]
    mov       dword ptr [esp+0x1c],eax
    mov       dword ptr [esp+0x20],edx
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi+0x8],ebx
    mov       dword ptr [edi+0xc],esi
    mov       esi,dword ptr [esp+0x2c]
    mov       eax,dword ptr [esp+0x68]
    mov       edx,dword ptr [esp+0x6c]
    xor       eax,dword ptr [esi]
    xor       edx,dword ptr [esi+0x4]
    xor       eax,dword ptr [ebp]
    xor       edx,dword ptr [ebp+0x4]
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    mov       ebx,dword ptr [esp+0x98]
    mov       esi,dword ptr [esp+0x9c]
    xor       ebx,eax
    xor       esi,edx
    mov       dword ptr [edi],ebx
    mov       dword ptr [edi+0x4],esi
    mov       esi,dword ptr [esp+0x70]
    mov       edi,dword ptr [esp+0x74]
    xor       esi,dword ptr [esp+0x50]
    xor       edi,dword ptr [esp+0x54]
    mov       dword ptr [ecx+0x38],esi
    mov       dword ptr [ecx+0x3c],edi
    mov       eax,dword ptr [esp+0x78]
    mov       edx,dword ptr [esp+0x7c]
    xor       eax,dword ptr [esp+0x58]
    xor       edx,dword ptr [esp+0x5c]
    mov       dword ptr [ecx+0x28],eax
    mov       dword ptr [ecx+0x2c],edx
    mov       ebx,dword ptr [esp+0x88]
    mov       esi,dword ptr [esp+0x8c]
    xor       ebx,dword ptr [esp+0x60]
    xor       esi,dword ptr [esp+0x64]
    mov       dword ptr [ecx+0x18],ebx
    mov       dword ptr [ecx+0x1c],esi
    mov       esi,dword ptr [esp+0x98]
    mov       edi,dword ptr [esp+0x9c]
    xor       esi,dword ptr [esp+0x1c]
    xor       edi,dword ptr [esp+0x20]
    mov       dword ptr [ecx+0x8],esi
    mov       dword ptr [ecx+0xc],edi
    sub       ebp,0xffffff80
    sub       dword ptr [esp+0x2c],0xffffff80
    inc       dword ptr [esp+0xa0]
    cmp       dword ptr [esp+0xa0],0x00000003
    jle       near X$6

    db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi

X$7:
    mov       edi,dword ptr [esp+0xdc]
    mov       esi,dword ptr [esp+0xf8]
    cld       
    mov       ecx,0x00000080
    repe movsd 
    mov       edx,dword ptr [esp+0xc0]
    mov       ebx,dword ptr [esp+0xd8]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x00000000
    add       eax,0x00000040
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd4]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x00000000
    add       eax,0x00000000
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd8]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x00000040
    add       eax,0x000000c0
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd4]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x00000040
    add       eax,0x00000080
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd8]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x00000080
    add       eax,0x00000140
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd4]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x00000080
    add       eax,0x00000100
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd8]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x000000c0
    add       eax,0x000001c0
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       ebx,dword ptr [esp+0xd4]
    mov       eax,dword ptr [esp+0xdc]
    add       ebx,0x000000c0
    add       eax,0x00000180
    movq      mm0,qword ptr [ebx]
    mov       dword ptr [edx+0x40],eax
    add       eax,0x00000008
    movq      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x44],eax
    add       eax,0x00000008
    movq      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x48],eax
    add       eax,0x00000008
    movq      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x4c],eax
    add       eax,0x00000008
    movq      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],eax
    add       eax,0x00000008
    movq      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x54],eax
    add       eax,0x00000008
    movq      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x58],eax
    add       eax,0x00000008
    movq      mm7,qword ptr [ebx+0x38]
    mov       dword ptr [edx+0x5c],eax
    call      near csc_transP2
    mov       eax,dword ptr [esp+0xdc]
    mov       edi,dword ptr [esp+0x100]
    mov       ebx,csc_tabe_mmx
    movq      mm7,qword ptr [eax+0x88]
    movq      mm1,qword ptr [eax+0x8]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x80]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x10]
    movq      mm2,qword ptr [eax+0x90]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x98]
    movq      mm3,qword ptr [eax+0x18]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x20]
    movq      mm4,qword ptr [eax+0xa0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0xa8]
    movq      mm5,qword ptr [eax+0x28]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x30]
    movq      mm6,qword ptr [eax+0xb0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0xb8]
    movq      mm7,qword ptr [eax+0x38]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax]
    movq      mm7,qword ptr [eax+0x80]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x188]
    movq      mm1,qword ptr [eax+0x108]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x180]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x110]
    movq      mm2,qword ptr [eax+0x190]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x198]
    movq      mm3,qword ptr [eax+0x118]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x120]
    movq      mm4,qword ptr [eax+0x1a0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1a8]
    movq      mm5,qword ptr [eax+0x128]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x130]
    movq      mm6,qword ptr [eax+0x1b0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1b8]
    movq      mm7,qword ptr [eax+0x138]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x100]
    movq      mm7,qword ptr [eax+0x180]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x100]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0xc8]
    movq      mm1,qword ptr [eax+0x48]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0xc0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x50]
    movq      mm2,qword ptr [eax+0xd0]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0xd8]
    movq      mm3,qword ptr [eax+0x58]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x60]
    movq      mm4,qword ptr [eax+0xe0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0xe8]
    movq      mm5,qword ptr [eax+0x68]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x70]
    movq      mm6,qword ptr [eax+0xf0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0xf8]
    movq      mm7,qword ptr [eax+0x78]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x40]
    movq      mm7,qword ptr [eax+0xc0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x40]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x1c8]
    movq      mm1,qword ptr [eax+0x148]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x1c0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x150]
    movq      mm2,qword ptr [eax+0x1d0]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x1d8]
    movq      mm3,qword ptr [eax+0x158]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x160]
    movq      mm4,qword ptr [eax+0x1e0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1e8]
    movq      mm5,qword ptr [eax+0x168]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x170]
    movq      mm6,qword ptr [eax+0x1f0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1f8]
    movq      mm7,qword ptr [eax+0x178]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x140]
    movq      mm7,qword ptr [eax+0x1c0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x140]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x108]
    movq      mm1,qword ptr [eax+0x8]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x100]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x10]
    movq      mm2,qword ptr [eax+0x110]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x118]
    movq      mm3,qword ptr [eax+0x18]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x20]
    movq      mm4,qword ptr [eax+0x120]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x128]
    movq      mm5,qword ptr [eax+0x28]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x30]
    movq      mm6,qword ptr [eax+0x130]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x138]
    movq      mm7,qword ptr [eax+0x38]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax]
    movq      mm7,qword ptr [eax+0x100]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x148]
    movq      mm1,qword ptr [eax+0x48]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x140]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x50]
    movq      mm2,qword ptr [eax+0x150]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x158]
    movq      mm3,qword ptr [eax+0x58]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x60]
    movq      mm4,qword ptr [eax+0x160]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x168]
    movq      mm5,qword ptr [eax+0x68]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x70]
    movq      mm6,qword ptr [eax+0x170]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x178]
    movq      mm7,qword ptr [eax+0x78]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x40]
    movq      mm7,qword ptr [eax+0x140]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x40]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x188]
    movq      mm1,qword ptr [eax+0x88]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x180]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x90]
    movq      mm2,qword ptr [eax+0x190]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x198]
    movq      mm3,qword ptr [eax+0x98]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xa0]
    movq      mm4,qword ptr [eax+0x1a0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1a8]
    movq      mm5,qword ptr [eax+0xa8]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xb0]
    movq      mm6,qword ptr [eax+0x1b0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1b8]
    movq      mm7,qword ptr [eax+0xb8]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x80]
    movq      mm7,qword ptr [eax+0x180]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x80]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x1c8]
    movq      mm1,qword ptr [eax+0xc8]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x1c0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xd0]
    movq      mm2,qword ptr [eax+0x1d0]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x1d8]
    movq      mm3,qword ptr [eax+0xd8]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xe0]
    movq      mm4,qword ptr [eax+0x1e0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1e8]
    movq      mm5,qword ptr [eax+0xe8]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xf0]
    movq      mm6,qword ptr [eax+0x1f0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1f8]
    movq      mm7,qword ptr [eax+0xf8]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0xc0]
    movq      mm7,qword ptr [eax+0x1c0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0xc0]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    mov       ebp,dword ptr [esp+0x38]
    mov       edx,dword ptr [esp+0x40]
    mov       dword ptr [esp+0x24],edx
    mov       dword ptr [esp+0xbc],csc_tabc_mmx+0x200
    mov       dword ptr [esp+0x18],0x00000007
    mov       dword ptr [esp+0x14],eax

    db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi 
    nop       

X$8:
    mov       eax,dword ptr [esp+0x24]
    mov       ebx,dword ptr [esp+0xbc]
    mov       edx,dword ptr [esp+0xc0]
    mov       edi,ebp
    mov       ecx,0x00000008

    db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]

.loop:
    movq      mm7,qword ptr [eax+0x38]
    mov       dword ptr [edx+0x40],edi
    pxor      mm7,qword ptr [ebx+0x38]
    add       edi,0x00000040
    mov       dword ptr [edx+0x44],edi
    movq      mm6,qword ptr [eax+0x30]
    add       edi,0x00000040
    pxor      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x48],edi
    movq      mm5,qword ptr [eax+0x28]
    add       edi,0x00000040
    pxor      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x4c],edi
    movq      mm4,qword ptr [eax+0x20]
    add       edi,0x00000040
    pxor      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],edi
    movq      mm3,qword ptr [eax+0x18]
    add       edi,0x00000040
    pxor      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x54],edi
    movq      mm2,qword ptr [eax+0x10]
    add       edi,0x00000040
    pxor      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x58],edi
    movq      mm1,qword ptr [eax+0x8]
    add       edi,0x00000040
    pxor      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x5c],edi
    movq      mm0,qword ptr [eax]
    sub       edi,0x000001c0
    pxor      mm0,qword ptr [ebx]
    push      eax
    lea       edx,[edx]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [edx+0x48]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [edx+0x40]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [edx+0x4c]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [edx+0x44]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [edx+0x50]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [edx+0x5c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [edx+0x58]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [edx+0x54]
    movq      qword ptr [eax],mm4
    pop       eax
    add       eax,0x00000040
    add       ebx,0x00000040
    add       edi,0x00000008
    dec       ecx
    jg        near .loop
    sub       edi,0x00000040
    mov       dword ptr [esp+0x24],eax
    mov       dword ptr [esp+0xbc],ebx
    mov       ebx,edi
    mov       edi,dword ptr [esp+0x100]
    mov       eax,dword ptr [esp+0x14]
    movq      mm0,qword ptr [ebx+0x20]
    movq      mm1,qword ptr [ebx+0x60]
    pxor      mm0,qword ptr [ebx-0x3e0]
    pxor      mm1,qword ptr [ebx-0x3a0]
    movq      qword ptr [ebx+0x20],mm0
    movq      qword ptr [ebx+0x60],mm1
    movq      mm2,qword ptr [ebx+0x28]
    movq      mm3,qword ptr [ebx+0x68]
    pxor      mm2,qword ptr [ebx-0x3d8]
    pxor      mm3,qword ptr [ebx-0x398]
    movq      qword ptr [ebx+0x28],mm2
    movq      qword ptr [ebx+0x68],mm3
    movq      mm4,qword ptr [ebx+0x30]
    movq      mm5,qword ptr [ebx+0x70]
    pxor      mm4,qword ptr [ebx-0x3d0]
    pxor      mm5,qword ptr [ebx-0x390]
    movq      qword ptr [ebx+0x30],mm4
    movq      qword ptr [ebx+0x70],mm5
    movq      mm0,qword ptr [ebx]
    movq      mm1,qword ptr [ebx+0x40]
    pxor      mm0,qword ptr [ebx-0x400]
    pxor      mm1,qword ptr [ebx-0x3c0]
    movq      qword ptr [ebx],mm0
    movq      qword ptr [ebx+0x40],mm1
    movq      mm6,qword ptr [ebx+0x38]
    movq      mm7,qword ptr [ebx+0x78]
    pxor      mm6,qword ptr [ebx-0x3c8]
    pxor      mm7,qword ptr [ebx-0x388]
    movq      qword ptr [ebx+0x38],mm6
    movq      qword ptr [ebx+0x78],mm7
    movq      mm0,qword ptr [ebx+0x8]
    movq      mm2,qword ptr [ebx+0x48]
    pxor      mm0,qword ptr [ebx-0x3f8]
    pxor      mm2,qword ptr [ebx-0x3b8]
    movq      qword ptr [ebx+0x8],mm0
    movq      qword ptr [ebx+0x48],mm2
    movq      mm3,qword ptr [ebx+0x10]
    movq      mm4,qword ptr [ebx+0x50]
    pxor      mm3,qword ptr [ebx-0x3f0]
    pxor      mm4,qword ptr [ebx-0x3b0]
    movq      qword ptr [ebx+0x10],mm3
    movq      qword ptr [ebx+0x50],mm4
    movq      mm5,qword ptr [ebx+0x18]
    movq      mm6,qword ptr [ebx+0x58]
    pxor      mm5,qword ptr [ebx-0x3e8]
    pxor      mm6,qword ptr [ebx-0x3a8]
    movq      qword ptr [ebx+0x18],mm5
    movq      qword ptr [ebx+0x58],mm6
    movq      mm7,qword ptr [eax+0x48]
    movq      mm1,qword ptr [eax+0x8]
    pxor      mm7,mm2
    pxor      mm1,mm0
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x40]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x10]
    movq      mm2,qword ptr [eax+0x50]
    pxor      mm7,mm3
    pxor      mm2,mm4
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x58]
    movq      mm3,qword ptr [eax+0x18]
    pxor      mm7,mm6
    pxor      mm3,mm5
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x20]
    movq      mm4,qword ptr [eax+0x60]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x68]
    movq      mm5,qword ptr [eax+0x28]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x30]
    movq      mm6,qword ptr [eax+0x70]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x78]
    movq      mm7,qword ptr [eax+0x38]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax]
    movq      mm7,qword ptr [eax+0x40]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    lea       esi,[eax]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm0,qword ptr [ebx+0x20]
    movq      mm1,qword ptr [ebx+0x60]
    pxor      mm0,qword ptr [ebx-0x3e0]
    pxor      mm1,qword ptr [ebx-0x3a0]
    movq      qword ptr [ebx+0x20],mm0
    movq      qword ptr [ebx+0x60],mm1
    movq      mm2,qword ptr [ebx+0x28]
    movq      mm3,qword ptr [ebx+0x68]
    pxor      mm2,qword ptr [ebx-0x3d8]
    pxor      mm3,qword ptr [ebx-0x398]
    movq      qword ptr [ebx+0x28],mm2
    movq      qword ptr [ebx+0x68],mm3
    movq      mm4,qword ptr [ebx+0x30]
    movq      mm5,qword ptr [ebx+0x70]
    pxor      mm4,qword ptr [ebx-0x3d0]
    pxor      mm5,qword ptr [ebx-0x390]
    movq      qword ptr [ebx+0x30],mm4
    movq      qword ptr [ebx+0x70],mm5
    movq      mm0,qword ptr [ebx]
    movq      mm1,qword ptr [ebx+0x40]
    pxor      mm0,qword ptr [ebx-0x400]
    pxor      mm1,qword ptr [ebx-0x3c0]
    movq      qword ptr [ebx],mm0
    movq      qword ptr [ebx+0x40],mm1
    movq      mm6,qword ptr [ebx+0x38]
    movq      mm7,qword ptr [ebx+0x78]
    pxor      mm6,qword ptr [ebx-0x3c8]
    pxor      mm7,qword ptr [ebx-0x388]
    movq      qword ptr [ebx+0x38],mm6
    movq      qword ptr [ebx+0x78],mm7
    movq      mm0,qword ptr [ebx+0x8]
    movq      mm2,qword ptr [ebx+0x48]
    pxor      mm0,qword ptr [ebx-0x3f8]
    pxor      mm2,qword ptr [ebx-0x3b8]
    movq      qword ptr [ebx+0x8],mm0
    movq      qword ptr [ebx+0x48],mm2
    movq      mm3,qword ptr [ebx+0x10]
    movq      mm4,qword ptr [ebx+0x50]
    pxor      mm3,qword ptr [ebx-0x3f0]
    pxor      mm4,qword ptr [ebx-0x3b0]
    movq      qword ptr [ebx+0x10],mm3
    movq      qword ptr [ebx+0x50],mm4
    movq      mm5,qword ptr [ebx+0x18]
    movq      mm6,qword ptr [ebx+0x58]
    pxor      mm5,qword ptr [ebx-0x3e8]
    pxor      mm6,qword ptr [ebx-0x3a8]
    movq      qword ptr [ebx+0x18],mm5
    movq      qword ptr [ebx+0x58],mm6
    movq      mm7,qword ptr [eax+0xc8]
    movq      mm1,qword ptr [eax+0x88]
    pxor      mm7,mm2
    pxor      mm1,mm0
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0xc0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x90]
    movq      mm2,qword ptr [eax+0xd0]
    pxor      mm7,mm3
    pxor      mm2,mm4
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0xd8]
    movq      mm3,qword ptr [eax+0x98]
    pxor      mm7,mm6
    pxor      mm3,mm5
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xa0]
    movq      mm4,qword ptr [eax+0xe0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0xe8]
    movq      mm5,qword ptr [eax+0xa8]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xb0]
    movq      mm6,qword ptr [eax+0xf0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0xf8]
    movq      mm7,qword ptr [eax+0xb8]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x80]
    movq      mm7,qword ptr [eax+0xc0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    lea       esi,[eax+0x80]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm0,qword ptr [ebx+0x20]
    movq      mm1,qword ptr [ebx+0x60]
    pxor      mm0,qword ptr [ebx-0x3e0]
    pxor      mm1,qword ptr [ebx-0x3a0]
    movq      qword ptr [ebx+0x20],mm0
    movq      qword ptr [ebx+0x60],mm1
    movq      mm2,qword ptr [ebx+0x28]
    movq      mm3,qword ptr [ebx+0x68]
    pxor      mm2,qword ptr [ebx-0x3d8]
    pxor      mm3,qword ptr [ebx-0x398]
    movq      qword ptr [ebx+0x28],mm2
    movq      qword ptr [ebx+0x68],mm3
    movq      mm4,qword ptr [ebx+0x30]
    movq      mm5,qword ptr [ebx+0x70]
    pxor      mm4,qword ptr [ebx-0x3d0]
    pxor      mm5,qword ptr [ebx-0x390]
    movq      qword ptr [ebx+0x30],mm4
    movq      qword ptr [ebx+0x70],mm5
    movq      mm0,qword ptr [ebx]
    movq      mm1,qword ptr [ebx+0x40]
    pxor      mm0,qword ptr [ebx-0x400]
    pxor      mm1,qword ptr [ebx-0x3c0]
    movq      qword ptr [ebx],mm0
    movq      qword ptr [ebx+0x40],mm1
    movq      mm6,qword ptr [ebx+0x38]
    movq      mm7,qword ptr [ebx+0x78]
    pxor      mm6,qword ptr [ebx-0x3c8]
    pxor      mm7,qword ptr [ebx-0x388]
    movq      qword ptr [ebx+0x38],mm6
    movq      qword ptr [ebx+0x78],mm7
    movq      mm0,qword ptr [ebx+0x8]
    movq      mm2,qword ptr [ebx+0x48]
    pxor      mm0,qword ptr [ebx-0x3f8]
    pxor      mm2,qword ptr [ebx-0x3b8]
    movq      qword ptr [ebx+0x8],mm0
    movq      qword ptr [ebx+0x48],mm2
    movq      mm3,qword ptr [ebx+0x10]
    movq      mm4,qword ptr [ebx+0x50]
    pxor      mm3,qword ptr [ebx-0x3f0]
    pxor      mm4,qword ptr [ebx-0x3b0]
    movq      qword ptr [ebx+0x10],mm3
    movq      qword ptr [ebx+0x50],mm4
    movq      mm5,qword ptr [ebx+0x18]
    movq      mm6,qword ptr [ebx+0x58]
    pxor      mm5,qword ptr [ebx-0x3e8]
    pxor      mm6,qword ptr [ebx-0x3a8]
    movq      qword ptr [ebx+0x18],mm5
    movq      qword ptr [ebx+0x58],mm6
    movq      mm7,qword ptr [eax+0x148]
    movq      mm1,qword ptr [eax+0x108]
    pxor      mm7,mm2
    pxor      mm1,mm0
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x140]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x110]
    movq      mm2,qword ptr [eax+0x150]
    pxor      mm7,mm3
    pxor      mm2,mm4
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x158]
    movq      mm3,qword ptr [eax+0x118]
    pxor      mm7,mm6
    pxor      mm3,mm5
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x120]
    movq      mm4,qword ptr [eax+0x160]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x168]
    movq      mm5,qword ptr [eax+0x128]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x130]
    movq      mm6,qword ptr [eax+0x170]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x178]
    movq      mm7,qword ptr [eax+0x138]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x100]
    movq      mm7,qword ptr [eax+0x140]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    lea       esi,[eax+0x100]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm0,qword ptr [ebx+0x20]
    movq      mm1,qword ptr [ebx+0x60]
    pxor      mm0,qword ptr [ebx-0x3e0]
    pxor      mm1,qword ptr [ebx-0x3a0]
    movq      qword ptr [ebx+0x20],mm0
    movq      qword ptr [ebx+0x60],mm1
    movq      mm2,qword ptr [ebx+0x28]
    movq      mm3,qword ptr [ebx+0x68]
    pxor      mm2,qword ptr [ebx-0x3d8]
    pxor      mm3,qword ptr [ebx-0x398]
    movq      qword ptr [ebx+0x28],mm2
    movq      qword ptr [ebx+0x68],mm3
    movq      mm4,qword ptr [ebx+0x30]
    movq      mm5,qword ptr [ebx+0x70]
    pxor      mm4,qword ptr [ebx-0x3d0]
    pxor      mm5,qword ptr [ebx-0x390]
    movq      qword ptr [ebx+0x30],mm4
    movq      qword ptr [ebx+0x70],mm5
    movq      mm0,qword ptr [ebx]
    movq      mm1,qword ptr [ebx+0x40]
    pxor      mm0,qword ptr [ebx-0x400]
    pxor      mm1,qword ptr [ebx-0x3c0]
    movq      qword ptr [ebx],mm0
    movq      qword ptr [ebx+0x40],mm1
    movq      mm6,qword ptr [ebx+0x38]
    movq      mm7,qword ptr [ebx+0x78]
    pxor      mm6,qword ptr [ebx-0x3c8]
    pxor      mm7,qword ptr [ebx-0x388]
    movq      qword ptr [ebx+0x38],mm6
    movq      qword ptr [ebx+0x78],mm7
    movq      mm0,qword ptr [ebx+0x8]
    movq      mm2,qword ptr [ebx+0x48]
    pxor      mm0,qword ptr [ebx-0x3f8]
    pxor      mm2,qword ptr [ebx-0x3b8]
    movq      qword ptr [ebx+0x8],mm0
    movq      qword ptr [ebx+0x48],mm2
    movq      mm3,qword ptr [ebx+0x10]
    movq      mm4,qword ptr [ebx+0x50]
    pxor      mm3,qword ptr [ebx-0x3f0]
    pxor      mm4,qword ptr [ebx-0x3b0]
    movq      qword ptr [ebx+0x10],mm3
    movq      qword ptr [ebx+0x50],mm4
    movq      mm5,qword ptr [ebx+0x18]
    movq      mm6,qword ptr [ebx+0x58]
    pxor      mm5,qword ptr [ebx-0x3e8]
    pxor      mm6,qword ptr [ebx-0x3a8]
    movq      qword ptr [ebx+0x18],mm5
    movq      qword ptr [ebx+0x58],mm6
    movq      mm7,qword ptr [eax+0x1c8]
    movq      mm1,qword ptr [eax+0x188]
    pxor      mm7,mm2
    pxor      mm1,mm0
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x1c0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x190]
    movq      mm2,qword ptr [eax+0x1d0]
    pxor      mm7,mm3
    pxor      mm2,mm4
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x1d8]
    movq      mm3,qword ptr [eax+0x198]
    pxor      mm7,mm6
    pxor      mm3,mm5
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x1a0]
    movq      mm4,qword ptr [eax+0x1e0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1e8]
    movq      mm5,qword ptr [eax+0x1a8]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x1b0]
    movq      mm6,qword ptr [eax+0x1f0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1f8]
    movq      mm7,qword ptr [eax+0x1b8]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x180]
    movq      mm7,qword ptr [eax+0x1c0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    lea       esi,[eax+0x180]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    mov       ebp,ebx
    mov       ebx,csc_tabe_mmx
    movq      mm7,qword ptr [eax+0x88]
    movq      mm1,qword ptr [eax+0x8]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x80]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x10]
    movq      mm2,qword ptr [eax+0x90]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x98]
    movq      mm3,qword ptr [eax+0x18]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x20]
    movq      mm4,qword ptr [eax+0xa0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0xa8]
    movq      mm5,qword ptr [eax+0x28]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x30]
    movq      mm6,qword ptr [eax+0xb0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0xb8]
    movq      mm7,qword ptr [eax+0x38]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax]
    movq      mm7,qword ptr [eax+0x80]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x188]
    movq      mm1,qword ptr [eax+0x108]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x180]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x110]
    movq      mm2,qword ptr [eax+0x190]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x198]
    movq      mm3,qword ptr [eax+0x118]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x120]
    movq      mm4,qword ptr [eax+0x1a0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1a8]
    movq      mm5,qword ptr [eax+0x128]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x130]
    movq      mm6,qword ptr [eax+0x1b0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1b8]
    movq      mm7,qword ptr [eax+0x138]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x100]
    movq      mm7,qword ptr [eax+0x180]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x100]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0xc8]
    movq      mm1,qword ptr [eax+0x48]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0xc0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x50]
    movq      mm2,qword ptr [eax+0xd0]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0xd8]
    movq      mm3,qword ptr [eax+0x58]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x60]
    movq      mm4,qword ptr [eax+0xe0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0xe8]
    movq      mm5,qword ptr [eax+0x68]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x70]
    movq      mm6,qword ptr [eax+0xf0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0xf8]
    movq      mm7,qword ptr [eax+0x78]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x40]
    movq      mm7,qword ptr [eax+0xc0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x40]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x1c8]
    movq      mm1,qword ptr [eax+0x148]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x1c0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x150]
    movq      mm2,qword ptr [eax+0x1d0]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x1d8]
    movq      mm3,qword ptr [eax+0x158]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x160]
    movq      mm4,qword ptr [eax+0x1e0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1e8]
    movq      mm5,qword ptr [eax+0x168]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x170]
    movq      mm6,qword ptr [eax+0x1f0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1f8]
    movq      mm7,qword ptr [eax+0x178]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x140]
    movq      mm7,qword ptr [eax+0x1c0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x140]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x108]
    movq      mm1,qword ptr [eax+0x8]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x100]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x10]
    movq      mm2,qword ptr [eax+0x110]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x118]
    movq      mm3,qword ptr [eax+0x18]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x20]
    movq      mm4,qword ptr [eax+0x120]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x128]
    movq      mm5,qword ptr [eax+0x28]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x30]
    movq      mm6,qword ptr [eax+0x130]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x138]
    movq      mm7,qword ptr [eax+0x38]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax]
    movq      mm7,qword ptr [eax+0x100]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x148]
    movq      mm1,qword ptr [eax+0x48]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x140]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x50]
    movq      mm2,qword ptr [eax+0x150]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x158]
    movq      mm3,qword ptr [eax+0x58]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x60]
    movq      mm4,qword ptr [eax+0x160]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x168]
    movq      mm5,qword ptr [eax+0x68]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x70]
    movq      mm6,qword ptr [eax+0x170]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x178]
    movq      mm7,qword ptr [eax+0x78]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x40]
    movq      mm7,qword ptr [eax+0x140]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x40]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x188]
    movq      mm1,qword ptr [eax+0x88]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x180]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0x90]
    movq      mm2,qword ptr [eax+0x190]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x198]
    movq      mm3,qword ptr [eax+0x98]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xa0]
    movq      mm4,qword ptr [eax+0x1a0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1a8]
    movq      mm5,qword ptr [eax+0xa8]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xb0]
    movq      mm6,qword ptr [eax+0x1b0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1b8]
    movq      mm7,qword ptr [eax+0xb8]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0x80]
    movq      mm7,qword ptr [eax+0x180]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0x80]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    movq      mm7,qword ptr [eax+0x1c8]
    movq      mm1,qword ptr [eax+0xc8]
    pxor      mm7,qword ptr [ebx+0x48]
    pxor      mm1,qword ptr [ebx+0x8]
    movq      qword ptr [edi+0x40],mm1
    lea       esi,[eax+0x1c0]
    pxor      mm1,mm7
    mov       dword ptr [edx+0x40],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xd0]
    movq      mm2,qword ptr [eax+0x1d0]
    pxor      mm7,qword ptr [ebx+0x10]
    pxor      mm2,qword ptr [ebx+0x50]
    movq      qword ptr [edi+0x8],mm7
    movq      qword ptr [edi+0x10],mm2
    mov       dword ptr [edx+0x44],esi
    pxor      mm2,mm7
    movq      mm7,qword ptr [eax+0x1d8]
    movq      mm3,qword ptr [eax+0xd8]
    pxor      mm7,qword ptr [ebx+0x58]
    pxor      mm3,qword ptr [ebx+0x18]
    movq      qword ptr [edi+0x48],mm3
    add       esi,0x00000008
    pxor      mm3,mm7
    mov       dword ptr [edx+0x48],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xe0]
    movq      mm4,qword ptr [eax+0x1e0]
    pxor      mm7,qword ptr [ebx+0x20]
    pxor      mm4,qword ptr [ebx+0x60]
    movq      qword ptr [edi+0x18],mm7
    movq      qword ptr [edi+0x20],mm4
    mov       dword ptr [edx+0x4c],esi
    pxor      mm4,mm7
    movq      mm7,qword ptr [eax+0x1e8]
    movq      mm5,qword ptr [eax+0xe8]
    pxor      mm7,qword ptr [ebx+0x68]
    pxor      mm5,qword ptr [ebx+0x28]
    movq      qword ptr [edi+0x50],mm5
    add       esi,0x00000008
    pxor      mm5,mm7
    mov       dword ptr [edx+0x50],esi
    add       esi,0x00000008
    pxor      mm7,qword ptr [eax+0xf0]
    movq      mm6,qword ptr [eax+0x1f0]
    pxor      mm7,qword ptr [ebx+0x30]
    pxor      mm6,qword ptr [ebx+0x70]
    movq      qword ptr [edi+0x28],mm7
    movq      qword ptr [edi+0x30],mm6
    mov       dword ptr [edx+0x54],esi
    pxor      mm6,mm7
    movq      mm0,qword ptr [eax+0x1f8]
    movq      mm7,qword ptr [eax+0xf8]
    pxor      mm0,qword ptr [ebx+0x78]
    pxor      mm7,qword ptr [ebx+0x38]
    movq      qword ptr [edi+0x58],mm7
    pxor      mm7,mm0
    movq      qword ptr [edi+0x60],mm7
    add       esi,0x00000008
    mov       dword ptr [edx+0x58],esi
    add       esi,0x00000008
    pxor      mm0,qword ptr [eax+0xc0]
    movq      mm7,qword ptr [eax+0x1c0]
    pxor      mm0,qword ptr [ebx]
    pxor      mm7,qword ptr [ebx+0x40]
    movq      qword ptr [edi+0x38],mm0
    movq      qword ptr [edi],mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edi+0x60]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    lea       ecx,[edx+0x40]
    movq      qword ptr [edx],mm0
    movq      qword ptr [edx+0x38],mm7
    movq      mm7,mm3
    movq      qword ptr [edx+0x18],mm3
    pxor      mm3,qword ptr [mmNOT]
    movq      qword ptr [edx+0x30],mm6
    movq      mm6,mm3
    por       mm7,mm2
    por       mm3,mm0
    movq      qword ptr [edx+0x28],mm5
    pxor      mm7,mm6
    pxor      mm4,mm3
    movq      mm5,mm7
    pxor      mm5,mm1
    pxor      mm7,qword ptr [edx+0x38]
    por       mm5,mm2
    movq      mm6,mm5
    pxor      mm6,qword ptr [edx]
    pxor      mm5,qword ptr [edx+0x30]
    por       mm6,mm1
    pxor      mm6,qword ptr [edx+0x28]
    movq      qword ptr [edx+0x28],mm6
    pand      mm6,mm4
    movq      qword ptr [edx+0x38],mm7
    movq      mm3,mm7
    por       mm7,mm4
    movq      qword ptr [edx+0x30],mm5
    pxor      mm7,mm6
    por       mm5,mm4
    pxor      mm2,mm7
    pxor      mm5,mm3
    movq      mm3,mm7
    mov       eax,dword ptr [ecx+0x8]
    movq      qword ptr [eax],mm2
    movq      mm6,qword ptr [edx+0x30]
    por       mm7,mm5
    pand      mm6,mm4
    pxor      mm7,mm4
    pxor      mm5,mm7
    pxor      mm0,mm7
    movq      mm7,qword ptr [edx+0x28]
    pxor      mm5,qword ptr [mmNOT]
    por       mm7,mm4
    mov       eax,dword ptr [ecx]
    movq      qword ptr [eax],mm0
    pxor      mm7,mm6
    movq      mm6,mm7
    por       mm7,mm3
    pxor      mm6,qword ptr [mmNOT]
    pxor      mm5,mm7
    pxor      mm6,qword ptr [edx+0x18]
    pxor      mm1,mm5
    mov       eax,dword ptr [ecx+0xc]
    movq      mm3,mm6
    movq      qword ptr [eax],mm6
    pxor      mm6,qword ptr [mmNOT]
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x4]
    movq      mm7,mm6
    movq      qword ptr [eax],mm1
    por       mm6,mm0
    pxor      mm3,mm7
    movq      mm7,qword ptr [edx+0x38]
    pxor      mm6,mm4
    mov       eax,dword ptr [ecx+0x10]
    movq      mm4,mm3
    movq      mm5,qword ptr [edx+0x30]
    pxor      mm3,mm1
    movq      qword ptr [eax],mm6
    pxor      mm7,mm4
    por       mm3,mm2
    mov       eax,dword ptr [ecx+0x1c]
    movq      qword ptr [eax],mm7
    movq      mm4,mm3
    movq      mm2,qword ptr [edx+0x28]
    pxor      mm4,mm0
    pxor      mm3,mm5
    por       mm4,mm1
    mov       eax,dword ptr [ecx+0x18]
    movq      qword ptr [eax],mm3
    pxor      mm4,mm2
    mov       eax,dword ptr [ecx+0x14]
    movq      qword ptr [eax],mm4
    pop       eax
    lea       esi,[eax+0xc0]
    movq      mm7,qword ptr [edi+0x58]
    mov       dword ptr [edx+0x40],esi
    movq      mm6,qword ptr [edi+0x28]
    add       esi,0x00000008
    pxor      mm7,qword ptr [edi+0x30]
    mov       dword ptr [edx+0x44],esi
    add       esi,0x00000008
    movq      mm5,qword ptr [edi+0x50]
    mov       dword ptr [edx+0x48],esi
    movq      mm4,qword ptr [edi+0x18]
    add       esi,0x00000008
    pxor      mm5,qword ptr [edi+0x20]
    mov       dword ptr [edx+0x4c],esi
    add       esi,0x00000008
    movq      mm3,qword ptr [edi+0x48]
    mov       dword ptr [edx+0x50],esi
    movq      mm2,qword ptr [edi+0x8]
    add       esi,0x00000008
    pxor      mm3,qword ptr [edi+0x10]
    mov       dword ptr [edx+0x54],esi
    add       esi,0x00000008
    movq      mm1,qword ptr [edi+0x40]
    mov       dword ptr [edx+0x58],esi
    movq      mm0,qword ptr [edi+0x38]
    add       esi,0x00000008
    pxor      mm1,qword ptr [edi]
    mov       dword ptr [edx+0x5c],esi
    push      eax
    call      near csc_transP2
    pop       eax
    add       ebx,0x00000080
    dec       dword ptr [esp+0x18]
    jne       near X$8
    mov       eax,dword ptr [esp+0x24]
    mov       ebx,dword ptr [esp+0xbc]
    mov       dword ptr [esp+0xcc],0xffffffff
    mov       dword ptr [esp+0xd0],0xffffffff
    mov       ecx,ebp
    mov       esi,0x00000000
    align 16 ; lea       esi,[esi]
.loop2:
    movq      mm7,qword ptr [eax+0x38]
    mov       dword ptr [edx+0x40],ecx
    pxor      mm7,qword ptr [ebx+0x38]
    add       ecx,0x00000040
    mov       dword ptr [edx+0x44],ecx
    movq      mm6,qword ptr [eax+0x30]
    add       ecx,0x00000040
    pxor      mm6,qword ptr [ebx+0x30]
    mov       dword ptr [edx+0x48],ecx
    movq      mm5,qword ptr [eax+0x28]
    add       ecx,0x00000040
    pxor      mm5,qword ptr [ebx+0x28]
    mov       dword ptr [edx+0x4c],ecx
    movq      mm4,qword ptr [eax+0x20]
    add       ecx,0x00000040
    pxor      mm4,qword ptr [ebx+0x20]
    mov       dword ptr [edx+0x50],ecx
    movq      mm3,qword ptr [eax+0x18]
    add       ecx,0x00000040
    pxor      mm3,qword ptr [ebx+0x18]
    mov       dword ptr [edx+0x54],ecx
    movq      mm2,qword ptr [eax+0x10]
    add       ecx,0x00000040
    pxor      mm2,qword ptr [ebx+0x10]
    mov       dword ptr [edx+0x58],ecx
    movq      mm1,qword ptr [eax+0x8]
    add       ecx,0x00000040
    pxor      mm1,qword ptr [ebx+0x8]
    mov       dword ptr [edx+0x5c],ecx
    movq      mm0,qword ptr [eax]
    sub       ecx,0x000001c0
    pxor      mm0,qword ptr [ebx]
    push      eax
    call      near csc_transP2
    pop       eax
    movq      mm0,qword ptr [esp+0xcc]
    mov       edi,dword ptr [esp+0xfc]
    mov       ebp,dword ptr [esp+0xdc]
    push      eax
    push      ebx
    lea       eax,[esi*8]
    add       edi,eax
    add       ebp,eax
    movq      mm1,qword ptr [edi+0x40]
    movq      mm2,qword ptr [edi]
    pxor      mm1,qword ptr [ebp+0x40]
    pxor      mm2,qword ptr [ebp]
    pxor      mm1,qword ptr [ecx+0x40]
    pxor      mm2,qword ptr [ecx]
    pxor      mm1,qword ptr [ecx-0x3c0]
    pxor      mm2,qword ptr [ecx-0x400]
    pandn     mm1,mm0
    add       edi,0x00000080
    pandn     mm2,mm1
    add       ebp,0x00000080
    movq      mm3,mm2
    movq      mm0,mm2
    punpckhdq mm3,mm3
    add       ecx,0x00000080
    movd      eax,mm2
    movd      ebx,mm3
    cmp       eax,0x00000000
    jne       .next_test1
    cmp       ebx,0x00000000
    je        near .goto_stepper
    mov       esi,esi
.next_test1:
    movq      mm1,qword ptr [edi+0x40]
    movq      mm2,qword ptr [edi]
    pxor      mm1,qword ptr [ebp+0x40]
    pxor      mm2,qword ptr [ebp]
    pxor      mm1,qword ptr [ecx+0x40]
    pxor      mm2,qword ptr [ecx]
    pxor      mm1,qword ptr [ecx-0x3c0]
    pxor      mm2,qword ptr [ecx-0x400]
    pandn     mm1,mm0
    add       edi,0x00000080
    pandn     mm2,mm1
    add       ebp,0x00000080
    movq      mm3,mm2
    movq      mm0,mm2
    punpckhdq mm3,mm3
    add       ecx,0x00000080
    movd      eax,mm2
    movd      ebx,mm3
    cmp       eax,0x00000000
    jne       .next_test2
    cmp       ebx,0x00000000
    je        near .goto_stepper
    align 16 ; lea       esi,[esi]
.next_test2:
    movq      mm1,qword ptr [edi+0x40]
    movq      mm2,qword ptr [edi]
    pxor      mm1,qword ptr [ebp+0x40]
    pxor      mm2,qword ptr [ebp]
    pxor      mm1,qword ptr [ecx+0x40]
    pxor      mm2,qword ptr [ecx]
    pxor      mm1,qword ptr [ecx-0x3c0]
    pxor      mm2,qword ptr [ecx-0x400]
    pandn     mm1,mm0
    add       edi,0x00000080
    pandn     mm2,mm1
    add       ebp,0x00000080
    movq      mm3,mm2
    movq      mm0,mm2
    punpckhdq mm3,mm3
    add       ecx,0x00000080
    movd      eax,mm2
    movd      ebx,mm3
    cmp       eax,0x00000000
    jne       .next_test3
    cmp       ebx,0x00000000
    je        .goto_stepper
    align 16 ;lea       esi,[esi]
.next_test3:
    movq      mm1,qword ptr [edi+0x40]
    movq      mm2,qword ptr [edi]
    pxor      mm1,qword ptr [ebp+0x40]
    pxor      mm2,qword ptr [ebp]
    pxor      mm1,qword ptr [ecx+0x40]
    pxor      mm2,qword ptr [ecx]
    pxor      mm1,qword ptr [ecx-0x3c0]
    pxor      mm2,qword ptr [ecx-0x400]
    pandn     mm1,mm0
    pandn     mm2,mm1
    movq      mm3,mm2
    movq      mm0,mm2
    punpckhdq mm3,mm3
    movd      eax,mm2
    movd      ebx,mm3
    cmp       eax,0x00000000
    jne       .next2
    cmp       ebx,0x00000000
    jne       .next2
    nop       
.goto_stepper:
    add       esp,0x00000008
    jmp       stepper
.next2:
    pop       ebx
    pop       eax
    movq      qword ptr [esp+0xcc],mm0
    add       eax,0x00000040
    add       ebx,0x00000040
    sub       ecx,0x00000178
    inc       esi
    cmp       esi,0x00000008
    jl        near .loop2
    mov       edi,dword ptr [esp+0xf0]
    mov       esi,dword ptr [esp+0x48]
    cld       
    mov       ecx,0x00000080
    repe movsd 
    cmp       dword ptr [esp+0xcc],0xdeedbeef
    jne       X$9
    cmp       dword ptr [esp+0xd0],0x00000000
    je        X$10

    db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
X$9:
    mov       eax,dword ptr [esp+0xcc]
    mov       edx,dword ptr [esp+0xd0]
    jmp       near X$23
    align 16
X$10:
    ;nop       
stepper:
    inc       dword ptr [esp+0xb8]
    mov       ebx,dword ptr [esp+0xb8]
    test      bl,0x01
    je        X$12
    mov       ebx,dword ptr [esp+0xc4]
    cmp       dword ptr [ebx],0x00000000
    je        near X$7

    db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi 
    db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi

X$11:
    mov       ecx,dword ptr [ebx]
    add       ebx,0x00000004
    mov       eax,dword ptr [ecx]
    mov       edx,dword ptr [ecx+0x4]
    not       eax
    not       edx
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    cmp       dword ptr [ebx],0x00000000
    jne       X$11
    jmp       near X$7
    align 16 ; lea       esi,[esi]
X$12:
    mov       esi,dword ptr [esp+0xb8]
    test      esi,0x00000002
    je        X$14
    mov       edi,dword ptr [esp+0xc4]
    mov       ebx,dword ptr [esp+0xc4]
    add       ebx,0x00000074
    cmp       dword ptr [edi+0x74],0x00000000
    je        near X$7

X$13:
    mov       ecx,dword ptr [ebx]
    add       ebx,0x00000004
    mov       eax,dword ptr [ecx]
    mov       edx,dword ptr [ecx+0x4]
    not       eax
    not       edx
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    cmp       dword ptr [ebx],0x00000000
    jne       X$13
    jmp       near X$7
    align 16 ; lea       esi,[esi]
X$14:
    mov       eax,dword ptr [esp+0xb8]
    test      al,0x04
    je        X$16
    mov       edx,dword ptr [esp+0xc4]
    mov       ebx,dword ptr [esp+0xc4]
    add       ebx,0x000000e8
    cmp       dword ptr [edx+0xe8],0x00000000
    je        near X$7

    nop
X$15:
    mov       ecx,dword ptr [ebx]
    add       ebx,0x00000004
    mov       eax,dword ptr [ecx]
    mov       edx,dword ptr [ecx+0x4]
    not       eax
    not       edx
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    cmp       dword ptr [ebx],0x00000000
    jne       X$15
    jmp       near X$7
    align 16 ;lea       esi,[esi]

X$16:
    mov       ecx,dword ptr [esp+0xb8]
    test      cl,0x08
    je        X$18
    mov       esi,dword ptr [esp+0xc4]
    mov       ebx,dword ptr [esp+0xc4]
    add       ebx,0x0000015c
    cmp       dword ptr [esi+0x15c],0x00000000
    je        near X$7

X$17:
    mov       ecx,dword ptr [ebx]
    add       ebx,0x00000004
    mov       eax,dword ptr [ecx]
    mov       edx,dword ptr [ecx+0x4]
    not       eax
    not       edx
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    cmp       dword ptr [ebx],0x00000000
    jne       X$17
    jmp       near X$7
    align 16 ;lea       esi,[esi]
X$18:
    mov       edi,dword ptr [esp+0xb8]
    test      edi,0x00000010
    je        X$20
    mov       eax,dword ptr [esp+0xc4]
    mov       ebx,dword ptr [esp+0xc4]
    add       ebx,0x000001d0
    cmp       dword ptr [eax+0x1d0],0x00000000
    je        near X$7

    db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;lea       edi,[0x0+edi*1]

X$19:
    mov       ecx,dword ptr [ebx]
    add       ebx,0x00000004
    mov       eax,dword ptr [ecx]
    mov       edx,dword ptr [ecx+0x4]
    not       eax
    not       edx
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    cmp       dword ptr [ebx],0x00000000
    jne       X$19
    jmp       near X$7
    align 16 ;lea       esi,[esi]
X$20:
    mov       edx,dword ptr [esp+0xb8]
    test      dl,0x20
    je        X$22
    mov       ecx,dword ptr [esp+0xc4]
    mov       ebx,dword ptr [esp+0xc4]
    add       ebx,0x00000244
    cmp       dword ptr [ecx+0x244],0x00000000
    je        near X$7

X$21:
    mov       ecx,dword ptr [ebx]
    add       ebx,0x00000004
    mov       eax,dword ptr [ecx]
    mov       edx,dword ptr [ecx+0x4]
    not       eax
    not       edx
    mov       dword ptr [ecx],eax
    mov       dword ptr [ecx+0x4],edx
    cmp       dword ptr [ebx],0x00000000
    jne       X$21
    jmp       near X$7
    align 16 ;lea       esi,[esi]

X$22:
    xor       eax,eax
    xor       edx,edx
X$23:
    pop       ebx
    pop       esi
    pop       edi
    pop       ebp
    add       esp,0x000000dc
    ret       
    nop       

    
    align 128 ; this is already para aligned actually
    
csc_unit_func_6b_mmx:
_csc_unit_func_6b_mmx:

    push      ebp
    mov       ebp,esp
    sub       esp,0x0000004c
    push      edi
    push      esi
    push      ebx
    mov       eax,dword ptr [ebp+0x10]
    mov       dword ptr [ebp-0x14],eax
    test      al,0x0f
    je        X$24
    add       dword ptr [ebp-0x14],0x0000000f
    and       byte ptr [ebp-0x14],0xf0
X$24:
    mov       edi,dword ptr [ebp-0x14]
    lea       edx,[edi+0x400]
    mov       dword ptr [ebp-0x18],edx
    lea       ecx,[edi+0x600]
    mov       dword ptr [ebp-0x1c],ecx
    lea       ebx,[edi+0x800]
    mov       dword ptr [ebp-0x14],ebx
    add       esp,0xfffffffc
    push      dword 0x00004400
    push      dword 0x00000000
    push      eax
    call      mymemset
    mov       edx,dword ptr [ebp+0x8]
    add       esp,0xfffffff8
    lea       esi,[ebp-0x4]
    lea       ebx,[ebp-0x8]
    mov       eax,dword ptr [edx+0x10]
    mov       dword ptr [ebp-0x8],eax
    mov       eax,dword ptr [edx+0x14]
    mov       dword ptr [ebp-0x4],eax
    push      esi
    push      ebx
    call      convert_key_from_inc_to_csc
    mov       ecx,dword ptr [ebp+0x8]
    mov       eax,dword ptr [ebp+0x8]
    mov       edx,0x00000001
    add       esp,0x00000020
    lea       ebx,[ebp-0x10]
    mov       ecx,dword ptr [ecx+0x4]
    mov       dword ptr [ebp-0x20],ecx
    mov       eax,dword ptr [eax+0xc]
    mov       ecx,dword ptr [ebp-0x4]
    mov       dword ptr [ebp-0x38],eax
    xor       eax,eax
    mov       dword ptr [ebp-0x30],ebx
    nop ; aligned
X$25:
    test      dword ptr [ebp-0x20],edx
    je        X$26
    mov       ebx,dword ptr [ebp-0x18]
    mov       dword ptr [ebx+eax*8],0xffffffff
    mov       dword ptr [ebx+eax*8+0x4],0xffffffff
    jmp       short X$27
    align 16 ;lea       esi,[esi]
X$26:
    mov       ebx,dword ptr [ebp-0x18]
    mov       dword ptr [ebx+eax*8],0x00000000
    mov       dword ptr [ebx+eax*8+0x4],0x00000000

    db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
X$27:
    test      dword ptr [ebp-0x38],edx
    je        X$28
    mov       ebx,dword ptr [ebp-0x1c]
    mov       dword ptr [ebx+eax*8],0xffffffff
    mov       dword ptr [ebx+eax*8+0x4],0xffffffff
    jmp       short X$29
    align 16 ;lea       esi,[esi]
X$28:
    mov       ebx,dword ptr [ebp-0x1c]
    mov       dword ptr [ebx+eax*8],0x00000000
    mov       dword ptr [ebx+eax*8+0x4],0x00000000

    db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
    db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]

X$29:
    test      edx,ecx
    je        X$30
    mov       dword ptr [edi+eax*8],0xffffffff
    mov       dword ptr [edi+eax*8+0x4],0xffffffff
    jmp       short X$31
    align 16 ;lea       esi,[esi]
X$30:
    mov       dword ptr [edi+eax*8],0x00000000
    mov       dword ptr [edi+eax*8+0x4],0x00000000

    nop ; aligned

X$31:
    add       edx,edx
    jne       X$32
    mov       edx,dword ptr [ebp+0x8]
    mov       ecx,dword ptr [ebp+0x8]
    mov       edx,dword ptr [edx]
    mov       dword ptr [ebp-0x20],edx
    mov       ecx,dword ptr [ecx+0x8]
    mov       dword ptr [ebp-0x38],ecx
    mov       ecx,dword ptr [ebp-0x8]
    mov       edx,0x00000001

    db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
X$32:
    inc       eax
    cmp       eax,0x0000003f
    jle       near X$25
    add       esp,0xfffffffc
    push      dword 0x00000200
    push      dword 0x00000000
    lea       eax,[edi+0x200]
    push      eax
    call      mymemset
    mov       edx,dword ptr [ebp-0x8]
    mov       eax,edx
    shr       eax,0x00000018
    mov       byte ptr [ebp-0x10],al
    mov       eax,dword ptr [ebp-0x10]
    mov       ebx,dword ptr [ebp-0x30]
    mov       ecx,edx
    shr       ecx,0x00000008
    and       ecx,0x0000ff00
    and       eax,0xff0000ff
    shl       edx,0x00000008
    and       edx,0x00ff0000
    or        eax,ecx
    or        eax,edx
    movzx     edx,byte ptr [ebp-0x8]
    shl       edx,0x00000018
    and       eax,0x00ffffff
    or        eax,edx
    mov       dword ptr [ebp-0x10],eax
    movzx     eax,byte ptr [ebp-0x1]
    mov       byte ptr [ebx+0x4],al
    movzx     eax,byte ptr [ebp-0x2]
    shl       eax,0x00000008
    add       esp,0x00000010
    mov       edx,dword ptr [ebx+0x4]
    xor       dh,dh
    or        edx,eax
    mov       dword ptr [ebx+0x4],edx
    movzx     eax,byte ptr [ebp-0x3]
    shl       eax,0x00000010
    movzx     edx,dx
    or        edx,eax
    mov       dword ptr [ebx+0x4],edx
    mov       ecx,dword ptr [csc_bit_order+0x18]

    ;.new_section .rodata, "dr2"
    ;      .byte    0xaa, 0xaa, 0xaa, 0xaa
    ;X$59: .byte    0xaa, 0xaa, 0xaa, 0xaa
    ;X$60: .byte    0xcc, 0xcc, 0xcc, 0xcc
    ;X$61: .byte    0xcc, 0xcc, 0xcc, 0xcc
    ;X$62: .byte    0xf0, 0xf0, 0xf0, 0xf0
    ;X$63: .byte    0xf0, 0xf0, 0xf0, 0xf0
    ;X$64: .byte    0x00, 0xff, 0x00, 0xff
    ;X$65: .byte    0x00, 0xff, 0x00, 0xff
    ;X$66: .byte    0x00, 0x00, 0xff, 0xff
    ;X$67: .byte    0x00, 0x00, 0xff, 0xff
    ;X$68: .byte    0x00, 0x00, 0x00, 0x00
    ;X$69: .byte    0xff, 0xff, 0xff, 0xff

    mov       eax,0AAAAAAAAh ;dword ptr .rodata = .byte 0xaa, 0xaa, 0xaa, 0xaa
    mov       edx,0AAAAAAAAh ;dword ptr X$59 = .byte    0xaa, 0xaa, 0xaa, 0xaa
    mov       dword ptr [edi+ecx*8],eax
    mov       dword ptr [edi+ecx*8+0x4],edx
    mov       ecx,dword ptr [csc_bit_order+0x1c]
    mov       eax,0CCCCCCCCh ;dword ptr X$60 = .byte    0xcc, 0xcc, 0xcc, 0xcc
    mov       edx,0CCCCCCCCh ;dword ptr X$61 = .byte    0xcc, 0xcc, 0xcc, 0xcc
    mov       dword ptr [edi+ecx*8],eax
    mov       dword ptr [edi+ecx*8+0x4],edx
    mov       ecx,dword ptr [csc_bit_order+0x20]
    mov       eax,0F0F0F0F0h ;dword ptr X$62 = .byte    0xf0, 0xf0, 0xf0, 0xf0
    mov       edx,0F0F0F0F0h ;dword ptr X$63 = .byte    0xf0, 0xf0, 0xf0, 0xf0
    mov       dword ptr [edi+ecx*8],eax
    mov       dword ptr [edi+ecx*8+0x4],edx
    mov       ecx,dword ptr [csc_bit_order+0x24]
    mov       eax,0FF00FF00h ; dword ptr X$64 = .byte   0x00, 0xff, 0x00, 0xff
    mov       edx,0FF00FF00h ; dword ptr X$65 = .byte   0x00, 0xff, 0x00, 0xff
    mov       dword ptr [edi+ecx*8],eax
    mov       dword ptr [edi+ecx*8+0x4],edx
    mov       ecx,dword ptr [csc_bit_order+0x28]
    mov       eax,0FFFF0000h ; dword ptr X$66 = .byte    0x00, 0x00, 0xff, 0xff
    mov       edx,0FFFF0000h ; dword ptr X$67 = .byte    0x00, 0x00, 0xff, 0xff
    mov       dword ptr [edi+ecx*8],eax
    mov       dword ptr [edi+ecx*8+0x4],edx
    mov       ecx,dword ptr [csc_bit_order+0x2c]
    mov       eax,000000000h ; dword ptr X$68 = .byte   0x00, 0x00, 0x00, 0x00
    mov       edx,0FFFFFFFFh ; dword ptr X$69 = .byte   0xff, 0xff, 0xff, 0xff
    mov       dword ptr [edi+ecx*8],eax
    mov       eax,dword ptr [ebp+0xc]
    mov       dword ptr [edi+ecx*8+0x4],edx
    mov       dword ptr [ebp-0x24],0x0000000c
    mov       edx,dword ptr [eax]
    cmp       edx,0x00001000
    jbe       X$34
    mov       ebx,0x00000001

    ;lea       esi,[esi]
    ;lea       edi,[edi]
    db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
    db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi
X$33:
    inc       dword ptr [ebp-0x24]
    mov       ecx,dword ptr [ebp-0x24]
    mov       eax,ebx
    shl       eax,cl
    cmp       edx,eax
    ja        X$33

    db 0x89, 0xf6                               ; mov      esi, esi
X$34:
    mov       eax,dword ptr [ebp-0x24]
    mov       esi,dword ptr [ebp-0x30]
    xor       ebx,ebx
    add       eax,0xfffffff4
    mov       dword ptr [ebp-0x34],eax

    ; aligned
X$35:
    mov       ecx,dword ptr [ebx*4+csc_bit_order]
    mov       dword ptr [edi+ecx*8],0x00000000
    mov       dword ptr [edi+ecx*8+0x4],0x00000000
    mov       eax,ecx
    test      ecx,ecx
    jge       X$36
    lea       eax,[ecx+0x7]
    nop ; aligned
X$36:
    sar       eax,0x00000003
    mov       edx,0x00000007
    sub       edx,eax
    shl       eax,0x00000003
    sub       ecx,eax
    mov       eax,0x00000001
    shl       eax,cl
    not       al
    and       byte ptr [edx+esi],al
    inc       ebx
    cmp       ebx,0x00000005
    jle       X$35
    xor       esi,esi
    cmp       esi,dword ptr [ebp-0x34]
    jae       X$39
    mov       ebx,dword ptr [ebp-0x30]
    mov       edx,dword ptr [ebp-0x34]
    mov       dword ptr [ebp-0x38],csc_bit_order+0x30
    mov       dword ptr [ebp-0x28],edx
    
    ;mov       esi,esi
    ;lea       edi,[edi]
    db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi
    db 0x90

X$37:
    mov       eax,dword ptr [ebp-0x38]
    mov       ecx,dword ptr [eax]
    mov       dword ptr [edi+ecx*8],0x00000000
    mov       dword ptr [edi+ecx*8+0x4],0x00000000
    mov       eax,ecx
    test      ecx,ecx
    jge       X$38
    lea       eax,[ecx+0x7]
    db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
X$38:
    sar       eax,0x00000003
    mov       edx,0x00000007
    sub       edx,eax
    shl       eax,0x00000003
    sub       ecx,eax
    mov       eax,0x00000001
    shl       eax,cl
    not       al
    and       byte ptr [edx+ebx],al
    add       dword ptr [ebp-0x38],0x00000004
    inc       esi
    cmp       esi,dword ptr [ebp-0x28]
    jb        X$37
    db 0x89, 0xf6           ; mov esi, esi
    db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi
X$39:
    mov       edx,dword ptr [ebp-0x30]
    xor       esi,esi
    mov       dword ptr [ebp-0x38],edx
    jmp       short X$43
    align 16 ;lea       esi,[esi]
X$40:
    xor       ebx,ebx
    test      esi,0x00000001
    jne       X$42
    mov       edx,0x00000001
    nop ;aligned       
X$41:
    inc       ebx
    mov       eax,edx
    mov       ecx,ebx
    shl       eax,cl
    test      eax,esi
    je        X$41
    db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi
    db 0x90
X$42:
    mov       ebx,dword ptr [ebx*4+csc_bit_order+0x30]
    mov       ecx,ebx
    shr       ecx,0x00000003
    mov       eax,dword ptr [edi+ebx*8]
    mov       edx,dword ptr [edi+ebx*8+0x4]
    not       eax
    not       edx
    mov       dword ptr [edi+ebx*8],eax
    mov       dword ptr [edi+ebx*8+0x4],edx
    mov       edx,0x00000007
    sub       edx,ecx
    and       ebx,0x00000007
    mov       ecx,ebx
    mov       ebx,dword ptr [ebp-0x38]
    mov       eax,0x00000001
    shl       eax,cl
    xor       byte ptr [edx+ebx],al
    db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
X$43:
    mov       eax,dword ptr [ebp-0x14]
    add       esp,0xfffffff4
    push      eax
    mov       edx,dword ptr [ebp-0x1c]
    push      edx
    mov       ecx,dword ptr [ebp-0x18]
    push      ecx
    mov       ebx,dword ptr [ebp-0x30]
    push      ebx
    push      edi
    call      cscipher_bitslicer_6b_mmx
    mov       dword ptr [ebp-0x40],eax
    mov       eax,dword ptr [ebp-0x40]
    mov       dword ptr [ebp-0x3c],edx
    add       esp,0x00000020
    or        eax,dword ptr [ebp-0x3c]
    jne       X$44
    mov       ecx,dword ptr [ebp-0x34]
    inc       esi
    mov       eax,esi
    shr       eax,cl
    test      eax,eax
    je        near X$40
    ;aligned
X$44:
    mov       eax,dword ptr [ebp-0x40]
    or        eax,dword ptr [ebp-0x3c]
    je        near X$56
    xor       eax,eax
    cmp       dword ptr [ebp-0x40],0x00000001
    jne       X$45
    cmp       dword ptr [ebp-0x3c],0x00000000
    je        X$46
    ;aligned
X$45:
    mov       edx,dword ptr [ebp-0x40]
    mov       ecx,dword ptr [ebp-0x3c]
    inc       eax
    shrd      edx,ecx,0x00000001
    shr       ecx,0x00000001
    mov       dword ptr [ebp-0x40],edx
    mov       dword ptr [ebp-0x3c],ecx
    cmp       edx,0x00000001
    jne       X$45
    test      ecx,ecx
    jne       X$45
    nop ;aligned
X$46:
    mov       dword ptr [ebp-0x8],0x00000000
    mov       dword ptr [ebp-0x4],0x00000000
    mov       esi,0x00000008
    mov       byte ptr [ebp-0x29],al
    mov       ebx,0xffffffe8

    db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi
    db 0x90
X$47:
    cmp       esi,0x0000001f
    jg        X$50
    mov       eax,dword ptr [edi+ebx*8+0x100]
    mov       edx,dword ptr [edi+ebx*8+0x104]
    mov       cl,byte ptr [ebp-0x29]
    shrd      eax,edx,cl
    shr       edx,cl
    test      cl,0x20
    je        X$48
    mov       eax,edx
    xor       edx,edx

X$48:
    and       eax,0x00000001
    and       edx,0x00000000
    mov       ecx,esi
    shld      edx,eax,cl
    shl       eax,cl
    test      cl,0x20
    je        X$49
    mov       edx,eax
    xor       eax,eax
X$49:
    or        dword ptr [ebp-0x4],eax
    jmp       short X$53
    align 16
X$50:
    mov       eax,dword ptr [edi+ebx*8+0x100]
    mov       edx,dword ptr [edi+ebx*8+0x104]
    mov       cl,byte ptr [ebp-0x29]
    shrd      eax,edx,cl
    shr       edx,cl
    test      cl,0x20
    je        X$51
    mov       eax,edx
    xor       edx,edx
X$51:
    and       eax,0x00000001
    and       edx,0x00000000
    mov       cl,bl
    shld      edx,eax,cl
    shl       eax,cl
    test      cl,0x20
    je        X$52
    mov       edx,eax
    xor       eax,eax
X$52:
    or        dword ptr [ebp-0x8],eax
X$53:
    inc       ebx
    inc       esi
    cmp       esi,0x0000003f
    jle       near X$47
    add       esp,0xfffffff8
    lea       ebx,[ebp-0x4]
    push      ebx
    lea       eax,[ebp-0x8]
    push      eax
    call      convert_key_from_csc_to_inc
    mov       ecx,dword ptr [ebp+0x8]
    mov       edx,dword ptr [ebp-0x4]
    add       esp,0x00000010
    mov       eax,dword ptr [ecx+0x14]
    cmp       edx,eax
    jae       X$54
    mov       ebx,dword ptr [ebp+0xc]
    sub       eax,edx
    mov       dword ptr [ebx],eax
    jmp       short X$55
    align 16
X$54:
    mov       ecx,dword ptr [ebp+0xc]
    sub       edx,eax
    mov       dword ptr [ecx],edx
X$55:
    mov       eax,dword ptr [ebp-0x4]
    mov       ebx,dword ptr [ebp+0x8]
    mov       dword ptr [ebx+0x14],eax
    mov       eax,dword ptr [ebp-0x8]
    mov       dword ptr [ebx+0x10],eax
    emms      
    mov       eax,0x00000002
    jmp       short X$58
    align 16
X$56:
    mov       ecx,dword ptr [ebp-0x24]
    mov       ebx,dword ptr [ebp+0xc]
    mov       eax,0x00000001
    shl       eax,cl
    mov       dword ptr [ebx],eax
    mov       edx,dword ptr [ebp+0x8]
    add       eax,dword ptr [edx+0x14]
    mov       dword ptr [edx+0x14],eax
    shr       eax,cl
    test      eax,eax
    jne       X$57
    inc       dword ptr [edx+0x10]
X$57:
    emms      
    mov       eax,0x00000001
X$58:
    lea       esp,[ebp-0x58]
    pop       ebx
    pop       esi
    pop       edi
    mov       esp,ebp
    pop       ebp
    ret       


__CODESECT__
    align 16
mymemset:
    push       edi
    push       ebx
    mov        edi, [esp+0xc]
    movzx eax, BYTE [esp+0x10] ;0f b6 44 24 10
    mov        ecx, [esp+0x14]
    push       edi
    cld              
    cmp        ecx, 15
    jle        short .here
    mov        ah,al
    mov        edx,eax
    shl        eax,16
    or         eax,edx
    mov        edx,edi
    neg        edx
    and        edx,3
    mov        ebx,ecx
    sub        ebx,edx
    mov        ecx,edx
    repz       stosb
    mov        ecx,ebx
    shr        ecx,2
    repz       stosd
    mov        ecx,ebx
    and        ecx,3
.here:
    repz stosb
    pop        eax
    pop        ebx
    pop        edi
    ret              

__CODESECT__
    align 32

    



;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;lea       edi,[0x0+edi*1]
;   db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
;   db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
;   db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi 
;   db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
;   db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi

;   align 16 ; lea       esi,[esi]
;   align 16 ;lea       esi,[esi]
;   mov       esi,esi
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;lea       edi,[0x0+edi*1]
;   db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
;
;   db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
;   db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi
;    
;   db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi 
;   db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
;   db 0x89, 0xf6           ; mov esi, esi

;2:
;   db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
;3:
;   db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
;4:
;   db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
;   db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi
;5:
;   db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi 
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
;6:
;   db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
;   db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal     0x0(%edi,1),%edi
;7:
;   db 0x8d, 0x76, 0x00                         ; leal     0x0(%esi),%esi
;   db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi
;8:
;   db 0x89, 0xf6           ; mov esi, esi
;   db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%edi),%edi
;9:
;   db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal     esi,[0x0+esi*1]
;A:
;   db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00       ; leal     0x0(%esi),%esi
;B:
;   db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi
;   db 0x90
;C:
;   db 0x8d, 0x74, 0x26, 0x00                   ; leal     0x0(%esi,1),%esi
;D:
;E:
;   db 0x89, 0xf6                               ; mov      esi, esi
