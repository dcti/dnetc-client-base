%include "csc-mac.inc"

extern csc_tabp, csc_tabe, csc_tabc, csc_bit_order
extern convert_key_from_inc_to_csc, convert_key_from_csc_to_inc
;extern memset

; this file has been produced by the following chain :
;
; - gcc 2.95.2 which produced a .s
;
; - intel2gas (http://hermes.terminal.at/intel2gas/index.html) which produced a .asm
;
; - hours of editing because :
;
;  * intel2gas is dumb
;
;  * nasm is totally stupid and non-efficient
;    . it systematically choose the longest version of an instruction
;      if it has the choice :-( 
;    . it can't assemble some instructions in a efficient manner (see the "db" lines)
;
;  * gas is smart
;    . it can skip bytes in code using a few long instructions instead of a load of nops
;    . you can put a limit on the number of bytes to fill with nops (or pseudo nops)
;      something which nasm can't do at all, even with the help of a custom macro
;    . it systematically choose the smallest version of an instruction
;      if it has the choice
;
; - this file should assemble to the same .o which gcc+gas produce
;   from the .cpp (modulo symbol names)

SECTION .text

        ALIGN 32
GLOBAL cscipher_bitslicer_6b_i
cscipher_bitslicer_6b_i: 
        sub  esp,2012
        push  ebp
        push  edi
        push  esi
        push  ebx
        mov  eax, [esp+2048]
        mov  [esp+1732],eax
        mov  edx, [esp+1732]
        mov  edi, [esp+1732]
        mov  esi, [esp+2032]
        add  eax,2816
        mov  [esp+1728],eax
        add  edx,3520
        mov  [esp+1724],edx
        sub  edx, BYTE -128
        mov  [esp+1720],edx
        add  esi,256
        lea  ebp, [esp+1752]
        lea  eax, [esp+1748]
        cld
        mov  ecx,64
        rep movsd
        mov  edi, [esp+1732]
        mov  esi, [esp+2032]
        lea  edx, [esp+1744]
        add  edi,256
        cld
        mov  ecx,64
        rep movsd
        mov  ebx, [esp+1732]
        mov  dword [esp+1648],0
        mov  dword [ebx+608],0
        mov  dword [ebx+640],0
        mov  dword [ebx+672],0
        mov  dword [ebx+704],0
        mov  dword [ebx+736],0
        mov  dword [ebx+512],-1
        mov  dword [ebx+544],-1
        mov  dword [ebx+576],-1
        mov  esi, [esp+1732]
        mov  dword [esp+1708], csc_tabc+32
        add  ebx,516
        mov  [esp+1716],ebx
        add  esi,288
        mov  [esp+1712],esi
        mov  dword [esp+1644],7
        lea  edi, [esp+1760]
        mov  [esp+144],edi
        mov  [esp+148],ebp
        mov  [esp+152],eax
        mov  [esp+156],edx
        lea  ecx, [esp+1740]
        mov  [esp+160],ecx
        lea  ebx, [esp+1736]
        mov  [esp+164],ebx
        ;; ALIGN 1<<4
	mov  esi,esi
	;lea  edi,[0x0+edi*1]
	db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00
L20:
        mov  esi, [esp+1712]
        mov  edi, [esi+28]
        mov  esi, [esp+1708]
        mov  ebp, [esp+1712]
        mov  ebx, [esp+1712]
        xor  edi, [esi+28]
        mov  [esp+1632],edi
        mov  eax, [ebp+24]
        xor  eax, [esi+24]
        mov  [esp+1628],eax
        mov  edx, [ebp+20]
        xor  edx, [esi+20]
        mov  [esp+1624],edx
        mov  ecx, [ebp+16]
        mov  edi, [ebx+12]
        xor  ecx, [esi+16]
        xor  edi, [esi+12]
        mov  [esp+1640],edi
        mov  ebx, [ebx+8]
        mov  eax, [esp+1712]
        mov  ebp,ecx
        mov  ecx, [esp+1708]
        xor  ebx, [esi+8]
        mov  edx, [eax+4]
        xor  edx, [esi+4]
        mov  [esp+1636],edx
        mov  esi, [eax]
        xor  esi, [ecx]
        mov  ecx,edi
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1636]
        mov  edx,ebx
        or  edx,edi
        xor  edx,ecx
        xor  [esp+1632],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1628],eax
        xor  eax,esi
        or  eax, [esp+1636]
        xor  [esp+1624],eax
        mov  eax,ebp
        or  eax, [esp+1632]
        mov  ecx,ebp
        and  ecx, [esp+1624]
        xor  ecx,eax
        mov  [esp+1620],ecx
        xor  [esp+1620],ebx
        mov  edi,ebp
        or  edi, [esp+1628]
        xor  edi, [esp+1632]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1616],ebx
        xor  [esp+1616],esi
        xor  ebx,edi
        mov  edi, [esp+1616]
        mov  edx,ebp
        or  edx, [esp+1624]
        mov  eax,ebp
        and  eax, [esp+1628]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1620]
        not  esi
        xor  esi, [esp+1640]
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1636]
        mov  eax,esi
        not  eax
        mov  [esp+28],eax
        or  [esp+28],edi
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1620]
        mov  eax,edx
        xor  eax,edi
        mov  edi, [esp+1716]
        or  eax,ebx
        xor  ecx, [esp+1632]
        mov  [edi+224],ecx
        xor  edx, [esp+1628]
        mov  [edi+192],edx
        xor  eax, [esp+1624]
        mov  [edi+160],eax
        mov  eax, [esp+28]
        xor  eax,ebp
        mov  [edi+128],eax
        mov  [edi+96],esi
        mov  edx, [esp+1620]
        mov  [edi+64],edx
        mov  [edi+32],ebx
        mov  ecx, [esp+1616]
        mov  [edi],ecx
        add  dword [esp+1708],BYTE 32
        add  dword [esp+1712],BYTE 32
        add  edi,BYTE 4
        mov  [esp+1716],edi
        dec  dword [esp+1644]
        jnz near L20
        mov  ebx, [esp+2036]
        mov  esi, [esp+1728]
        mov  edi, [esp+1732]
        mov  ebp, [esp+1732]
        mov  dword [esp+1612],2
        add  ebx,BYTE 7
        mov  [esp+132],ebx
        add  esi,-232
        mov  [esp+140],esi
        add  edi,512
        mov  [esp+120],edi
        add  ebp,768
        mov  [esp+116],ebp
        mov  [esp+1600],esi
        ;;ALIGN 1<<4
	db 0x8d, 0x74, 0x26, 0x00
L33:
        mov  edx, [esp+132]
        sub  edx, [esp+1612]
        mov  eax,7
        sub  eax, [esp+1612]
        mov  ebx, [esp+1732]
        mov  dl, [edx]
        xor  dl, [csc_tabp+eax]
        mov  [esp+1611],dl
        xor  byte [esp+1611],64
        movzx  eax,byte [esp+1611]
        movzx  edx,dl
        mov  dl, [csc_tabp+edx]
        xor  dl, [csc_tabp+eax]
        mov  [esp+1611],dl
        imul  edx, [1612+esp], BYTE 116
        mov  eax, [esp+1600]
	;MISMATCH: "        leal (%eax,%edx),%ecx"
	lea  ecx, [eax+edx]
        mov  eax, [esp+1612]
        sal  eax,5
	;MISMATCH: "        leal 280(%ebx,%eax),%eax"
	lea  eax, [280+ebx+eax]
        mov  [ecx],eax
        mov  esi, [esp+1612]
        mov  edi, [esp+1612]
        mov  ebp, [esp+140]
        mov  dword [esp+1604],1
        mov  dword [esp+1596],0
        mov  [esp+124],edx
        inc  esi
        mov  [esp+128],esi
        mov  eax,ebx
        add  eax,512
        lea  eax, [eax+edi*4]
        mov  [esp+104],eax
	;MISMATCH: "        leal 4(%ebp,%edx),%esi"
	lea  esi,[4+ebp+edx]
        add  ecx,BYTE 4
        ;;ALIGN 1<<4
	db 0x89, 0xf6  ; movl   %esi,%esi
	db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00  ; leal   0x0(%edi,1),%edi
L37:
        mov  edx, [esp+1596]
        movzx  eax,byte [esp+1611]
        bt  eax,edx
        jnc near L36
        mov  ebx, [esp+1612]
        mov  ebp, [esp+104]
        mov  [ecx],ebp
        mov  ebp, [esp+1724]
        lea  edi, [ebx+edx*8]
        add  ecx,BYTE 4
        mov  ebx,edi
        shr  ebx,4
        sal  ebx,5
        add  ebp,ebx
        mov  eax,edi
        and  eax,BYTE 7
        lea  eax, [ebp+eax*4]
        mov  [ecx],eax
        add  esi,BYTE 8
        add  ecx,BYTE 4
        add  dword [esp+1604],BYTE 2
        mov  edx,edi
        and  edx,BYTE 15
        cmp  edx,BYTE 7
        ja L39
        mov  eax, [esp+1720]
        add  eax,ebx
        lea  eax, [eax+edx*4]
        mov  [esi],eax
        jmp short L514
	ALIGN 1<<4 ; IF < 7

L39:
        mov  edx, [esp+1720]
        add  edx,ebx
        lea  eax, [edi+1]
        and  eax,BYTE 7
        sal  eax,2
        add  edx,eax
        mov  [esi],edx
        add  esi,BYTE 4
        add  ecx,BYTE 4
        inc  dword [esp+1604]
        test  edi,1
        je L36
        add  ebp,eax
        mov  [esi],ebp
L514:
        add  esi,BYTE 4
        add  ecx,BYTE 4
        inc  dword [esp+1604]
L36:
        add  dword [esp+104],BYTE 32
        inc  dword [esp+1596]
        cmp  dword [esp+1596],BYTE 7
        jle near L37
        mov  edx, [esp+1604]
        mov  ecx, [esp+124]
        mov  ebx, [esp+1600]
        lea  eax, [ecx+edx*4]
        mov  dword [ebx+eax],0
        mov  esi, [esp+128]
        mov  [esp+1612],esi
        cmp  esi,BYTE 7
        jle near L33
        mov  edi, [esp+120]
        mov  ebp, [esp+1724]
        mov  [esp+1716],edi
        mov  edi, [esp+1720]
        mov  ebx, [esp+2040]
        mov  dword [esp+1592],3
        mov  [esp+20],ebp
        ;;ALIGN 4
	nop
	db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00  ; leal   0x0(%esi,1),%esi
L47:
        mov  eax, [esp+1716]
        mov  edx, [ebx+32]
        xor  edx, [eax+32]
        mov  [esp+1700],edx
        mov  ecx, [ebx+36]
        xor  ecx, [eax+36]
        mov  [esp+1696],ecx
        mov  esi, [ebx+40]
        xor  esi, [eax+40]
        mov  [esp+1692],esi
        mov  esi, [ebx+44]
        mov  ebp, [ebx+48]
        xor  esi, [eax+44]
        xor  ebp, [eax+48]
        mov  [esp+1688],ebp
        mov  edx, [ebx+52]
        mov  ecx, [ebx+56]
        xor  edx, [eax+52]
        xor  ecx, [eax+56]
        mov  [esp+1684],ecx
        mov  ecx, [ebx+60]
        mov  ebp, [ebx+28]
        xor  ecx, [eax+60]
        xor  ebp, [eax+28]
        mov  eax, [esp+20]
        mov  [esp+1668],ebp
        xor  ebp,ecx
        mov  [eax+28],ebp
        mov  ebp, [esp+1716]
        mov  eax,edx
        xor  eax, [ebx+24]
        xor  eax, [ebp+24]
        mov  [edi+24],eax
        mov  ebp, [esp+20]
        xor  eax, [esp+1684]
        mov  [ebp+24],eax
        mov  eax, [esp+1716]
        mov  ebp, [ebx+20]
        xor  ebp, [eax+20]
        mov  eax, [esp+20]
        mov  [esp+1672],ebp
        xor  edx,ebp
        mov  [eax+20],edx
        mov  edx, [esp+1716]
        mov  eax,esi
        xor  eax, [ebx+16]
        xor  eax, [edx+16]
        mov  [edi+16],eax
        mov  ebp, [esp+20]
        xor  eax, [esp+1688]
        mov  [ebp+16],eax
        mov  eax, [ebx+12]
        xor  eax, [edx+12]
        mov  [esp+1676],eax
        xor  esi,eax
        mov  [ebp+12],esi
        mov  eax, [esp+1696]
        xor  eax, [ebx+8]
        xor  eax, [edx+8]
        mov  [edi+8],eax
        xor  eax, [esp+1692]
        mov  [ebp+8],eax
        mov  esi, [ebx+4]
        mov  eax, [esp+1696]
        xor  esi, [edx+4]
        mov  [esp+1680],esi
        xor  eax,esi
        mov  [ebp+4],eax
        mov  eax,ecx
        xor  eax, [ebx]
        xor  eax, [edx]
        mov  [edi],eax
        xor  eax, [esp+1700]
        mov  [ebp],eax
        mov  edx, [esp+1684]
        xor  edx, [esp+1668]
        mov  [edi+28],edx
        mov  ecx, [esp+1688]
        xor  ecx, [esp+1672]
        mov  [edi+20],ecx
        mov  esi, [esp+1692]
        xor  esi, [esp+1676]
        mov  [edi+12],esi
        mov  ebp, [esp+1700]
        xor  ebp, [esp+1680]
        mov  [edi+4],ebp
        add  dword [esp+1716],BYTE 64
        add  dword [esp+20],BYTE 32
        add  edi,BYTE 32
        add  ebx,BYTE 64
        dec  dword [esp+1592]
        jns near L47
        ;;ALIGN 1<<4
	nop
	db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ; leal   0x0(%esi,1),%esi
L49:
        mov  edi, [esp+144]
        mov  esi, [esp+2040]
        cld
        mov  ecx,64
        rep movsd
        mov  ebx, [esp+1724]
        mov  esi, [esp+1724]
        mov  eax, [esp+1724]
        mov  ebx, [ebx+12]
        mov  [esp+100],ebx
        mov  ebx, [esi+8]
        mov  edi, [esi+4]
        mov  [esp+96],edi
        mov  esi, [esi]
        mov  edx, [esp+100]
        not  edx
        mov  ebp,edx
        or  ebp,esi
        xor  ebp, [eax+16]
        mov  eax,ebx
        or  eax, [esp+100]
        xor  eax,edx
        mov  edx, [esp+1724]
        mov  ecx, [esp+1724]
        mov  [esp+1588],eax
        mov  edx, [edx+28]
        xor  [esp+1588],edx
        mov  edx,edi
        xor  edx,eax
        or  edx,ebx
        mov  [esp+1584],edx
        mov  ecx, [ecx+24]
        mov  [esp+1580],esi
        xor  [esp+1580],edx
        or  [esp+1580],edi
        mov  edi, [esp+1724]
        xor  [esp+1584],ecx
        mov  eax,ebp
        mov  ecx,ebp
        mov  edx,ebp
        mov  edi, [edi+20]
        xor  [esp+1580],edi
        or  eax, [esp+1588]
        and  ecx, [esp+1580]
        xor  ecx,eax
        mov  [esp+1576],ecx
        xor  [esp+1576],ebx
        mov  edi,ebp
        or  edi, [esp+1584]
        xor  edi, [esp+1588]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1572],ebx
        xor  [esp+1572],esi
        or  edx, [esp+1580]
        mov  eax,ebp
        and  eax, [esp+1584]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1576]
        not  esi
        xor  esi, [esp+100]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1572]
        xor  ebp,edi
        mov  [esp+1808],ebp
        mov  ebp, [esp+1576]
        not  ebx
        xor  ebx, [esp+96]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1576]
        mov  eax,edx
        xor  eax, [esp+1572]
        or  eax,ebx
        xor  eax, [esp+1580]
        mov  [esp+1812],eax
        mov  eax, [esp+1572]
        xor  edx, [esp+1584]
        mov  [esp+1816],edx
        mov  edx, [esp+1720]
        xor  ecx, [esp+1588]
        mov  [esp+1820],ecx
        mov  [esp+1804],esi
        mov  [esp+1800],ebp
        mov  [esp+1796],ebx
        mov  [esp+1792],eax
        mov  edx, [edx+12]
        mov  [esp+92],edx
        mov  ecx, [esp+1720]
        not  edx
        mov  ebx, [ecx+8]
        mov  esi, [ecx+4]
        mov  [esp+88],esi
        mov  edi, [ecx]
        mov  [esp+84],edi
        mov  esi,edx
        or  esi,edi
        xor  esi, [ecx+16]
        mov  eax,ebx
        or  eax, [esp+92]
        xor  eax,edx
        mov  [esp+1568],eax
        mov  ebp, [ecx+28]
        mov  edx, [esp+88]
        xor  [esp+1568],ebp
        xor  edx,eax
        or  edx,ebx
        mov  [esp+1564],edx
        mov  eax, [ecx+24]
        xor  edi,edx
        mov  edx, [esp+84]
        xor  [esp+1564],eax
        or  edi, [esp+88]
        xor  edi, [ecx+20]
        mov  eax,esi
        or  eax, [esp+1568]
        mov  ecx,esi
        and  ecx,edi
        xor  ecx,eax
        mov  [esp+1560],ecx
        xor  [esp+1560],ebx
        mov  ebx,esi
        or  ebx, [esp+1564]
        xor  ebx, [esp+1568]
        mov  ebp,ebx
        or  ebp,ecx
        xor  ebp,esi
        mov  [esp+1556],ebp
        xor  [esp+1556],edx
        mov  edx,esi
        or  edx,edi
        mov  eax,esi
        and  eax, [esp+1564]
        xor  eax,edx
        mov  edx, [esp+92]
        mov  [esp+1552],eax
        not  dword [esp+1552]
        xor  [esp+1552],edx
        or  eax,ecx
        mov  ecx, [esp+88]
        mov  edx, [esp+1552]
        xor  ebx,ebp
        xor  ebx,eax
        not  ebx
        xor  ebx,ecx
        mov  [esp+1548],ebx
        mov  ebx, [esp+1556]
        mov  eax, [esp+1560]
        not  edx
        mov  [esp+1544],edx
        or  [esp+1544],ebx
        xor  [esp+1544],esi
        mov  esi, [esp+1568]
        or  eax, [esp+1552]
        xor  eax,edx
        mov  edx, [esp+1548]
        mov  ebp, [esp+1564]
        mov  [esp+1540],eax
        xor  edx,eax
        mov  eax, [esp+1548]
        xor  [esp+1540],esi
        or  edx, [esp+1560]
        mov  [esp+1536],edx
        mov  [esp+1532],ebx
        xor  [esp+1532],edx
        mov  edx, [esp+1540]
        xor  [esp+1536],ebp
        mov  ecx, [esp+1536]
        or  [esp+1532],eax
        xor  [esp+1532],edi
        mov  [esp+1788],edx
        mov  [esp+1784],ecx
        mov  ebx, [esp+1532]
        mov  esi, [esp+1544]
        mov  edi, [esp+1552]
        mov  ebp, [esp+1560]
        mov  [esp+1764],eax
        mov  eax, [esp+1556]
        mov  edx, [esp+1724]
        mov  ecx, [esp+1724]
        mov  [esp+1780],ebx
        mov  [esp+1776],esi
        mov  [esp+1772],edi
        mov  [esp+1768],ebp
        mov  [esp+1760],eax
        mov  edx, [edx+44]
        mov  [esp+80],edx
        mov  ebx, [ecx+40]
        mov  esi, [ecx+36]
        mov  [esp+76],esi
        mov  esi, [ecx+32]
        not  edx
        mov  ebp,edx
        or  ebp,esi
        xor  ebp, [ecx+48]
        mov  eax,ebx
        or  eax, [esp+80]
        xor  eax,edx
        mov  [esp+1528],eax
        mov  edi, [ecx+60]
        mov  edx, [esp+76]
        xor  [esp+1528],edi
        xor  edx,eax
        or  edx,ebx
        mov  [esp+1524],edx
        mov  eax, [ecx+56]
        mov  [esp+1520],esi
        xor  [esp+1520],edx
        mov  edx, [esp+76]
        xor  [esp+1524],eax
        or  [esp+1520],edx
        mov  ecx, [ecx+52]
        xor  [esp+1520],ecx
        mov  eax,ebp
        or  eax, [esp+1528]
        mov  ecx,ebp
        and  ecx, [esp+1520]
        xor  ecx,eax
        mov  [esp+1516],ecx
        xor  [esp+1516],ebx
        mov  edi,ebp
        or  edi, [esp+1524]
        xor  edi, [esp+1528]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1512],ebx
        xor  [esp+1512],esi
        mov  edx,ebp
        or  edx, [esp+1520]
        mov  eax,ebp
        and  eax, [esp+1524]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1516]
        not  esi
        xor  esi, [esp+80]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        or  ecx,esi
        mov  [esp+1868],esi
        mov  esi, [esp+1516]
        not  eax
        mov  edi,eax
        or  edi, [esp+1512]
        xor  ebp,edi
        mov  edi, [esp+1512]
        not  ebx
        xor  ebx, [esp+76]
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1516]
        mov  eax,edx
        xor  eax, [esp+1512]
        or  eax,ebx
        xor  ecx, [esp+1528]
        mov  [esp+1884],ecx
        xor  edx, [esp+1524]
        mov  [esp+1880],edx
        xor  eax, [esp+1520]
        mov  [esp+1876],eax
        mov  [esp+1872],ebp
        mov  [esp+1864],esi
        mov  [esp+1860],ebx
        mov  [esp+1856],edi
        mov  ebp, [esp+1720]
        mov  eax, [esp+1720]
        mov  edi, [esp+1720]
        mov  ebp, [ebp+44]
        mov  [esp+72],ebp
        mov  ebx, [eax+40]
        mov  edx, [eax+36]
        mov  [esp+68],edx
        mov  ecx, [eax+32]
        mov  [esp+64],ecx
        mov  edx,ebp
        not  edx
        mov  esi,edx
        or  esi,ecx
        xor  esi, [eax+48]
        mov  eax,ebx
        or  eax,ebp
        xor  eax,edx
        mov  [esp+1508],eax
        mov  edi, [edi+60]
        mov  edx, [esp+68]
        mov  ebp, [esp+1720]
        xor  edx,eax
        mov  eax, [esp+1720]
        xor  [esp+1508],edi
        or  edx,ebx
        mov  [esp+1504],edx
        mov  edi,ecx
        xor  edi,edx
        mov  edx, [esp+64]
        or  edi, [esp+68]
        mov  ecx,esi
        mov  ebp, [ebp+56]
        xor  [esp+1504],ebp
        xor  edi, [eax+52]
        mov  eax,esi
        or  eax, [esp+1508]
        and  ecx,edi
        xor  ecx,eax
        mov  [esp+1500],ecx
        xor  [esp+1500],ebx
        mov  ebx,esi
        or  ebx, [esp+1504]
        xor  ebx, [esp+1508]
        mov  ebp,ebx
        or  ebp,ecx
        xor  ebp,esi
        mov  [esp+1496],ebp
        xor  [esp+1496],edx
        mov  edx,esi
        or  edx,edi
        mov  eax,esi
        and  eax, [esp+1504]
        xor  eax,edx
        mov  [esp+1492],eax
        mov  edx, [esp+72]
        or  eax,ecx
        mov  ecx, [esp+68]
        not  dword [esp+1492]
        xor  [esp+1492],edx
        mov  edx, [esp+1492]
        xor  ebx,ebp
        xor  ebx,eax
        not  ebx
        xor  ebx,ecx
        mov  [esp+1488],ebx
        mov  ebx, [esp+1496]
        mov  eax, [esp+1500]
        not  edx
        mov  [esp+1484],edx
        or  [esp+1484],ebx
        xor  [esp+1484],esi
        mov  esi, [esp+1508]
        or  eax, [esp+1492]
        xor  eax,edx
        mov  edx, [esp+1488]
        mov  ebp, [esp+1504]
        mov  [esp+1480],eax
        xor  edx,eax
        mov  eax, [esp+1488]
        xor  [esp+1480],esi
        or  edx, [esp+1500]
        mov  [esp+1476],edx
        mov  [esp+1472],ebx
        xor  [esp+1472],edx
        mov  edx, [esp+1480]
        xor  [esp+1476],ebp
        or  [esp+1472],eax
        xor  [esp+1472],edi
        mov  [esp+1852],edx
        mov  ecx, [esp+1476]
        mov  ebx, [esp+1472]
        mov  esi, [esp+1484]
        mov  edi, [esp+1492]
        mov  ebp, [esp+1500]
        mov  [esp+1828],eax
        mov  eax, [esp+1496]
        mov  edx, [esp+1724]
        mov  [esp+1848],ecx
        mov  ecx, [esp+1724]
        mov  [esp+1844],ebx
        mov  [esp+1840],esi
        mov  [esp+1836],edi
        mov  [esp+1832],ebp
        mov  [esp+1824],eax
        mov  edx, [edx+76]
        mov  [esp+60],edx
        mov  ebx, [ecx+72]
        mov  esi, [ecx+68]
        mov  [esp+56],esi
        mov  esi, [ecx+64]
        not  edx
        mov  ebp,edx
        or  ebp,esi
        xor  ebp, [ecx+80]
        mov  eax,ebx
        or  eax, [esp+60]
        xor  eax,edx
        mov  [esp+1468],eax
        mov  edi, [ecx+92]
        mov  edx, [esp+56]
        xor  [esp+1468],edi
        xor  edx,eax
        or  edx,ebx
        mov  [esp+1464],edx
        mov  eax, [ecx+88]
        xor  [esp+1464],eax
        mov  [esp+1460],esi
        xor  [esp+1460],edx
        mov  edx, [esp+56]
        or  [esp+1460],edx
        mov  ecx, [ecx+84]
        xor  [esp+1460],ecx
        mov  eax,ebp
        or  eax, [esp+1468]
        mov  ecx,ebp
        and  ecx, [esp+1460]
        xor  ecx,eax
        mov  [esp+1456],ecx
        xor  [esp+1456],ebx
        mov  edi,ebp
        or  edi, [esp+1464]
        xor  edi, [esp+1468]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1452],ebx
        xor  [esp+1452],esi
        mov  edx,ebp
        or  edx, [esp+1460]
        mov  eax,ebp
        and  eax, [esp+1464]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1456]
        not  esi
        xor  esi, [esp+60]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        or  ecx,esi
        mov  [esp+1932],esi
        mov  esi, [esp+1456]
        not  ebx
        xor  ebx, [esp+56]
        not  eax
        mov  edi,eax
        or  edi, [esp+1452]
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1456]
        mov  eax,edx
        xor  eax, [esp+1452]
        or  eax,ebx
        xor  ecx, [esp+1468]
        mov  [esp+1948],ecx
        xor  edx, [esp+1464]
        mov  [esp+1944],edx
        xor  eax, [esp+1460]
        mov  [esp+1940],eax
        xor  ebp,edi
        mov  [esp+1936],ebp
        mov  [esp+1928],esi
        mov  edi, [esp+1452]
        mov  ebp, [esp+1720]
        mov  eax, [esp+1720]
        mov  ecx, [esp+1720]
        mov  [esp+1920],edi
        mov  edi, [esp+1720]
        mov  [esp+1924],ebx
        mov  ebp, [ebp+76]
        mov  [esp+52],ebp
        mov  ebx, [eax+72]
        mov  edx, [eax+68]
        mov  [esp+48],edx
        mov  esi, [eax+64]
        mov  edx,ebp
        not  edx
        mov  ebp,edx
        or  ebp,esi
        xor  ebp, [eax+80]
        mov  eax,ebx
        or  eax, [esp+52]
        xor  eax,edx
        mov  [esp+1448],eax
        mov  ecx, [ecx+92]
        mov  edx, [esp+48]
        xor  [esp+1448],ecx
        xor  edx,eax
        or  edx,ebx
        mov  [esp+1444],edx
        mov  edi, [edi+88]
        mov  eax, [esp+48]
        mov  [esp+1440],esi
        xor  [esp+1440],edx
        mov  edx, [esp+1720]
        xor  [esp+1444],edi
        or  [esp+1440],eax
        mov  eax,ebp
        or  eax, [esp+1448]
        mov  ecx,ebp
        mov  edi,ebp
        mov  edx, [edx+84]
        xor  [esp+1440],edx
        and  ecx, [esp+1440]
        xor  ecx,eax
        mov  [esp+1436],ecx
        xor  [esp+1436],ebx
        or  edi, [esp+1444]
        xor  edi, [esp+1448]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1432],ebx
        xor  [esp+1432],esi
        mov  edx,ebp
        or  edx, [esp+1440]
        mov  eax,ebp
        and  eax, [esp+1444]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1436]
        not  esi
        xor  esi, [esp+52]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        not  eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+1448]
        mov  [esp+1916],ecx
        mov  ecx, [esp+1436]
        not  ebx
        xor  ebx, [esp+48]
        mov  edi,eax
        xor  edx,ebx
        or  edx, [esp+1436]
        mov  eax,edx
        xor  eax, [esp+1432]
        or  eax,ebx
        mov  [esp+1892],ebx
        mov  ebx, [esp+1432]
        mov  [esp+1900],esi
        mov  esi, [esp+1724]
        or  edi, [esp+1432]
        xor  ebp,edi
        mov  edi, [esp+1724]
        xor  edx, [esp+1444]
        mov  [esp+1912],edx
        xor  eax, [esp+1440]
        mov  [esp+1908],eax
        mov  [esp+1904],ebp
        mov  [esp+1896],ecx
        mov  [esp+1888],ebx
        mov  esi, [esi+108]
        mov  [esp+44],esi
        mov  ebx, [edi+104]
        mov  ebp, [edi+100]
        mov  [esp+40],ebp
        mov  esi, [edi+96]
        mov  edx, [esp+44]
        not  edx
        mov  ebp,edx
        or  ebp,esi
        xor  ebp, [edi+112]
        mov  eax,ebx
        or  eax, [esp+44]
        xor  eax,edx
        mov  [esp+1428],eax
        mov  edx, [edi+124]
        xor  [esp+1428],edx
        mov  edx, [esp+40]
        xor  edx,eax
        or  edx,ebx
        mov  [esp+1424],edx
        mov  ecx, [edi+120]
        mov  edi, [esp+40]
        mov  eax, [esp+1724]
        xor  [esp+1424],ecx
        mov  [esp+1420],esi
        xor  [esp+1420],edx
        or  [esp+1420],edi
        mov  ecx,ebp
        mov  edi,ebp
        or  edi, [esp+1424]
        xor  edi, [esp+1428]
        mov  edx,ebp
        mov  eax, [eax+116]
        xor  [esp+1420],eax
        mov  eax,ebp
        or  eax, [esp+1428]
        and  ecx, [esp+1420]
        xor  ecx,eax
        mov  [esp+1416],ecx
        xor  [esp+1416],ebx
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1412],ebx
        xor  [esp+1412],esi
        or  edx, [esp+1420]
        mov  eax,ebp
        and  eax, [esp+1424]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1416]
        not  esi
        xor  esi, [esp+44]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+40]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1412]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1416]
        mov  eax,edx
        xor  eax, [esp+1412]
        xor  ecx, [esp+1428]
        mov  [esp+2012],ecx
        xor  edx, [esp+1424]
        mov  [esp+2008],edx
        mov  edx, [esp+1416]
        mov  ecx, [esp+1412]
        or  eax,ebx
        mov  [esp+1988],ebx
        mov  ebx, [esp+1720]
        mov  [esp+1996],esi
        mov  esi, [esp+1720]
        xor  eax, [esp+1420]
        mov  [esp+2004],eax
        xor  ebp,edi
        mov  [esp+2000],ebp
        mov  [esp+1992],edx
        mov  [esp+1984],ecx
        mov  ebx, [ebx+108]
        mov  [esp+36],ebx
        mov  ecx, [esi+104]
        mov  edi, [esi+100]
        mov  [esp+32],edi
        mov  edi, [esi+96]
        mov  edx,ebx
        not  edx
        mov  ebp,edx
        or  ebp,edi
        xor  ebp, [esi+112]
        mov  eax,ecx
        or  eax,ebx
        xor  eax,edx
        mov  [esp+1408],eax
        mov  edx, [esi+124]
        xor  [esp+1408],edx
        mov  edx, [esp+32]
        xor  edx,eax
        or  edx,ecx
        mov  [esp+1404],edx
        mov  ebx, [esi+120]
        mov  esi, [esp+32]
        xor  [esp+1404],ebx
        mov  [esp+1400],edi
        xor  [esp+1400],edx
        or  [esp+1400],esi
        mov  eax, [esp+1720]
        mov  ebx,ebp
        mov  esi,ebp
        or  esi, [esp+1404]
        xor  esi, [esp+1408]
        mov  edx,ebp
        mov  eax, [eax+116]
        xor  [esp+1400],eax
        mov  eax,ebp
        or  eax, [esp+1408]
        and  ebx, [esp+1400]
        xor  ebx,eax
        mov  [esp+1396],ebx
        xor  [esp+1396],ecx
        mov  ecx,esi
        or  ecx,ebx
        xor  ecx,ebp
        mov  [esp+1392],ecx
        xor  [esp+1392],edi
        or  edx, [esp+1400]
        mov  eax,ebp
        and  eax, [esp+1404]
        xor  eax,edx
        mov  edi,eax
        or  eax,ebx
        mov  ebx, [esp+1396]
        not  edi
        xor  edi, [esp+36]
        xor  ecx,esi
        xor  ecx,eax
        not  ecx
        xor  ecx, [esp+32]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  ebx,edi
        xor  ebx,eax
        mov  edx,ebx
        xor  edx,ecx
        or  edx, [esp+1396]
        mov  eax,edx
        xor  edx, [esp+1404]
        mov  [esp+1976],edx
        mov  edx, [esp+1396]
        or  esi, [esp+1392]
        xor  eax, [esp+1392]
        or  eax,ecx
        xor  eax, [esp+1400]
        mov  [esp+1756],eax
        xor  ebx, [esp+1408]
        mov  [esp+1980],ebx
        mov  [esp+1972],eax
        xor  ebp,esi
        mov  [esp+1968],ebp
        mov  [esp+1964],edi
        mov  [esp+1960],edx
        mov  [esp+1956],ecx
        mov  ecx, [esp+1392]
        mov  esi, [esp+1496]
        mov  dword [esp+1704], csc_tabe
        mov  ebx, [esp+1704]
        mov  edi, [esp+1488]
        mov  ebp, [esp+1500]
        mov  eax, [esp+1484]
        mov  edx, [esp+1472]
        mov  [esp+1952],ecx
        mov  ecx, [esp+1476]
        xor  esi, [ebx+32]
        mov  [esp+1700],esi
        mov  esi, [esp+1492]
        xor  edi, [ebx+36]
        mov  [esp+1696],edi
        xor  ebp, [ebx+40]
        mov  [esp+1692],ebp
        xor  esi, [ebx+44]
        xor  eax, [ebx+48]
        mov  [esp+1688],eax
        xor  edx, [ebx+52]
        xor  ecx, [ebx+56]
        mov  [esp+1684],ecx
        mov  ecx, [esp+1480]
        mov  edi, [esp+1540]
        mov  ebp, [esp+1536]
        xor  ecx, [ebx+60]
        xor  edi, [ebx+28]
        mov  [esp+1668],edi
        mov  [esp+1380],ecx
        xor  [esp+1380],edi
        mov  [esp+1664],edx
        xor  [esp+1664],ebp
        mov  eax, [ebx+24]
        mov  ebx, [esp+1684]
        xor  [esp+1664],eax
        mov  edi, [esp+1664]
        mov  eax, [esp+1532]
        mov  ebp, [esp+1704]
        mov  [esp+1376],ebx
        xor  [esp+1376],edi
        mov  edi, [esp+1704]
        xor  eax, [ebp+20]
        mov  [esp+1372],eax
        xor  [esp+1372],edx
        mov  edx, [esp+1544]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1552]
        mov  edx, [esp+1696]
        xor  [esp+1660],ebx
        mov  ebx, [esp+1560]
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        xor  esi,eax
        mov  [esp+1388],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1548]
        xor  [esp+1656],esi
        xor  eax, [edi+4]
        mov  [esp+1384],eax
        xor  [esp+1384],edx
        mov  edx, [csc_tabe]
        mov  esi, [esp+1700]
        xor  ecx, [esp+1556]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+1388]
        mov  [esp+1680],eax
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1384]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+1388]
        xor  edx,ecx
        xor  [esp+1380],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1376],eax
        xor  eax,esi
        or  eax, [esp+1384]
        xor  [esp+1372],eax
        mov  eax,ebp
        or  eax, [esp+1380]
        mov  ecx,ebp
        and  ecx, [esp+1372]
        xor  ecx,eax
        mov  [esp+1368],ecx
        xor  [esp+1368],ebx
        mov  edi,ebp
        or  edi, [esp+1376]
        xor  edi, [esp+1380]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1364],ebx
        xor  [esp+1364],esi
        mov  edx,ebp
        or  edx, [esp+1372]
        mov  eax,ebp
        and  eax, [esp+1376]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1368]
        not  esi
        xor  esi, [esp+1388]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        not  eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+1380]
        mov  [esp+1852],ecx
        mov  ecx, [esp+1368]
        not  ebx
        xor  ebx, [esp+1384]
        mov  edi,eax
        xor  edx,ebx
        or  edx, [esp+1368]
        mov  eax,edx
        xor  eax, [esp+1364]
        or  eax,ebx
        mov  [esp+1828],ebx
        mov  ebx, [esp+1364]
        mov  [esp+1836],esi
        mov  esi, [esp+1684]
        or  edi, [esp+1364]
        xor  ebp,edi
        mov  edi, [esp+1688]
        mov  [esp+1840],ebp
        mov  ebp, [esp+1692]
        xor  eax, [esp+1372]
        mov  [esp+1844],eax
        mov  eax, [esp+1700]
        xor  edx, [esp+1376]
        mov  [esp+1848],edx
        mov  edx, [esp+1656]
        mov  [esp+1832],ecx
        mov  [esp+1824],ebx
        xor  esi, [esp+1668]
        xor  edi, [esp+1672]
        xor  ebp, [esp+1676]
        mov  [esp+1360],ebp
        xor  eax, [esp+1680]
        mov  [esp+1356],eax
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx, [esp+1360]
        xor  edx,eax
        xor  esi,edx
        mov  [esp+1352],esi
        mov  eax, [esp+1356]
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  ebx, [esp+1656]
        or  eax, [esp+1656]
        mov  [esp+1348],eax
        xor  [esp+1348],edx
        xor  eax, [esp+1652]
        or  eax, [esp+1356]
        xor  edi,eax
        mov  [esp+1344],edi
        mov  eax,ebp
        or  eax,esi
        mov  ecx,ebp
        and  ecx,edi
        mov  edi,ebp
        or  edi, [esp+1348]
        xor  edi,esi
        mov  esi, [esp+1652]
        xor  ecx,eax
        mov  [esp+1340],ecx
        xor  [esp+1340],ebx
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1336],ebx
        xor  [esp+1336],esi
        mov  edx,ebp
        or  edx, [esp+1344]
        mov  eax,ebp
        and  eax, [esp+1348]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1340]
        not  esi
        xor  esi, [esp+1360]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1356]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1336]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1340]
        mov  eax,edx
        xor  eax, [esp+1336]
        or  eax,ebx
        xor  ecx, [esp+1352]
        mov  [esp+1788],ecx
        xor  edx, [esp+1348]
        mov  [esp+1784],edx
        xor  eax, [esp+1344]
        mov  [esp+1780],eax
        xor  ebp,edi
        mov  [esp+1776],ebp
        mov  [esp+1772],esi
        mov  edi, [esp+1340]
        mov  ebp, [esp+1336]
        mov  edx, [esp+1952]
        mov  dword [esp+1704], csc_tabe+64
        mov  eax, [esp+1704]
        mov  ecx, [esp+1956]
        mov  [esp+1764],ebx
        mov  ebx, [esp+1960]
        mov  esi, [esp+1964]
        mov  [esp+1768],edi
        mov  edi, [esp+1968]
        mov  [esp+1760],ebp
        mov  ebp, [esp+1976]
        xor  edx, [eax+32]
        mov  [esp+1700],edx
        mov  edx, [esp+1972]
        xor  ecx, [eax+36]
        mov  [esp+1696],ecx
        mov  ecx, [esp+1980]
        xor  ebx, [eax+40]
        mov  [esp+1692],ebx
        mov  ebx, [esp+1916]
        xor  esi, [eax+44]
        xor  edi, [eax+48]
        mov  [esp+1688],edi
        xor  edx, [eax+52]
        xor  ebp, [eax+56]
        mov  [esp+1684],ebp
        xor  ecx, [eax+60]
        xor  ebx, [eax+28]
        mov  [esp+1668],ebx
        mov  [esp+1324],ecx
        xor  [esp+1324],ebx
        mov  [esp+1664],edx
        mov  edi, [esp+1912]
        xor  [esp+1664],edi
        mov  ebp, [eax+24]
        mov  eax, [esp+1684]
        xor  [esp+1664],ebp
        mov  ebx, [esp+1664]
        mov  ebp, [esp+1908]
        mov  edi, [esp+1704]
        mov  [esp+1320],eax
        mov  eax, [esp+1904]
        xor  [esp+1320],ebx
        mov  ebx, [esp+1900]
        xor  ebp, [edi+20]
        mov  [esp+1672],ebp
        mov  [esp+1316],ebp
        xor  [esp+1316],edx
        mov  [esp+1660],esi
        xor  [esp+1660],eax
        mov  edx, [edi+16]
        mov  ebp, [esp+1688]
        xor  [esp+1660],edx
        xor  ebx, [edi+12]
        mov  [esp+1332],ebx
        xor  [esp+1332],esi
        mov  esi, [esp+1696]
        xor  ebp, [esp+1660]
        mov  [esp+1676],ebx
        mov  [esp+1656],esi
        mov  edi, [esp+1896]
        mov  eax, [esp+1704]
        mov  ebx, [esp+1692]
        mov  esi, [esp+1892]
        mov  edx, [esp+1704]
        xor  [esp+1656],edi
        mov  edi, [esp+1696]
        xor  ecx, [esp+1888]
        mov  eax, [eax+8]
        xor  [esp+1656],eax
        mov  eax,[csc_tabe+64]
        xor  esi, [edx+4]
        mov  [esp+1680],esi
        mov  [esp+1328],esi
        mov  esi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+1332]
        xor  [esp+1328],edi
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1328]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+1332]
        xor  edx,ecx
        xor  [esp+1324],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1320],eax
        xor  eax,esi
        or  eax, [esp+1328]
        xor  [esp+1316],eax
        mov  eax,ebp
        or  eax, [esp+1324]
        mov  ecx,ebp
        and  ecx, [esp+1316]
        xor  ecx,eax
        mov  [esp+1312],ecx
        xor  [esp+1312],ebx
        mov  edi,ebp
        or  edi, [esp+1320]
        xor  edi, [esp+1324]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1308],ebx
        xor  [esp+1308],esi
        mov  edx,ebp
        or  edx, [esp+1316]
        mov  eax,ebp
        and  eax, [esp+1320]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1312]
        not  esi
        xor  esi, [esp+1332]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1328]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1312]
        mov  eax,edx
        xor  edx, [esp+1320]
        mov  [esp+1976],edx
        mov  edx, [esp+1312]
        xor  ecx, [esp+1324]
        mov  [esp+1980],ecx
        mov  ecx, [esp+1308]
        xor  eax, [esp+1308]
        or  eax,ebx
        mov  [esp+1956],ebx
        mov  ebx, [esp+1684]
        mov  [esp+1964],esi
        mov  esi, [esp+1688]
        or  edi, [esp+1308]
        xor  ebp,edi
        mov  edi, [esp+1692]
        xor  eax, [esp+1316]
        mov  [esp+1972],eax
        mov  [esp+1968],ebp
        mov  [esp+1960],edx
        mov  [esp+1952],ecx
        xor  ebx, [esp+1668]
        xor  esi, [esp+1672]
        xor  edi, [esp+1676]
        mov  [esp+1304],edi
        mov  ebp, [esp+1700]
        mov  edx, [esp+1656]
        xor  ebp, [esp+1680]
        mov  [esp+1300],ebp
        mov  eax,edi
        not  eax
        mov  ebp,eax
        or  edx,edi
        xor  edx,eax
        mov  eax, [esp+1300]
        xor  ebx,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+1296],ebx
        or  eax, [esp+1656]
        mov  [esp+1292],eax
        xor  eax, [esp+1652]
        or  eax, [esp+1300]
        xor  esi,eax
        mov  eax,ebp
        or  eax,ebx
        mov  ebx, [esp+1656]
        mov  [esp+1288],esi
        mov  ecx,ebp
        and  ecx,esi
        mov  esi, [esp+1652]
        xor  [esp+1292],edx
        xor  ecx,eax
        mov  [esp+1284],ecx
        xor  [esp+1284],ebx
        mov  edi,ebp
        or  edi, [esp+1292]
        xor  edi, [esp+1296]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1280],ebx
        xor  [esp+1280],esi
        mov  edx,ebp
        or  edx, [esp+1288]
        mov  eax,ebp
        and  eax, [esp+1292]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1284]
        not  esi
        xor  esi, [esp+1304]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1300]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1280]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1284]
        mov  eax,edx
        xor  eax, [esp+1280]
        xor  ecx, [esp+1296]
        mov  [esp+1916],ecx
        xor  ebp,edi
        mov  edi, [esp+1284]
        mov  [esp+1904],ebp
        mov  ebp, [esp+1280]
        xor  edx, [esp+1292]
        mov  [esp+1912],edx
        mov  edx, [esp+1856]
        or  eax,ebx
        xor  eax, [esp+1288]
        mov  [esp+1908],eax
        mov  dword [esp+1704], csc_tabe+128
        mov  eax, [esp+1704]
        mov  ecx, [esp+1860]
        mov  [esp+1892],ebx
        mov  ebx, [esp+1864]
        mov  [esp+1900],esi
        mov  esi, [esp+1868]
        mov  [esp+1896],edi
        mov  edi, [esp+1872]
        mov  [esp+1888],ebp
        mov  ebp, [esp+1880]
        xor  edx, [eax+32]
        mov  [esp+1700],edx
        mov  edx, [esp+1876]
        xor  ecx, [eax+36]
        mov  [esp+1696],ecx
        mov  ecx, [esp+1884]
        xor  ebx, [eax+40]
        mov  [esp+1692],ebx
        mov  ebx, [esp+1820]
        xor  esi, [eax+44]
        xor  edi, [eax+48]
        mov  [esp+1688],edi
        xor  edx, [eax+52]
        xor  ebp, [eax+56]
        mov  [esp+1684],ebp
        xor  ecx, [eax+60]
        xor  ebx, [eax+28]
        mov  [esp+1668],ebx
        mov  edi, [esp+1816]
        mov  [esp+1268],ecx
        xor  [esp+1268],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [eax+24]
        mov  eax, [esp+1684]
        xor  [esp+1664],ebp
        mov  ebx, [esp+1664]
        mov  ebp, [esp+1812]
        mov  edi, [esp+1704]
        mov  [esp+1264],eax
        mov  eax, [esp+1808]
        xor  [esp+1264],ebx
        mov  ebx, [esp+1804]
        xor  ebp, [edi+20]
        mov  [esp+1672],ebp
        mov  [esp+1260],ebp
        xor  [esp+1260],edx
        mov  [esp+1660],esi
        xor  [esp+1660],eax
        mov  edx, [edi+16]
        mov  ebp, [esp+1688]
        xor  [esp+1660],edx
        xor  ebp, [esp+1660]
        xor  ebx, [edi+12]
        mov  [esp+1676],ebx
        mov  [esp+1276],ebx
        xor  [esp+1276],esi
        mov  esi, [esp+1696]
        mov  edi, [esp+1800]
        mov  eax, [esp+1704]
        mov  ebx, [esp+1692]
        mov  [esp+1656],esi
        mov  esi, [esp+1796]
        mov  edx, [esp+1704]
        xor  [esp+1656],edi
        mov  edi, [esp+1696]
        xor  ecx, [esp+1792]
        mov  eax, [eax+8]
        xor  [esp+1656],eax
        mov  eax, [csc_tabe+128]
        xor  esi, [edx+4]
        mov  [esp+1680],esi
        mov  [esp+1272],esi
        mov  esi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+1276]
        xor  [esp+1272],edi
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1272]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+1276]
        xor  edx,ecx
        xor  [esp+1268],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1264],eax
        xor  eax,esi
        or  eax, [esp+1272]
        xor  [esp+1260],eax
        mov  eax,ebp
        or  eax, [esp+1268]
        mov  ecx,ebp
        and  ecx, [esp+1260]
        xor  ecx,eax
        mov  [esp+1256],ecx
        xor  [esp+1256],ebx
        mov  edi,ebp
        or  edi, [esp+1264]
        xor  edi, [esp+1268]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1252],ebx
        xor  [esp+1252],esi
        mov  edx,ebp
        or  edx, [esp+1260]
        mov  eax,ebp
        and  eax, [esp+1264]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1256]
        not  esi
        xor  esi, [esp+1276]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1272]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1256]
        mov  eax,edx
        xor  edx, [esp+1264]
        mov  [esp+1880],edx
        mov  edx, [esp+1256]
        xor  ecx, [esp+1268]
        mov  [esp+1884],ecx
        mov  ecx, [esp+1252]
        xor  eax, [esp+1252]
        or  eax,ebx
        mov  [esp+1860],ebx
        mov  ebx, [esp+1684]
        mov  [esp+1868],esi
        mov  esi, [esp+1688]
        or  edi, [esp+1252]
        xor  ebp,edi
        mov  edi, [esp+1692]
        xor  eax, [esp+1260]
        mov  [esp+1876],eax
        mov  [esp+1872],ebp
        mov  [esp+1864],edx
        mov  [esp+1856],ecx
        xor  ebx, [esp+1668]
        xor  esi, [esp+1672]
        xor  edi, [esp+1676]
        mov  [esp+1248],edi
        mov  ebp, [esp+1700]
        mov  edx, [esp+1656]
        xor  ebp, [esp+1680]
        mov  [esp+1244],ebp
        mov  eax,edi
        not  eax
        mov  ebp,eax
        or  edx,edi
        xor  edx,eax
        mov  eax, [esp+1244]
        xor  ebx,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+1240],ebx
        or  eax, [esp+1656]
        mov  [esp+1236],eax
        xor  eax, [esp+1652]
        or  eax, [esp+1244]
        xor  esi,eax
        mov  eax,ebp
        or  eax,ebx
        mov  ebx, [esp+1656]
        mov  [esp+1232],esi
        mov  ecx,ebp
        and  ecx,esi
        mov  esi, [esp+1652]
        xor  [esp+1236],edx
        xor  ecx,eax
        mov  [esp+1228],ecx
        xor  [esp+1228],ebx
        mov  edi,ebp
        or  edi, [esp+1236]
        xor  edi, [esp+1240]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1224],ebx
        xor  [esp+1224],esi
        mov  edx,ebp
        or  edx, [esp+1232]
        mov  eax,ebp
        and  eax, [esp+1236]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1228]
        not  esi
        xor  esi, [esp+1248]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1244]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1224]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1228]
        mov  eax,edx
        xor  eax, [esp+1224]
        xor  ecx, [esp+1240]
        mov  [esp+1820],ecx
        xor  ebp,edi
        mov  edi, [esp+1228]
        mov  [esp+1808],ebp
        mov  ebp, [esp+1224]
        xor  edx, [esp+1236]
        mov  [esp+1816],edx
        mov  edx, [esp+1984]
        or  eax,ebx
        xor  eax, [esp+1232]
        mov  [esp+1812],eax
        mov  dword [esp+1704], csc_tabe+192
        mov  eax, [esp+1704]
        mov  ecx, [esp+1988]
        mov  [esp+1796],ebx
        mov  ebx, [esp+1992]
        mov  [esp+1804],esi
        mov  esi, [esp+1996]
        mov  [esp+1800],edi
        mov  edi, [esp+2000]
        mov  [esp+1792],ebp
        mov  ebp, [esp+2008]
        xor  edx, [eax+32]
        mov  [esp+1700],edx
        mov  edx, [esp+2004]
        xor  ecx, [eax+36]
        mov  [esp+1696],ecx
        mov  ecx, [esp+2012]
        xor  ebx, [eax+40]
        mov  [esp+1692],ebx
        mov  ebx, [esp+1948]
        xor  esi, [eax+44]
        xor  edi, [eax+48]
        mov  [esp+1688],edi
        xor  edx, [eax+52]
        xor  ebp, [eax+56]
        mov  [esp+1684],ebp
        xor  ecx, [eax+60]
        xor  ebx, [eax+28]
        mov  [esp+1668],ebx
        mov  edi, [esp+1944]
        mov  [esp+1212],ecx
        xor  [esp+1212],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [eax+24]
        mov  eax, [esp+1684]
        xor  [esp+1664],ebp
        mov  ebx, [esp+1664]
        mov  ebp, [esp+1940]
        mov  edi, [esp+1704]
        mov  [esp+1208],eax
        mov  eax, [esp+1936]
        xor  [esp+1208],ebx
        mov  ebx, [esp+1932]
        xor  ebp, [edi+20]
        mov  [esp+1672],ebp
        mov  [esp+1204],ebp
        xor  [esp+1204],edx
        mov  [esp+1660],esi
        xor  [esp+1660],eax
        mov  edx, [edi+16]
        mov  ebp, [esp+1688]
        xor  [esp+1660],edx
        xor  ebp, [esp+1660]
        xor  ebx, [edi+12]
        mov  [esp+1676],ebx
        mov  [esp+1220],ebx
        xor  [esp+1220],esi
        mov  esi, [esp+1696]
        mov  edi, [esp+1928]
        mov  eax, [esp+1704]
        mov  ebx, [esp+1692]
        mov  [esp+1656],esi
        mov  esi, [esp+1924]
        mov  edx, [esp+1704]
        xor  [esp+1656],edi
        mov  edi, [esp+1696]
        xor  ecx, [esp+1920]
        mov  eax, [eax+8]
        xor  [esp+1656],eax
        mov  eax, [csc_tabe+192]
        xor  esi, [edx+4]
        mov  [esp+1680],esi
        mov  [esp+1216],esi
        mov  esi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+1220]
        xor  [esp+1216],edi
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1216]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+1220]
        xor  edx,ecx
        xor  [esp+1212],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1208],eax
        xor  eax,esi
        or  eax, [esp+1216]
        xor  [esp+1204],eax
        mov  eax,ebp
        or  eax, [esp+1212]
        mov  ecx,ebp
        and  ecx, [esp+1204]
        xor  ecx,eax
        mov  [esp+1200],ecx
        xor  [esp+1200],ebx
        mov  edi,ebp
        or  edi, [esp+1208]
        xor  edi, [esp+1212]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1196],ebx
        xor  [esp+1196],esi
        mov  edx,ebp
        or  edx, [esp+1204]
        mov  eax,ebp
        and  eax, [esp+1208]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1200]
        not  esi
        xor  esi, [esp+1220]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1216]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1200]
        mov  eax,edx
        xor  edx, [esp+1208]
        mov  [esp+2008],edx
        mov  edx, [esp+1200]
        xor  ecx, [esp+1212]
        mov  [esp+2012],ecx
        mov  ecx, [esp+1196]
        xor  eax, [esp+1196]
        or  eax,ebx
        mov  [esp+1988],ebx
        mov  ebx, [esp+1684]
        mov  [esp+1996],esi
        mov  esi, [esp+1688]
        or  edi, [esp+1196]
        xor  eax, [esp+1204]
        mov  [esp+2004],eax
        xor  ebp,edi
        mov  [esp+2000],ebp
        mov  [esp+1992],edx
        mov  [esp+1984],ecx
        xor  ebx, [esp+1668]
        xor  esi, [esp+1672]
        mov  [esp+1168],esi
        mov  edi, [esp+1692]
        mov  ebp, [esp+1700]
        mov  edx, [esp+1656]
        xor  edi, [esp+1676]
        mov  [esp+1192],edi
        xor  ebp, [esp+1680]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+1652]
        xor  esi, [esp+1660]
        or  edx,edi
        xor  edx,eax
        xor  ebx,edx
        mov  eax,ebp
        xor  eax,edx
        or  eax, [esp+1656]
        mov  edi,eax
        xor  eax, [esp+1652]
        or  eax,ebp
        xor  [esp+1168],eax
        mov  eax,esi
        or  eax,ebx
        mov  ecx,esi
        and  ecx, [esp+1168]
        xor  ecx,eax
        xor  [esp+1656],ecx
        mov  eax, [esp+1656]
        mov  edx, [esp+1652]
        mov  [esp+1180],ebp
        mov  [esp+1172],ebx
        xor  edi, [esp+1664]
        mov  [esp+1188],eax
        mov  ebx,esi
        or  ebx,edi
        xor  ebx, [esp+1172]
        mov  ebp,ebx
        or  ebp,ecx
        xor  ebp,esi
        mov  [esp+1176],ebp
        xor  [esp+1176],edx
        mov  eax,esi
        or  eax, [esp+1168]
        mov  edx,esi
        and  edx,edi
        xor  edx,eax
        mov  eax,edx
        not  eax
        xor  [esp+1192],eax
        mov  eax, [esp+1192]
        or  edx,ecx
        mov  ecx, [esp+1180]
        xor  ebx,ebp
        xor  ebx,edx
        not  ebx
        xor  ebx,ecx
        mov  [esp+1184],ebx
        mov  ebx, [esp+1188]
        mov  [esp+1164],eax
        mov  [esp+1160],ebx
        mov  ebp, [esp+1176]
        mov  edx,eax
        not  edx
        mov  [esp+1156],edx
        mov  eax,ebx
        or  eax, [esp+1164]
        xor  eax,edx
        mov  edx, [esp+1172]
        mov  [esp+1152],eax
        xor  [esp+1152],edx
        mov  edx, [esp+1184]
        mov  ecx, [esp+1184]
        xor  edx,eax
        or  edx,ebx
        mov  ebx, [esp+1168]
        or  [esp+1156],ebp
        xor  [esp+1156],esi
        mov  esi, [esp+1152]
        mov  [esp+1148],edx
        xor  [esp+1148],edi
        mov  edi, [esp+1148]
        mov  [esp+1144],ebp
        xor  [esp+1144],edx
        or  [esp+1144],ecx
        xor  [esp+1144],ebx
        mov  ebp, [esp+1144]
        mov  eax, [esp+1156]
        mov  edx, [esp+1164]
        mov  [esp+1948],esi
        mov  [esp+1944],edi
        mov  [esp+1940],ebp
        mov  [esp+1936],eax
        mov  [esp+1932],edx
        mov  ecx, [esp+1160]
        mov  ebx, [esp+1184]
        mov  esi, [esp+1176]
        mov  ebp, [esp+1888]
        add  dword [esp+1704],BYTE 64
        mov  edi, [esp+1704]
        mov  eax, [esp+1892]
        mov  edx, [esp+1896]
        mov  [esp+1920],esi
        mov  esi, [esp+1900]
        mov  [esp+1928],ecx
        mov  ecx, [esp+1904]
        mov  [esp+1924],ebx
        mov  ebx, [esp+1912]
        xor  ebp, [edi+32]
        mov  [esp+1700],ebp
        xor  eax, [edi+36]
        mov  [esp+1696],eax
        xor  edx, [edi+40]
        mov  [esp+1692],edx
        mov  edx, [esp+1908]
        xor  esi, [edi+44]
        xor  ecx, [edi+48]
        mov  [esp+1688],ecx
        mov  ecx, [esp+1916]
        mov  ebp, [esp+1788]
        xor  edx, [edi+52]
        xor  ebx, [edi+56]
        mov  [esp+1684],ebx
        xor  ecx, [edi+60]
        xor  ebp, [edi+28]
        mov  [esp+1668],ebp
        mov  [esp+1132],ecx
        mov  eax, [esp+1784]
        xor  [esp+1132],ebp
        mov  [esp+1664],edx
        xor  [esp+1664],eax
        mov  ebx, [edi+24]
        mov  edi, [esp+1684]
        xor  [esp+1664],ebx
        mov  ebp, [esp+1664]
        mov  ebx, [esp+1780]
        mov  eax, [esp+1704]
        mov  [esp+1128],edi
        mov  edi, [esp+1776]
        xor  [esp+1128],ebp
        xor  ebx, [eax+20]
        mov  [esp+1672],ebx
        mov  [esp+1124],ebx
        xor  [esp+1124],edx
        mov  [esp+1660],esi
        xor  [esp+1660],edi
        mov  ebp, [eax+16]
        xor  [esp+1660],ebp
        mov  ebp, [esp+1688]
        mov  edx, [esp+1772]
        xor  ebp, [esp+1660]
        xor  edx, [eax+12]
        mov  [esp+1676],edx
        mov  [esp+1140],edx
        mov  ebx, [esp+1696]
        xor  [esp+1140],esi
        mov  esi, [esp+1768]
        mov  [esp+1656],ebx
        xor  [esp+1656],esi
        mov  edi, [eax+8]
        mov  ebx, [esp+1692]
        mov  edx, [esp+1764]
        mov  esi, [esp+1696]
        xor  [esp+1656],edi
        xor  edx, [eax+4]
        mov  [esp+1680],edx
        mov  [esp+1136],edx
        xor  [esp+1136],esi
        mov  edi, [eax]
        mov  esi, [esp+1700]
        xor  ecx, [esp+1760]
        xor  ecx,edi
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+1140]
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1136]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+1140]
        xor  edx,ecx
        xor  [esp+1132],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1128],eax
        xor  eax,esi
        or  eax, [esp+1136]
        xor  [esp+1124],eax
        mov  eax,ebp
        or  eax, [esp+1132]
        mov  ecx,ebp
        and  ecx, [esp+1124]
        xor  ecx,eax
        mov  [esp+1120],ecx
        xor  [esp+1120],ebx
        mov  edi,ebp
        or  edi, [esp+1128]
        xor  edi, [esp+1132]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1116],ebx
        xor  [esp+1116],esi
        mov  edx,ebp
        or  edx, [esp+1124]
        mov  eax,ebp
        and  eax, [esp+1128]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1120]
        not  esi
        xor  esi, [esp+1140]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1116]
        xor  ebp,edi
        mov  [esp+1904],ebp
        mov  ebp, [esp+1120]
        not  ebx
        xor  ebx, [esp+1136]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1120]
        mov  eax,edx
        xor  eax, [esp+1116]
        or  eax,ebx
        xor  eax, [esp+1124]
        mov  [esp+1908],eax
        mov  eax, [esp+1116]
        xor  edx, [esp+1128]
        mov  [esp+1912],edx
        mov  edx, [esp+1684]
        xor  ecx, [esp+1132]
        mov  [esp+1916],ecx
        mov  ecx, [esp+1688]
        mov  [esp+1892],ebx
        mov  ebx, [esp+1692]
        mov  [esp+1900],esi
        mov  [esp+1896],ebp
        mov  [esp+1888],eax
        xor  edx, [esp+1668]
        mov  [esp+1104],edx
        xor  ecx, [esp+1672]
        xor  ebx, [esp+1676]
        mov  [esp+1112],ebx
        mov  esi, [esp+1700]
        mov  edx, [esp+1656]
        mov  edi, [esp+1664]
        xor  esi, [esp+1680]
        mov  eax,ebx
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx,ebx
        xor  edx,eax
        xor  [esp+1104],edx
        mov  eax,esi
        xor  eax,edx
        or  eax, [esp+1656]
        mov  [esp+1100],eax
        xor  eax, [esp+1652]
        or  eax,esi
        xor  ecx,eax
        mov  [esp+1096],ecx
        mov  eax,ebp
        or  eax, [esp+1104]
        mov  ecx,ebp
        and  ecx, [esp+1096]
        xor  ecx,eax
        mov  eax, [esp+1656]
        mov  edx, [esp+1652]
        mov  [esp+1108],esi
        xor  [esp+1100],edi
        mov  [esp+1092],ecx
        xor  [esp+1092],eax
        mov  edi,ebp
        or  edi, [esp+1100]
        xor  edi, [esp+1104]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1088],ebx
        xor  [esp+1088],edx
        mov  edx,ebp
        or  edx, [esp+1096]
        mov  eax,ebp
        and  eax, [esp+1100]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1092]
        not  esi
        xor  esi, [esp+1112]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1108]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1088]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1092]
        mov  eax,edx
        xor  eax, [esp+1088]
        xor  ecx, [esp+1104]
        mov  [esp+1788],ecx
        mov  ecx, [esp+1092]
        or  eax,ebx
        mov  [esp+1764],ebx
        mov  ebx, [esp+1088]
        xor  ebp,edi
        mov  edi, [esp+1176]
        mov  [esp+1772],esi
        add  dword [esp+1704],BYTE 64
        mov  esi, [esp+1704]
        mov  [esp+1776],ebp
        mov  ebp, [esp+1184]
        xor  eax, [esp+1096]
        mov  [esp+1780],eax
        mov  eax, [esp+1160]
        xor  edx, [esp+1100]
        mov  [esp+1784],edx
        mov  edx, [esp+1704]
        mov  [esp+1768],ecx
        mov  ecx, [esp+1156]
        mov  [esp+1760],ebx
        mov  ebx, [esp+1704]
        xor  edi, [esi+32]
        mov  [esp+1700],edi
        xor  ebp, [esi+36]
        mov  [esp+1696],ebp
        xor  eax, [esi+40]
        mov  esi, [esp+1164]
        mov  [esp+1692],eax
        xor  esi, [edx+44]
        xor  ecx, [edx+48]
        mov  edx, [esp+1144]
        mov  edi, [esp+1148]
        mov  [esp+1688],ecx
        xor  edx, [ebx+52]
        xor  edi, [ebx+56]
        mov  [esp+1684],edi
        mov  ecx, [esp+1152]
        mov  ebp, [esp+1820]
        mov  eax, [esp+1816]
        xor  ecx, [ebx+60]
        xor  ebp, [ebx+28]
        mov  [esp+1668],ebp
        mov  [esp+1076],ecx
        xor  [esp+1076],ebp
        mov  [esp+1664],edx
        xor  [esp+1664],eax
        mov  ebx, [ebx+24]
        xor  [esp+1664],ebx
        mov  [esp+1072],edi
        mov  edi, [esp+1664]
        mov  eax, [esp+1812]
        mov  ebp, [esp+1704]
        xor  [esp+1072],edi
        mov  edi, [esp+1704]
        xor  eax, [ebp+20]
        mov  [esp+1068],eax
        xor  [esp+1068],edx
        mov  edx, [esp+1808]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1804]
        xor  [esp+1660],ebx
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        mov  edx, [esp+1696]
        mov  ebx, [esp+1800]
        mov  [esp+1084],eax
        xor  [esp+1084],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1796]
        xor  [esp+1656],esi
        xor  eax, [edi+4]
        mov  [esp+1680],eax
        mov  [esp+1080],eax
        xor  [esp+1080],edx
        mov  edx, [edi]
        mov  esi, [esp+1700]
        xor  ecx, [esp+1792]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+1084]
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+1080]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+1084]
        xor  edx,ecx
        xor  [esp+1076],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+1072],eax
        xor  eax,esi
        or  eax, [esp+1080]
        xor  [esp+1068],eax
        mov  eax,ebp
        or  eax, [esp+1076]
        mov  ecx,ebp
        and  ecx, [esp+1068]
        xor  ecx,eax
        mov  [esp+1064],ecx
        xor  [esp+1064],ebx
        mov  edi,ebp
        or  edi, [esp+1072]
        xor  edi, [esp+1076]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1060],ebx
        xor  [esp+1060],esi
        mov  edx,ebp
        or  edx, [esp+1068]
        mov  eax,ebp
        and  eax, [esp+1072]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1064]
        not  esi
        xor  esi, [esp+1084]
        xor  ebx,edi
        xor  ebx,eax
        mov  eax,esi
        not  eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+1076]
        mov  [esp+1948],ecx
        mov  ecx, [esp+1064]
        not  ebx
        xor  ebx, [esp+1080]
        mov  edi,eax
        xor  edx,ebx
        or  edx, [esp+1064]
        mov  eax,edx
        xor  eax, [esp+1060]
        or  eax,ebx
        mov  [esp+1924],ebx
        mov  ebx, [esp+1060]
        mov  [esp+1932],esi
        mov  esi, [esp+1684]
        or  edi, [esp+1060]
        xor  ebp,edi
        mov  edi, [esp+1688]
        mov  [esp+1936],ebp
        mov  ebp, [esp+1692]
        xor  edx, [esp+1072]
        mov  [esp+1944],edx
        xor  eax, [esp+1068]
        mov  [esp+1940],eax
        mov  [esp+1928],ecx
        mov  [esp+1920],ebx
        xor  esi, [esp+1668]
        xor  edi, [esp+1672]
        xor  ebp, [esp+1676]
        mov  [esp+1056],ebp
        mov  eax, [esp+1700]
        mov  edx, [esp+1656]
        xor  eax, [esp+1680]
        mov  [esp+1052],eax
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  edx, [esp+1056]
        xor  edx,eax
        mov  eax, [esp+1052]
        xor  esi,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  ebx, [esp+1656]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+1048],esi
        or  eax, [esp+1656]
        mov  [esp+1044],eax
        xor  [esp+1044],edx
        xor  eax, [esp+1652]
        or  eax, [esp+1052]
        xor  edi,eax
        mov  [esp+1040],edi
        mov  eax,ebp
        or  eax,esi
        mov  ecx,ebp
        and  ecx,edi
        mov  edi,ebp
        or  edi, [esp+1044]
        xor  edi,esi
        mov  esi, [esp+1652]
        xor  ecx,eax
        mov  [esp+1036],ecx
        xor  [esp+1036],ebx
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+1032],ebx
        xor  [esp+1032],esi
        mov  edx,ebp
        or  edx, [esp+1040]
        mov  eax,ebp
        and  eax, [esp+1044]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+1036]
        not  esi
        xor  esi, [esp+1056]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1052]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+1032]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+1036]
        mov  eax,edx
        xor  eax, [esp+1032]
        xor  ecx, [esp+1048]
        mov  [esp+1820],ecx
        xor  ebp,edi
        mov  edi, [esp+1036]
        mov  [esp+1808],ebp
        mov  ebp, [esp+1032]
        xor  edx, [esp+1044]
        mov  [esp+1816],edx
        mov  edx, [esp+1952]
        or  eax,ebx
        xor  eax, [esp+1040]
        mov  [esp+1812],eax
        add  dword [esp+1704],BYTE 64
        mov  eax, [esp+1704]
        mov  ecx, [esp+1956]
        mov  [esp+1796],ebx
        mov  ebx, [esp+1960]
        mov  [esp+1804],esi
        mov  esi, [esp+1964]
        mov  [esp+1800],edi
        mov  edi, [esp+1968]
        mov  [esp+1792],ebp
        mov  ebp, [esp+1976]
        xor  edx, [eax+32]
        mov  [esp+1700],edx
        mov  edx, [esp+1972]
        xor  ecx, [eax+36]
        mov  [esp+1696],ecx
        mov  ecx, [esp+1980]
        xor  ebx, [eax+40]
        mov  [esp+1692],ebx
        mov  ebx, [esp+1852]
        xor  esi, [eax+44]
        xor  edi, [eax+48]
        mov  [esp+1688],edi
        xor  edx, [eax+52]
        xor  ebp, [eax+56]
        mov  [esp+1684],ebp
        xor  ecx, [eax+60]
        xor  ebx, [eax+28]
        mov  [esp+1668],ebx
        mov  edi, [esp+1848]
        mov  [esp+1024],ecx
        xor  [esp+1024],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [eax+24]
        mov  eax, [esp+1684]
        xor  [esp+1664],ebp
        mov  ebx, [esp+1664]
        mov  ebp, [esp+1844]
        mov  edi, [esp+1704]
        mov  [esp+1020],eax
        mov  eax, [esp+1840]
        xor  [esp+1020],ebx
        mov  ebx, [esp+1688]
        xor  ebp, [edi+20]
        mov  [esp+1672],ebp
        mov  [esp+1016],ebp
        xor  [esp+1016],edx
        mov  [esp+1660],esi
        xor  [esp+1660],eax
        mov  edx, [edi+16]
        xor  [esp+1660],edx
        mov  edi, [esp+1660]
        mov  eax, [esp+1836]
        mov  ebp, [esp+1704]
        mov  [esp+1028],ebx
        xor  [esp+1028],edi
        xor  eax, [ebp+12]
        mov  [esp+1676],eax
        mov  edx, [esp+1696]
        mov  ebx, [esp+1832]
        mov  ebp,eax
        xor  ebp,esi
        mov  esi, [esp+1704]
        mov  edi, [esp+1692]
        mov  [esp+1656],edx
        mov  edx, [esp+1828]
        mov  eax, [esp+1704]
        mov  [esp+1752],ebp
        xor  [esp+1656],ebx
        xor  ecx, [esp+1824]
        mov  esi, [esi+8]
        xor  [esp+1656],esi
        xor  edi, [esp+1656]
        mov  [esp+1748],edi
        xor  edx, [eax+4]
        mov  [esp+1680],edx
        mov  esi,edx
        xor  esi, [esp+1696]
        mov  [esp+1744],esi
        mov  ebx, [eax]
        xor  ecx,ebx
        mov  ebx, [esp+1700]
        mov  [esp+1652],ecx
        xor  ebx,ecx
        mov  ecx,ebp
        not  ecx
        mov  eax,ecx
        or  eax,ebx
        xor  [esp+1028],eax
        mov  edx,edi
        or  edx,ebp
        xor  edx,ecx
        mov  eax,esi
        xor  eax,edx
        or  eax,edi
        xor  [esp+1020],eax
        xor  eax,ebx
        or  eax,esi
        xor  [esp+1016],eax
        mov  eax, [esp+1028]
        mov  [esp+1740],ebx
        mov  ebx, [esp+1028]
        mov  esi, [esp+152]
        xor  [esp+1024],edx
        or  eax, [esp+1024]
        and  ebx, [esp+1016]
        xor  ebx,eax
        xor  [esi],ebx
        mov  esi, [esp+1028]
        mov  edi, [esp+160]
        or  esi, [esp+1020]
        xor  esi, [esp+1024]
        mov  ecx,esi
        or  ecx,ebx
        xor  ecx, [esp+1028]
        xor  [edi],ecx
        mov  eax, [esp+1028]
        mov  edx, [esp+1028]
        mov  ebp, [esp+148]
        or  eax, [esp+1016]
        and  edx, [esp+1020]
        xor  edx,eax
        mov  eax,edx
        not  eax
        or  edx,ebx
        xor  ecx,esi
        xor  [ebp],eax
        mov  eax, [esp+156]
        xor  ecx,edx
        not  ecx
        xor  [eax],ecx
        mov  esi, [esp+1752]
        mov  edi, [esp+1748]
        mov  ebp, [esp+1744]
        mov  eax,esi
        not  eax
        mov  ebx,eax
        or  ebx, [esp+1740]
        mov  ecx,edi
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebp
        or  edx,edi
        mov  eax,edx
        xor  eax, [esp+1740]
        or  eax,ebp
        xor  ecx, [esp+1024]
        mov  [esp+1980],ecx
        xor  edx, [esp+1020]
        mov  [esp+1976],edx
        xor  eax, [esp+1016]
        mov  [esp+1972],eax
        xor  ebx, [esp+1028]
        mov  [esp+1968],ebx
        mov  [esp+1964],esi
        mov  [esp+1960],edi
        mov  [esp+1956],ebp
        mov  edx, [esp+1740]
        mov  ecx, [esp+1684]
        mov  ebx, [esp+1688]
        mov  esi, [esp+1692]
        mov  edi, [esp+1700]
        mov  [esp+1952],edx
        mov  edx, [esp+1656]
        xor  ecx, [esp+1668]
        xor  esi, [esp+1676]
        xor  edi, [esp+1680]
        mov  eax,esi
        not  eax
        mov  ebp,eax
        or  edx,esi
        xor  edx,eax
        xor  ecx,edx
        mov  eax,edi
        xor  eax,edx
        mov  edx, [esp+1664]
        xor  ebx, [esp+1672]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+1004],ecx
        or  eax, [esp+1656]
        mov  [esp+1000],eax
        xor  eax, [esp+1652]
        or  eax,edi
        xor  ebx,eax
        mov  [esp+996],ebx
        mov  eax,ebp
        or  eax,ecx
        mov  ecx,ebp
        and  ecx,ebx
        mov  ebx, [esp+1656]
        mov  [esp+1012],esi
        mov  esi, [esp+1652]
        mov  [esp+1008],edi
        xor  [esp+1000],edx
        xor  ecx,eax
        mov  [esp+992],ecx
        xor  [esp+992],ebx
        mov  edi,ebp
        or  edi, [esp+1000]
        xor  edi, [esp+1004]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+988],ebx
        xor  [esp+988],esi
        mov  edx,ebp
        or  edx, [esp+996]
        mov  eax,ebp
        and  eax, [esp+1000]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+992]
        not  esi
        xor  esi, [esp+1012]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+1008]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+988]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+992]
        mov  eax,edx
        xor  eax, [esp+988]
        xor  ecx, [esp+1004]
        mov  [esp+1852],ecx
        xor  ebp,edi
        mov  edi, [esp+992]
        mov  [esp+1840],ebp
        mov  ebp, [esp+988]
        xor  edx, [esp+1000]
        mov  [esp+1848],edx
        mov  edx, [esp+1984]
        or  eax,ebx
        xor  eax, [esp+996]
        mov  [esp+1844],eax
        add  dword [esp+1704],BYTE 64
        mov  eax, [esp+1704]
        mov  ecx, [esp+1988]
        mov  [esp+1828],ebx
        mov  ebx, [esp+1992]
        mov  [esp+1836],esi
        mov  esi, [esp+1996]
        mov  [esp+1832],edi
        mov  edi, [esp+2000]
        mov  [esp+1824],ebp
        mov  ebp, [esp+2008]
        xor  edx, [eax+32]
        mov  [esp+1700],edx
        mov  edx, [esp+2004]
        xor  ecx, [eax+36]
        mov  [esp+1696],ecx
        mov  ecx, [esp+2012]
        xor  ebx, [eax+40]
        mov  [esp+1692],ebx
        mov  ebx, [esp+1884]
        xor  esi, [eax+44]
        xor  edi, [eax+48]
        mov  [esp+1688],edi
        xor  edx, [eax+52]
        xor  ebp, [eax+56]
        mov  [esp+1684],ebp
        xor  ecx, [eax+60]
        xor  ebx, [eax+28]
        mov  [esp+1668],ebx
        mov  edi, [esp+1880]
        mov  [esp+968],ecx
        xor  [esp+968],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [eax+24]
        mov  eax, [esp+1684]
        xor  [esp+1664],ebp
        mov  ebx, [esp+1664]
        mov  ebp, [esp+1876]
        mov  edi, [esp+1704]
        mov  [esp+964],eax
        mov  eax, [esp+1872]
        xor  [esp+964],ebx
        mov  ebx, [esp+1868]
        xor  ebp, [edi+20]
        mov  [esp+1672],ebp
        mov  [esp+960],ebp
        xor  [esp+960],edx
        mov  [esp+1660],esi
        xor  [esp+1660],eax
        mov  edx, [edi+16]
        mov  ebp, [esp+1688]
        xor  [esp+1660],edx
        xor  ebp, [esp+1660]
        xor  ebx, [edi+12]
        mov  [esp+1676],ebx
        mov  [esp+984],ebx
        xor  [esp+984],esi
        mov  esi, [esp+1696]
        mov  edi, [esp+1864]
        mov  eax, [esp+1704]
        mov  ebx, [esp+1692]
        mov  [esp+1656],esi
        mov  esi, [esp+1860]
        mov  edx, [esp+1704]
        xor  [esp+1656],edi
        mov  edi, [esp+1696]
        xor  ecx, [esp+1856]
        mov  eax, [eax+8]
        xor  [esp+1656],eax
        xor  esi, [edx+4]
        mov  [esp+1680],esi
        mov  [esp+976],esi
        xor  [esp+976],edi
        mov  eax, [edx]
        mov  esi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  esi,ecx
        mov  ecx, [esp+984]
        not  ecx
        mov  eax,ecx
        or  eax,esi
        xor  ebp,eax
        mov  eax, [esp+976]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+984]
        xor  edx,ecx
        xor  [esp+968],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+964],eax
        xor  eax,esi
        or  eax, [esp+976]
        xor  [esp+960],eax
        mov  eax,ebp
        or  eax, [esp+968]
        mov  ecx,ebp
        and  ecx, [esp+960]
        xor  ecx,eax
        mov  [esp+980],ecx
        xor  [esp+980],ebx
        mov  edi,ebp
        or  edi, [esp+964]
        xor  edi, [esp+968]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+972],ebx
        xor  [esp+972],esi
        mov  edx,ebp
        or  edx, [esp+960]
        mov  eax,ebp
        and  eax, [esp+964]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+980]
        not  esi
        xor  esi, [esp+984]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+976]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+980]
        mov  eax,edx
        xor  edx, [esp+964]
        mov  [esp+2008],edx
        mov  edx, [esp+980]
        xor  ecx, [esp+968]
        mov  [esp+2012],ecx
        mov  ecx, [esp+972]
        xor  eax, [esp+972]
        or  eax,ebx
        mov  [esp+1988],ebx
        mov  ebx, [esp+1684]
        mov  [esp+1996],esi
        mov  esi, [esp+1688]
        or  edi, [esp+972]
        xor  ebp,edi
        mov  edi, [esp+1692]
        xor  eax, [esp+960]
        mov  [esp+2004],eax
        mov  [esp+2000],ebp
        mov  [esp+1992],edx
        mov  [esp+1984],ecx
        xor  ebx, [esp+1668]
        xor  esi, [esp+1672]
        xor  edi, [esp+1676]
        mov  [esp+956],edi
        mov  ebp, [esp+1700]
        mov  edx, [esp+1656]
        xor  ebp, [esp+1680]
        mov  [esp+952],ebp
        mov  eax,edi
        not  eax
        mov  ebp,eax
        or  edx,edi
        xor  edx,eax
        mov  eax, [esp+952]
        xor  ebx,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+948],ebx
        or  eax, [esp+1656]
        mov  [esp+944],eax
        xor  eax, [esp+1652]
        or  eax, [esp+952]
        xor  esi,eax
        mov  eax,ebp
        or  eax,ebx
        mov  ebx, [esp+1656]
        mov  [esp+940],esi
        mov  ecx,ebp
        and  ecx,esi
        mov  esi, [esp+1652]
        xor  [esp+944],edx
        xor  ecx,eax
        mov  [esp+936],ecx
        xor  [esp+936],ebx
        mov  edi,ebp
        or  edi, [esp+944]
        xor  edi, [esp+948]
        mov  ebx,edi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+932],ebx
        xor  [esp+932],esi
        mov  edx,ebp
        or  edx, [esp+940]
        mov  eax,ebp
        and  eax, [esp+944]
        xor  eax,edx
        mov  esi,eax
        or  eax,ecx
        mov  ecx, [esp+936]
        not  esi
        xor  esi, [esp+956]
        xor  ebx,edi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+952]
        mov  eax,esi
        not  eax
        mov  edi,eax
        or  edi, [esp+932]
        or  ecx,esi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+936]
        mov  eax,edx
        xor  eax, [esp+932]
        xor  ecx, [esp+948]
        mov  [esp+1884],ecx
        xor  ebp,edi
        mov  edi, [esp+936]
        mov  [esp+1872],ebp
        mov  ebp, [esp+932]
        or  eax,ebx
        xor  eax, [esp+940]
        mov  [esp+1876],eax
        mov  eax, [esp+116]
        xor  edx, [esp+944]
        mov  [esp+1880],edx
        mov  edx, [esp+120]
        mov  [esp+1868],esi
        mov  [esp+1864],edi
        mov  [esp+1860],ebx
        mov  [esp+1856],ebp
        mov  [esp+1716],eax
        mov  [esp+1712],edx
        mov  dword [esp+1708],csc_tabc+256
        mov  eax,7
        ALIGN 4
L247:
        mov  dword [esp+928],8
        dec  eax
        mov  [esp+136],eax
        ;;ALIGN 1<<4
	db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00  ; leal   0x0(%esi),%esi
	db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00  ; leal   0x0(%edi,1),%edi
L251:
        mov  ecx, [esp+1712]
        mov  ebx, [ecx+28]
        mov  ecx, [esp+1708]
        mov  esi, [esp+1712]
        xor  ebx, [ecx+28]
        mov  [esp+916],ebx
        mov  edi, [esi+24]
        xor  edi, [ecx+24]
        mov  [esp+912],edi
        mov  ebp, [esi+20]
        xor  ebp, [ecx+20]
        mov  [esp+908],ebp
        mov  eax, [esi+16]
        mov  edx, [esi+12]
        xor  eax, [ecx+16]
        xor  edx, [ecx+12]
        mov  [esp+924],edx
        mov  ebx, [esi+8]
        mov  edi, [esi+4]
        xor  ebx, [ecx+8]
        xor  edi, [ecx+4]
        mov  [esp+920],edi
        mov  edi, [esi]
        mov  ebp,eax
        xor  edi, [ecx]
        mov  ecx,edx
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+920]
        mov  edx,ebx
        or  edx, [esp+924]
        xor  edx,ecx
        xor  [esp+916],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+912],eax
        xor  eax,edi
        or  eax, [esp+920]
        xor  [esp+908],eax
        mov  eax,ebp
        or  eax, [esp+916]
        mov  ecx,ebp
        and  ecx, [esp+908]
        xor  ecx,eax
        mov  [esp+904],ecx
        xor  [esp+904],ebx
        mov  esi,ebp
        or  esi, [esp+912]
        xor  esi, [esp+916]
        mov  ebx,esi
        or  ebx,ecx
        xor  ebx,ebp
        mov  [esp+900],ebx
        xor  [esp+900],edi
        mov  edx,ebp
        or  edx, [esp+908]
        mov  eax,ebp
        and  eax, [esp+912]
        xor  eax,edx
        mov  edx, [esp+900]
        mov  edi,eax
        or  eax,ecx
        mov  ecx, [esp+904]
        xor  ebx,esi
        mov  esi, [esp+1716]
        not  edi
        xor  edi, [esp+924]
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+920]
        mov  eax,edi
        not  eax
        mov  [esp+24],eax
        or  [esp+24],edx
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+904]
        mov  eax,edx
        xor  eax, [esp+900]
        or  eax,ebx
        xor  ecx, [esp+916]
        mov  [esi+224],ecx
        xor  edx, [esp+912]
        mov  [esi+192],edx
        xor  eax, [esp+908]
        mov  [esi+160],eax
        mov  eax, [esp+24]
        xor  eax,ebp
        mov  [esi+128],eax
        mov  [esi+96],edi
        mov  edx, [esp+904]
        mov  [esi+64],edx
        mov  [esi+32],ebx
        mov  ecx, [esp+900]
        mov  [esi],ecx
        add  dword [esp+1708],BYTE 32
        add  dword [esp+1712],BYTE 32
        add  esi,BYTE 4
        mov  [esp+1716],esi
        dec  dword [esp+928]
        jnz near L251
        add  esi,BYTE -32
        mov  [esp+1716],esi
        mov  eax, [esi+32]
        xor  eax, [esi+-480]
        mov  [esi+32],eax
        xor  eax, [esp+1792]
        mov  [esp+1700],eax
        mov  eax, [esi+36]
        xor  eax, [esi+-476]
        mov  [esi+36],eax
        xor  eax, [esp+1796]
        mov  [esp+1696],eax
        mov  eax, [esi+40]
        xor  eax, [esi+-472]
        mov  [esi+40],eax
        xor  eax, [esp+1800]
        mov  [esp+1692],eax
        mov  eax, [esi+44]
        xor  eax, [esi+-468]
        mov  [esi+44],eax
        mov  ebx, [esp+1716]
        mov  esi, [esp+1804]
        xor  esi,eax
        mov  eax, [ebx+48]
        xor  eax, [ebx+-464]
        mov  [ebx+48],eax
        xor  eax, [esp+1808]
        mov  [esp+1688],eax
        mov  eax, [ebx+52]
        xor  eax, [ebx+-460]
        mov  [ebx+52],eax
        mov  edx, [esp+1812]
        xor  edx,eax
        mov  eax, [ebx+56]
        xor  eax, [ebx+-456]
        mov  [ebx+56],eax
        xor  eax, [esp+1816]
        mov  [esp+1684],eax
        mov  eax, [ebx+60]
        xor  eax, [ebx+-452]
        mov  [ebx+60],eax
        mov  ecx, [esp+1820]
        xor  ecx,eax
        mov  eax, [ebx+28]
        xor  eax, [ebx+-484]
        mov  [ebx+28],eax
        mov  edi, [esp+1784]
        xor  eax, [esp+1788]
        mov  [esp+1668],eax
        mov  [esp+888],ecx
        xor  [esp+888],eax
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  eax, [ebx+24]
        xor  eax, [ebx+-488]
        mov  [ebx+24],eax
        mov  ebp, [esp+1684]
        xor  [esp+1664],eax
        mov  eax, [esp+1664]
        mov  [esp+884],ebp
        xor  [esp+884],eax
        mov  eax, [ebx+20]
        xor  eax, [ebx+-492]
        mov  [ebx+20],eax
        xor  eax, [esp+1780]
        mov  [esp+1672],eax
        mov  [esp+880],eax
        xor  [esp+880],edx
        mov  [esp+1660],esi
        mov  edx, [esp+1776]
        xor  [esp+1660],edx
        mov  eax, [ebx+16]
        xor  eax, [ebx+-496]
        mov  [ebx+16],eax
        mov  ebp, [esp+1688]
        xor  [esp+1660],eax
        mov  eax, [ebx+12]
        xor  ebp, [esp+1660]
        xor  eax, [ebx+-500]
        mov  [ebx+12],eax
        mov  ebx, [esp+1696]
        xor  eax, [esp+1772]
        mov  [esp+896],eax
        xor  [esp+896],esi
        mov  esi, [esp+1768]
        mov  edi, [esp+1716]
        mov  [esp+1676],eax
        mov  [esp+1656],ebx
        xor  [esp+1656],esi
        mov  eax, [edi+8]
        xor  eax, [edi+-504]
        mov  [edi+8],eax
        mov  ebx, [esp+1692]
        xor  [esp+1656],eax
        mov  eax, [edi+4]
        xor  ebx, [esp+1656]
        xor  eax, [edi+-508]
        mov  [edi+4],eax
        xor  eax, [esp+1764]
        mov  [esp+1680],eax
        mov  [esp+892],eax
        mov  eax, [esp+1696]
        xor  [esp+892],eax
        mov  eax, [edi]
        xor  ecx, [esp+1760]
        xor  eax, [edi+-512]
        mov  [edi],eax
        mov  edi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+896]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+892]
        mov  edx,ebx
        or  edx, [esp+896]
        xor  edx,ecx
        xor  [esp+888],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+884],eax
        xor  eax,edi
        or  eax, [esp+892]
        xor  [esp+880],eax
        mov  eax,ebp
        or  eax, [esp+888]
        mov  esi,ebp
        and  esi, [esp+880]
        xor  esi,eax
        mov  [esp+876],esi
        xor  [esp+876],ebx
        mov  ecx,ebp
        or  ecx, [esp+884]
        xor  ecx, [esp+888]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+872],ebx
        xor  ebx,ecx
        mov  ecx, [esp+876]
        xor  [esp+872],edi
        mov  edx,ebp
        or  edx, [esp+880]
        mov  eax,ebp
        and  eax, [esp+884]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+896]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+892]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+872]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+876]
        mov  eax,edx
        xor  eax, [esp+872]
        xor  ecx, [esp+888]
        mov  [esp+1820],ecx
        xor  edx, [esp+884]
        mov  [esp+1816],edx
        mov  edx, [esp+876]
        mov  ecx, [esp+872]
        or  eax,ebx
        mov  [esp+1796],ebx
        mov  ebx, [esp+1684]
        xor  ebp,esi
        mov  esi, [esp+1688]
        mov  [esp+1804],edi
        mov  edi, [esp+1692]
        mov  [esp+1808],ebp
        mov  ebp, [esp+1700]
        mov  [esp+1800],edx
        mov  edx, [esp+1656]
        xor  eax, [esp+880]
        mov  [esp+1812],eax
        xor  edi, [esp+1676]
        xor  ebp, [esp+1680]
        mov  [esp+864],ebp
        mov  eax,edi
        not  eax
        mov  ebp,eax
        or  edx,edi
        xor  edx,eax
        mov  eax, [esp+864]
        xor  ebx, [esp+1668]
        xor  ebx,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  [esp+1792],ecx
        xor  esi, [esp+1672]
        mov  [esp+868],edi
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+860],ebx
        or  eax, [esp+1656]
        mov  [esp+856],eax
        xor  [esp+856],edx
        xor  eax, [esp+1652]
        or  eax, [esp+864]
        xor  esi,eax
        mov  [esp+852],esi
        mov  ecx, [esp+1656]
        mov  edi, [esp+1652]
        mov  eax,ebp
        or  eax,ebx
        mov  esi,ebp
        and  esi, [esp+852]
        xor  esi,eax
        mov  [esp+848],esi
        xor  [esp+848],ecx
        mov  ecx,ebp
        or  ecx, [esp+856]
        xor  ecx,ebx
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+844],ebx
        xor  ebx,ecx
        mov  ecx, [esp+848]
        xor  [esp+844],edi
        mov  edx,ebp
        or  edx, [esp+852]
        mov  eax,ebp
        and  eax, [esp+856]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+868]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+844]
        xor  ebp,esi
        mov  [esp+1776],ebp
        mov  ebp, [esp+848]
        not  ebx
        xor  ebx, [esp+864]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+848]
        mov  eax,edx
        xor  eax, [esp+844]
        or  eax,ebx
        xor  eax, [esp+852]
        mov  [esp+1780],eax
        mov  eax, [esp+844]
        xor  edx, [esp+856]
        mov  [esp+1784],edx
        add  dword [esp+1716],BYTE 64
        mov  edx, [esp+1716]
        xor  ecx, [esp+860]
        mov  [esp+1788],ecx
        mov  [esp+1772],edi
        mov  [esp+1768],ebp
        mov  [esp+1764],ebx
        mov  [esp+1760],eax
        mov  eax, [edx+32]
        xor  eax, [edx+-480]
        mov  [edx+32],eax
        xor  eax, [esp+1856]
        mov  [esp+1700],eax
        mov  eax, [edx+36]
        xor  eax, [edx+-476]
        mov  [edx+36],eax
        xor  eax, [esp+1860]
        mov  [esp+1696],eax
        mov  eax, [edx+40]
        xor  eax, [edx+-472]
        mov  [edx+40],eax
        xor  eax, [esp+1864]
        mov  [esp+1692],eax
        mov  eax, [edx+44]
        xor  eax, [edx+-468]
        mov  [edx+44],eax
        mov  esi, [esp+1868]
        xor  esi,eax
        mov  eax, [edx+48]
        xor  eax, [edx+-464]
        mov  [edx+48],eax
        xor  eax, [esp+1872]
        mov  [esp+1688],eax
        mov  eax, [edx+52]
        xor  eax, [edx+-460]
        mov  [edx+52],eax
        mov  ecx, [esp+1716]
        mov  edx, [esp+1876]
        xor  edx,eax
        mov  eax, [ecx+56]
        xor  eax, [ecx+-456]
        mov  [ecx+56],eax
        xor  eax, [esp+1880]
        mov  [esp+1684],eax
        mov  eax, [ecx+60]
        xor  eax, [ecx+-452]
        mov  [ecx+60],eax
        mov  ebx, [esp+1716]
        mov  ecx, [esp+1884]
        xor  ecx,eax
        mov  eax, [ebx+28]
        xor  eax, [ebx+-484]
        mov  [ebx+28],eax
        mov  edi, [esp+1848]
        xor  eax, [esp+1852]
        mov  [esp+1668],eax
        mov  [esp+832],ecx
        xor  [esp+832],eax
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  eax, [ebx+24]
        xor  eax, [ebx+-488]
        mov  [ebx+24],eax
        mov  ebp, [esp+1684]
        xor  [esp+1664],eax
        mov  eax, [esp+1664]
        mov  [esp+828],ebp
        xor  [esp+828],eax
        mov  eax, [ebx+20]
        xor  eax, [ebx+-492]
        mov  [ebx+20],eax
        xor  eax, [esp+1844]
        mov  [esp+824],eax
        xor  [esp+824],edx
        mov  edx, [esp+1840]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  eax, [ebx+16]
        xor  eax, [ebx+-496]
        mov  [ebx+16],eax
        mov  ebp, [esp+1688]
        xor  [esp+1660],eax
        mov  eax, [ebx+12]
        xor  ebp, [esp+1660]
        xor  eax, [ebx+-500]
        mov  [ebx+12],eax
        mov  ebx, [esp+1696]
        xor  eax, [esp+1836]
        mov  [esp+840],eax
        xor  [esp+840],esi
        mov  esi, [esp+1832]
        mov  edi, [esp+1716]
        mov  [esp+1676],eax
        mov  [esp+1656],ebx
        xor  [esp+1656],esi
        mov  eax, [edi+8]
        xor  eax, [edi+-504]
        mov  [edi+8],eax
        mov  ebx, [esp+1692]
        xor  [esp+1656],eax
        mov  eax, [edi+4]
        xor  ebx, [esp+1656]
        xor  eax, [edi+-508]
        mov  [edi+4],eax
        xor  eax, [esp+1828]
        mov  [esp+1680],eax
        mov  [esp+836],eax
        mov  eax, [esp+1696]
        xor  [esp+836],eax
        mov  eax, [edi]
        xor  ecx, [esp+1824]
        xor  eax, [edi+-512]
        mov  [edi],eax
        mov  edi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+840]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+836]
        mov  edx,ebx
        or  edx, [esp+840]
        xor  edx,ecx
        xor  [esp+832],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+828],eax
        xor  eax,edi
        or  eax, [esp+836]
        xor  [esp+824],eax
        mov  eax,ebp
        or  eax, [esp+832]
        mov  esi,ebp
        and  esi, [esp+824]
        xor  esi,eax
        mov  [esp+820],esi
        xor  [esp+820],ebx
        mov  ecx,ebp
        or  ecx, [esp+828]
        xor  ecx, [esp+832]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+816],ebx
        xor  ebx,ecx
        mov  ecx, [esp+820]
        xor  [esp+816],edi
        mov  edx,ebp
        or  edx, [esp+824]
        mov  eax,ebp
        and  eax, [esp+828]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+840]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+836]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+816]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+820]
        mov  eax,edx
        xor  eax, [esp+816]
        xor  ecx, [esp+832]
        mov  [esp+1884],ecx
        xor  edx, [esp+828]
        mov  [esp+1880],edx
        mov  edx, [esp+820]
        mov  ecx, [esp+816]
        or  eax,ebx
        mov  [esp+1860],ebx
        mov  ebx, [esp+1684]
        xor  ebp,esi
        mov  esi, [esp+1688]
        mov  [esp+1868],edi
        mov  edi, [esp+1692]
        mov  [esp+1872],ebp
        mov  ebp, [esp+1700]
        mov  [esp+1864],edx
        mov  edx, [esp+1656]
        xor  eax, [esp+824]
        mov  [esp+1876],eax
        xor  edi, [esp+1676]
        xor  ebp, [esp+1680]
        mov  [esp+808],ebp
        mov  eax,edi
        not  eax
        mov  ebp,eax
        or  edx,edi
        xor  edx,eax
        mov  eax, [esp+808]
        xor  ebx, [esp+1668]
        xor  ebx,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  [esp+1856],ecx
        xor  esi, [esp+1672]
        mov  [esp+812],edi
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+804],ebx
        or  eax, [esp+1656]
        mov  [esp+800],eax
        xor  [esp+800],edx
        xor  eax, [esp+1652]
        or  eax, [esp+808]
        xor  esi,eax
        mov  [esp+796],esi
        mov  ecx, [esp+1656]
        mov  edi, [esp+1652]
        mov  eax,ebp
        or  eax,ebx
        mov  esi,ebp
        and  esi, [esp+796]
        xor  esi,eax
        mov  [esp+792],esi
        xor  [esp+792],ecx
        mov  ecx,ebp
        or  ecx, [esp+800]
        xor  ecx,ebx
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+788],ebx
        xor  ebx,ecx
        mov  ecx, [esp+792]
        xor  [esp+788],edi
        mov  edx,ebp
        or  edx, [esp+796]
        mov  eax,ebp
        and  eax, [esp+800]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+812]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+788]
        xor  ebp,esi
        mov  [esp+1840],ebp
        mov  ebp, [esp+792]
        not  ebx
        xor  ebx, [esp+808]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+792]
        mov  eax,edx
        xor  eax, [esp+788]
        or  eax,ebx
        xor  eax, [esp+796]
        mov  [esp+1844],eax
        mov  eax, [esp+788]
        xor  edx, [esp+800]
        mov  [esp+1848],edx
        add  dword [esp+1716],BYTE 64
        mov  edx, [esp+1716]
        xor  ecx, [esp+804]
        mov  [esp+1852],ecx
        mov  [esp+1836],edi
        mov  [esp+1832],ebp
        mov  [esp+1828],ebx
        mov  [esp+1824],eax
        mov  eax, [edx+32]
        xor  eax, [edx+-480]
        mov  [edx+32],eax
        xor  eax, [esp+1920]
        mov  [esp+1700],eax
        mov  eax, [edx+36]
        xor  eax, [edx+-476]
        mov  [edx+36],eax
        xor  eax, [esp+1924]
        mov  [esp+1696],eax
        mov  eax, [edx+40]
        xor  eax, [edx+-472]
        mov  [edx+40],eax
        xor  eax, [esp+1928]
        mov  [esp+1692],eax
        mov  eax, [edx+44]
        xor  eax, [edx+-468]
        mov  [edx+44],eax
        mov  esi, [esp+1932]
        xor  esi,eax
        mov  eax, [edx+48]
        xor  eax, [edx+-464]
        mov  [edx+48],eax
        xor  eax, [esp+1936]
        mov  [esp+1688],eax
        mov  eax, [edx+52]
        xor  eax, [edx+-460]
        mov  [edx+52],eax
        mov  ecx, [esp+1716]
        mov  edx, [esp+1940]
        xor  edx,eax
        mov  eax, [ecx+56]
        xor  eax, [ecx+-456]
        mov  [ecx+56],eax
        xor  eax, [esp+1944]
        mov  [esp+1684],eax
        mov  eax, [ecx+60]
        xor  eax, [ecx+-452]
        mov  [ecx+60],eax
        mov  ebx, [esp+1716]
        mov  ecx, [esp+1948]
        xor  ecx,eax
        mov  eax, [ebx+28]
        xor  eax, [ebx+-484]
        mov  [ebx+28],eax
        mov  edi, [esp+1912]
        xor  eax, [esp+1916]
        mov  [esp+1668],eax
        mov  [esp+772],ecx
        xor  [esp+772],eax
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  eax, [ebx+24]
        xor  eax, [ebx+-488]
        mov  [ebx+24],eax
        mov  ebp, [esp+1684]
        xor  [esp+1664],eax
        mov  eax, [esp+1664]
        mov  [esp+768],ebp
        xor  [esp+768],eax
        mov  eax, [ebx+20]
        xor  eax, [ebx+-492]
        mov  [ebx+20],eax
        xor  eax, [esp+1908]
        mov  [esp+764],eax
        xor  [esp+764],edx
        mov  edx, [esp+1904]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  eax, [ebx+16]
        xor  eax, [ebx+-496]
        mov  [ebx+16],eax
        mov  ebx, [esp+1688]
        xor  [esp+1660],eax
        mov  edi, [esp+1660]
        mov  ebp, [esp+1716]
        mov  [esp+776],ebx
        xor  [esp+776],edi
        mov  eax, [ebp+12]
        xor  eax, [ebp+-500]
        mov  [ebp+12],eax
        xor  eax, [esp+1900]
        mov  [esp+1676],eax
        mov  [esp+784],eax
        mov  eax, [esp+1696]
        mov  edx, [esp+1896]
        xor  [esp+784],esi
        mov  [esp+1656],eax
        xor  [esp+1656],edx
        mov  eax, [ebp+8]
        xor  eax, [ebp+-504]
        mov  [ebp+8],eax
        mov  ebx, [esp+1692]
        xor  [esp+1656],eax
        mov  eax, [ebp+4]
        xor  ebx, [esp+1656]
        xor  eax, [ebp+-508]
        mov  [ebp+4],eax
        mov  esi, [esp+1696]
        xor  eax, [esp+1892]
        mov  [esp+1680],eax
        mov  [esp+780],eax
        xor  [esp+780],esi
        mov  eax, [ebp]
        xor  ecx, [esp+1888]
        xor  eax, [ebp+-512]
        mov  [ebp],eax
        mov  edi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+784]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  [esp+776],eax
        mov  eax, [esp+780]
        mov  edx,ebx
        or  edx, [esp+784]
        xor  edx,ecx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+768],eax
        xor  eax,edi
        or  eax, [esp+780]
        xor  [esp+764],eax
        mov  eax, [esp+776]
        mov  esi, [esp+776]
        mov  ecx, [esp+776]
        xor  [esp+772],edx
        mov  edx, [esp+776]
        or  eax, [esp+772]
        and  esi, [esp+764]
        xor  esi,eax
        mov  eax, [esp+776]
        mov  [esp+760],esi
        xor  [esp+760],ebx
        or  ecx, [esp+768]
        xor  ecx, [esp+772]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx, [esp+776]
        mov  ebp,ebx
        xor  ebx,ecx
        mov  ecx, [esp+760]
        xor  ebp,edi
        or  edx, [esp+764]
        and  eax, [esp+768]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+784]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+780]
        mov  eax,edi
        not  eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+760]
        xor  ecx, [esp+772]
        mov  [esp+1948],ecx
        mov  [esp+1932],edi
        mov  edi, [esp+760]
        mov  esi,eax
        or  esi,ebp
        mov  eax,edx
        xor  eax,ebp
        mov  [esp+1920],ebp
        mov  ebp, [esp+1684]
        or  eax,ebx
        xor  eax, [esp+764]
        mov  [esp+1940],eax
        mov  eax, [esp+1688]
        xor  edx, [esp+768]
        mov  [esp+1944],edx
        mov  edx, [esp+1692]
        mov  ecx, [esp+1700]
        xor  eax, [esp+1672]
        mov  [esp+740],eax
        xor  edx, [esp+1676]
        mov  [esp+756],edx
        mov  eax,edx
        mov  edx, [esp+1656]
        mov  [esp+1924],ebx
        mov  ebx, [esp+1664]
        xor  esi, [esp+776]
        mov  [esp+1936],esi
        mov  [esp+1928],edi
        xor  ebp, [esp+1668]
        mov  [esp+748],ebp
        xor  ecx, [esp+1680]
        mov  [esp+752],ecx
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx, [esp+756]
        xor  edx,eax
        xor  [esp+748],edx
        mov  eax,ecx
        xor  eax,edx
        or  eax, [esp+1656]
        mov  [esp+744],eax
        xor  [esp+744],ebx
        xor  eax, [esp+1652]
        or  eax,ecx
        xor  [esp+740],eax
        mov  edi, [esp+1656]
        mov  eax,ebp
        or  eax, [esp+748]
        mov  esi,ebp
        and  esi, [esp+740]
        xor  esi,eax
        mov  eax, [esp+1652]
        mov  [esp+736],esi
        xor  [esp+736],edi
        mov  ecx,ebp
        or  ecx, [esp+744]
        xor  ecx, [esp+748]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+732],ebx
        xor  ebx,ecx
        mov  ecx, [esp+736]
        xor  [esp+732],eax
        mov  edx,ebp
        or  edx, [esp+740]
        mov  eax,ebp
        and  eax, [esp+744]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+756]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+752]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+736]
        mov  eax,edx
        xor  edx, [esp+744]
        mov  [esp+1912],edx
        mov  edx, [esp+736]
        xor  ecx, [esp+748]
        mov  [esp+1916],ecx
        mov  ecx, [esp+732]
        or  esi, [esp+732]
        xor  eax, [esp+732]
        or  eax,ebx
        xor  eax, [esp+740]
        mov  [esp+1908],eax
        xor  ebp,esi
        mov  [esp+1904],ebp
        mov  [esp+1900],edi
        mov  [esp+1896],edx
        mov  [esp+1892],ebx
        mov  [esp+1888],ecx
        add  dword [esp+1716],BYTE 64
        mov  ebx, [esp+1716]
        mov  eax, [ebx+32]
        xor  eax, [ebx+-480]
        mov  [ebx+32],eax
        xor  eax, [esp+1984]
        mov  [esp+1700],eax
        mov  eax, [ebx+36]
        xor  eax, [ebx+-476]
        mov  [ebx+36],eax
        xor  eax, [esp+1988]
        mov  [esp+1696],eax
        mov  eax, [ebx+40]
        xor  eax, [ebx+-472]
        mov  [ebx+40],eax
        xor  eax, [esp+1992]
        mov  [esp+1692],eax
        mov  eax, [ebx+44]
        xor  eax, [ebx+-468]
        mov  [ebx+44],eax
        mov  esi, [esp+1996]
        xor  esi,eax
        mov  eax, [ebx+48]
        xor  eax, [ebx+-464]
        mov  [ebx+48],eax
        xor  eax, [esp+2000]
        mov  [esp+1688],eax
        mov  eax, [ebx+52]
        xor  eax, [ebx+-460]
        mov  [ebx+52],eax
        mov  edx, [esp+2004]
        xor  edx,eax
        mov  eax, [ebx+56]
        xor  eax, [ebx+-456]
        mov  [ebx+56],eax
        xor  eax, [esp+2008]
        mov  [esp+1684],eax
        mov  eax, [ebx+60]
        xor  eax, [ebx+-452]
        mov  [ebx+60],eax
        mov  ecx, [esp+2012]
        xor  ecx,eax
        mov  eax, [ebx+28]
        xor  eax, [ebx+-484]
        mov  [ebx+28],eax
        mov  edi, [esp+1976]
        xor  eax, [esp+1980]
        mov  [esp+1668],eax
        mov  [esp+720],ecx
        xor  [esp+720],eax
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  eax, [ebx+24]
        xor  eax, [ebx+-488]
        mov  [ebx+24],eax
        mov  ebp, [esp+1684]
        xor  [esp+1664],eax
        mov  eax, [esp+1664]
        mov  [esp+716],ebp
        xor  [esp+716],eax
        mov  eax, [ebx+20]
        xor  eax, [ebx+-492]
        mov  [ebx+20],eax
        xor  eax, [esp+1972]
        mov  [esp+1672],eax
        mov  [esp+712],eax
        xor  [esp+712],edx
        mov  edx, [esp+1968]
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  eax, [ebx+16]
        xor  eax, [ebx+-496]
        mov  [ebx+16],eax
        mov  ebp, [esp+1688]
        xor  [esp+1660],eax
        mov  eax, [ebx+12]
        xor  ebp, [esp+1660]
        xor  eax, [ebx+-500]
        mov  [ebx+12],eax
        mov  ebx, [esp+1696]
        xor  eax, [esp+1964]
        mov  [esp+728],eax
        xor  [esp+728],esi
        mov  esi, [esp+1960]
        mov  edi, [esp+1716]
        mov  [esp+1676],eax
        mov  [esp+1656],ebx
        xor  [esp+1656],esi
        mov  eax, [edi+8]
        xor  eax, [edi+-504]
        mov  [edi+8],eax
        mov  ebx, [esp+1692]
        xor  [esp+1656],eax
        mov  eax, [edi+4]
        xor  ebx, [esp+1656]
        xor  eax, [edi+-508]
        mov  [edi+4],eax
        xor  eax, [esp+1956]
        mov  [esp+1680],eax
        mov  [esp+724],eax
        mov  eax, [esp+1696]
        xor  [esp+724],eax
        mov  eax, [edi]
        xor  ecx, [esp+1952]
        xor  eax, [edi+-512]
        mov  [edi],eax
        mov  edi, [esp+1700]
        xor  ecx,eax
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+728]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+724]
        mov  edx,ebx
        or  edx, [esp+728]
        xor  edx,ecx
        xor  [esp+720],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+716],eax
        xor  eax,edi
        or  eax, [esp+724]
        xor  [esp+712],eax
        mov  eax,ebp
        or  eax, [esp+720]
        mov  esi,ebp
        and  esi, [esp+712]
        xor  esi,eax
        mov  [esp+708],esi
        xor  [esp+708],ebx
        mov  ecx,ebp
        or  ecx, [esp+716]
        xor  ecx, [esp+720]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+704],ebx
        xor  ebx,ecx
        mov  ecx, [esp+708]
        xor  [esp+704],edi
        mov  edx,ebp
        or  edx, [esp+712]
        mov  eax,ebp
        and  eax, [esp+716]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+728]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+724]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+704]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+708]
        mov  eax,edx
        xor  eax, [esp+704]
        xor  ecx, [esp+720]
        mov  [esp+2012],ecx
        xor  edx, [esp+716]
        mov  [esp+2008],edx
        mov  edx, [esp+708]
        mov  ecx, [esp+704]
        or  eax,ebx
        mov  [esp+1988],ebx
        mov  ebx, [esp+1684]
        xor  ebp,esi
        mov  esi, [esp+1688]
        mov  [esp+1996],edi
        mov  edi, [esp+1692]
        mov  [esp+2000],ebp
        mov  ebp, [esp+1700]
        mov  [esp+1992],edx
        mov  edx, [esp+1656]
        xor  eax, [esp+712]
        mov  [esp+2004],eax
        xor  ebx, [esp+1668]
        xor  esi, [esp+1672]
        mov  [esp+688],esi
        xor  edi, [esp+1676]
        mov  [esp+700],edi
        xor  ebp, [esp+1680]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+1652]
        xor  esi, [esp+1660]
        or  edx,edi
        xor  edx,eax
        xor  ebx,edx
        mov  [esp+692],ebx
        mov  eax,ebp
        xor  eax,edx
        or  eax, [esp+1656]
        mov  edi,eax
        xor  eax, [esp+1652]
        or  eax,ebp
        xor  [esp+688],eax
        mov  eax,esi
        or  eax,ebx
        mov  ebx,esi
        and  ebx, [esp+688]
        xor  ebx,eax
        mov  eax, [esp+1656]
        mov  [esp+1984],ecx
        mov  [esp+696],ebp
        xor  edi, [esp+1664]
        mov  [esp+684],ebx
        xor  [esp+684],eax
        mov  edx, [esp+1652]
        mov  ecx,esi
        or  ecx,edi
        xor  ecx, [esp+692]
        mov  ebp,ecx
        or  ebp,ebx
        xor  ebp,esi
        mov  [esp+680],ebp
        xor  [esp+680],edx
        mov  edx,esi
        or  edx, [esp+688]
        mov  eax,esi
        and  eax,edi
        xor  eax,edx
        mov  edx, [esp+700]
        mov  [esp+676],eax
        or  eax,ebx
        mov  ebx, [esp+696]
        not  dword [esp+676]
        xor  [esp+676],edx
        mov  edx, [esp+676]
        xor  ecx,ebp
        mov  ebp, [esp+680]
        xor  ecx,eax
        mov  eax, [esp+684]
        not  edx
        mov  [esp+668],edx
        or  eax, [esp+676]
        xor  eax,edx
        mov  edx, [esp+692]
        not  ecx
        xor  ecx,ebx
        mov  [esp+672],ecx
        or  [esp+668],ebp
        xor  [esp+668],esi
        mov  [esp+664],eax
        xor  [esp+664],edx
        mov  edx,ecx
        xor  edx,eax
        or  edx, [esp+684]
        mov  [esp+660],edx
        xor  [esp+660],edi
        mov  [esp+656],ebp
        xor  [esp+656],edx
        or  [esp+656],ecx
        mov  ecx, [esp+688]
        mov  ebx, [esp+664]
        mov  esi, [esp+660]
        xor  [esp+656],ecx
        mov  edi, [esp+656]
        mov  ebp, [esp+668]
        mov  eax, [esp+676]
        mov  edx, [esp+684]
        mov  ecx, [esp+672]
        mov  [esp+1980],ebx
        mov  ebx, [esp+680]
        mov  [esp+1972],edi
        mov  edi, [esp+1824]
        mov  [esp+1976],esi
        mov  dword [esp+1704], csc_tabe
        mov  esi, [esp+1704]
        mov  [esp+1968],ebp
        mov  ebp, [esp+1828]
        mov  [esp+1964],eax
        mov  eax, [esp+1832]
        mov  [esp+1960],edx
        mov  [esp+1956],ecx
        mov  [esp+1952],ebx
        add  dword [esp+1716],BYTE 64
        xor  edi, [esi+32]
        mov  [esp+1700],edi
        xor  ebp, [esi+36]
        mov  [esp+1696],ebp
        xor  eax, [esi+40]
        mov  [esp+1692],eax
        mov  esi, [esp+1836]
        mov  edx, [esp+1704]
        mov  ecx, [esp+1840]
        mov  ebx, [esp+1704]
        mov  edi, [esp+1848]
        mov  ebp, [esp+1788]
        mov  eax, [esp+1784]
        xor  esi, [edx+44]
        xor  ecx, [edx+48]
        mov  edx, [esp+1844]
        mov  [esp+1688],ecx
        mov  ecx, [esp+1852]
        xor  edx, [ebx+52]
        xor  edi, [ebx+56]
        mov  [esp+1684],edi
        xor  ecx, [ebx+60]
        xor  ebp, [ebx+28]
        mov  [esp+1668],ebp
        mov  [esp+644],ecx
        xor  [esp+644],ebp
        mov  [esp+1664],edx
        xor  [esp+1664],eax
        mov  ebx, [ebx+24]
        xor  [esp+1664],ebx
        mov  [esp+640],edi
        mov  edi, [esp+1664]
        mov  ebp, [esp+1704]
        mov  eax, [esp+1780]
        xor  [esp+640],edi
        xor  eax, [ebp+20]
        mov  [esp+1672],eax
        mov  [esp+636],eax
        xor  [esp+636],edx
        mov  edx, [esp+1776]
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1772]
        mov  edi, [esp+1704]
        mov  edx, [esp+1696]
        xor  [esp+1660],ebx
        mov  ebx, [esp+1768]
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        mov  [esp+652],eax
        xor  [esp+652],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1764]
        xor  [esp+1656],esi
        xor  ebx, [esp+1656]
        xor  eax, [edi+4]
        mov  [esp+1680],eax
        mov  [esp+648],eax
        xor  [esp+648],edx
        mov  edx, [csc_tabe]
        mov  edi, [esp+1700]
        xor  ecx, [esp+1760]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+652]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+648]
        mov  edx,ebx
        or  edx, [esp+652]
        xor  edx,ecx
        xor  [esp+644],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+640],eax
        xor  eax,edi
        or  eax, [esp+648]
        xor  [esp+636],eax
        mov  eax,ebp
        or  eax, [esp+644]
        mov  esi,ebp
        and  esi, [esp+636]
        xor  esi,eax
        mov  [esp+632],esi
        xor  [esp+632],ebx
        mov  ecx,ebp
        or  ecx, [esp+640]
        xor  ecx, [esp+644]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+628],ebx
        xor  ebx,ecx
        mov  ecx, [esp+632]
        xor  [esp+628],edi
        mov  edx,ebp
        or  edx, [esp+636]
        mov  eax,ebp
        and  eax, [esp+640]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+652]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+648]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+628]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+632]
        mov  eax,edx
        xor  eax, [esp+628]
        xor  ecx, [esp+644]
        mov  [esp+1852],ecx
        mov  ecx, [esp+632]
        or  eax,ebx
        mov  [esp+1828],ebx
        mov  ebx, [esp+628]
        xor  ebp,esi
        mov  esi, [esp+1684]
        mov  [esp+1836],edi
        mov  edi, [esp+1688]
        mov  [esp+1840],ebp
        mov  ebp, [esp+1692]
        xor  eax, [esp+636]
        mov  [esp+1844],eax
        mov  eax, [esp+1700]
        xor  edx, [esp+640]
        mov  [esp+1848],edx
        mov  edx, [esp+1656]
        xor  ebp, [esp+1676]
        mov  [esp+624],ebp
        xor  eax, [esp+1680]
        mov  [esp+620],eax
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  edx, [esp+624]
        xor  edx,eax
        mov  eax, [esp+620]
        xor  esi, [esp+1668]
        xor  esi,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  [esp+1832],ecx
        mov  [esp+1824],ebx
        xor  edi, [esp+1672]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+616],esi
        or  eax, [esp+1656]
        mov  [esp+612],eax
        xor  [esp+612],edx
        xor  eax, [esp+1652]
        or  eax, [esp+620]
        xor  edi,eax
        mov  [esp+608],edi
        mov  ecx, [esp+1656]
        mov  eax,ebp
        or  eax,esi
        mov  esi,ebp
        and  esi,edi
        mov  edi, [esp+1652]
        xor  esi,eax
        mov  [esp+604],esi
        xor  [esp+604],ecx
        mov  ecx,ebp
        or  ecx, [esp+612]
        xor  ecx, [esp+616]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+600],ebx
        xor  ebx,ecx
        mov  ecx, [esp+604]
        xor  [esp+600],edi
        mov  edx,ebp
        or  edx, [esp+608]
        mov  eax,ebp
        and  eax, [esp+612]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+624]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+600]
        xor  ebp,esi
        mov  [esp+1776],ebp
        mov  ebp, [esp+604]
        not  ebx
        xor  ebx, [esp+620]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+604]
        mov  eax,edx
        xor  eax, [esp+600]
        or  eax,ebx
        xor  eax, [esp+608]
        mov  [esp+1780],eax
        mov  eax, [esp+600]
        xor  ecx, [esp+616]
        mov  [esp+1788],ecx
        mov  ecx, [esp+680]
        xor  edx, [esp+612]
        mov  [esp+1784],edx
        mov  dword [esp+1704],csc_tabe+64
        mov  edx, [esp+1704]
        mov  [esp+1772],edi
        mov  [esp+1768],ebp
        mov  [esp+1764],ebx
        mov  [esp+1760],eax
        xor  ecx, [edx+32]
        mov  [esp+1700],ecx
        mov  ebx, [esp+672]
        mov  esi, [esp+684]
        xor  ebx, [edx+36]
        mov  [esp+1696],ebx
        xor  esi, [edx+40]
        mov  [esp+1692],esi
        mov  esi, [esp+676]
        mov  edi, [esp+668]
        xor  esi, [edx+44]
        xor  edi, [edx+48]
        mov  edx, [esp+656]
        mov  ebp, [esp+1704]
        mov  eax, [esp+660]
        mov  ecx, [esp+664]
        mov  ebx, [esp+1916]
        mov  [esp+1688],edi
        mov  edi, [esp+1912]
        xor  edx, [ebp+52]
        xor  eax, [ebp+56]
        mov  [esp+1684],eax
        xor  ecx, [ebp+60]
        xor  ebx, [ebp+28]
        mov  [esp+1668],ebx
        mov  [esp+588],ecx
        xor  [esp+588],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [ebp+24]
        xor  [esp+1664],ebp
        mov  [esp+584],eax
        mov  eax, [esp+1664]
        xor  [esp+584],eax
        mov  ebx, [esp+1704]
        mov  edi, [esp+1908]
        mov  ebp, [esp+1904]
        xor  edi, [ebx+20]
        mov  [esp+1672],edi
        mov  [esp+580],edi
        xor  [esp+580],edx
        mov  [esp+1660],esi
        xor  [esp+1660],ebp
        mov  eax, [ebx+16]
        mov  ebp, [esp+1688]
        mov  edx, [esp+1900]
        xor  [esp+1660],eax
        xor  edx, [ebx+12]
        mov  ebx, [esp+1696]
        mov  [esp+596],edx
        xor  [esp+596],esi
        mov  esi, [esp+1896]
        mov  edi, [esp+1704]
        mov  [esp+1656],ebx
        mov  ebx, [esp+1692]
        mov  [esp+1676],edx
        mov  edx, [esp+1892]
        mov  eax, [esp+1704]
        xor  ebp, [esp+1660]
        xor  [esp+1656],esi
        mov  edi, [edi+8]
        xor  [esp+1656],edi
        xor  ebx, [esp+1656]
        xor  edx, [eax+4]
        mov  [esp+1680],edx
        mov  esi, [esp+1696]
        mov  edi, [csc_tabe+64]
        xor  ecx, [esp+1888]
        xor  ecx,edi
        mov  edi, [esp+1700]
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+596]
        mov  [esp+592],edx
        xor  [esp+592],esi
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+592]
        mov  edx,ebx
        or  edx, [esp+596]
        xor  edx,ecx
        xor  [esp+588],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+584],eax
        xor  eax,edi
        or  eax, [esp+592]
        xor  [esp+580],eax
        mov  eax,ebp
        or  eax, [esp+588]
        mov  esi,ebp
        and  esi, [esp+580]
        xor  esi,eax
        mov  [esp+576],esi
        xor  [esp+576],ebx
        mov  ecx,ebp
        or  ecx, [esp+584]
        xor  ecx, [esp+588]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+572],ebx
        xor  ebx,ecx
        mov  ecx, [esp+576]
        xor  [esp+572],edi
        mov  edx,ebp
        or  edx, [esp+580]
        mov  eax,ebp
        and  eax, [esp+584]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+596]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+592]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+572]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+576]
        mov  eax,edx
        xor  eax, [esp+572]
        xor  ecx, [esp+588]
        mov  [esp+1980],ecx
        xor  ebp,esi
        mov  [esp+1968],ebp
        mov  ebp, [esp+576]
        or  eax,ebx
        xor  eax, [esp+580]
        mov  [esp+1972],eax
        mov  eax, [esp+572]
        xor  edx, [esp+584]
        mov  [esp+1976],edx
        mov  edx, [esp+1684]
        mov  ecx, [esp+1688]
        mov  [esp+1956],ebx
        mov  ebx, [esp+1692]
        mov  esi, [esp+1700]
        xor  edx, [esp+1668]
        mov  [esp+560],edx
        mov  edx, [esp+1656]
        mov  [esp+1964],edi
        mov  edi, [esp+1664]
        mov  [esp+1960],ebp
        mov  [esp+1952],eax
        xor  ecx, [esp+1672]
        xor  ebx, [esp+1676]
        mov  [esp+568],ebx
        xor  esi, [esp+1680]
        mov  [esp+564],esi
        mov  eax,ebx
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx,ebx
        xor  edx,eax
        xor  [esp+560],edx
        mov  eax,esi
        xor  eax,edx
        or  eax, [esp+1656]
        mov  [esp+556],eax
        xor  [esp+556],edi
        xor  eax, [esp+1652]
        or  eax,esi
        xor  ecx,eax
        mov  [esp+552],ecx
        mov  eax,ebp
        or  eax, [esp+560]
        mov  esi,ebp
        and  esi,ecx
        xor  esi,eax
        mov  eax, [esp+1656]
        mov  edx, [esp+1652]
        mov  [esp+548],esi
        xor  [esp+548],eax
        mov  ecx,ebp
        or  ecx, [esp+556]
        xor  ecx, [esp+560]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+544],ebx
        xor  ebx,ecx
        mov  ecx, [esp+548]
        xor  [esp+544],edx
        mov  edx,ebp
        or  edx, [esp+552]
        mov  eax,ebp
        and  eax, [esp+556]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+568]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+560]
        mov  [esp+1916],ecx
        mov  ecx, [esp+548]
        not  ebx
        xor  ebx, [esp+564]
        mov  esi,eax
        xor  edx,ebx
        or  edx, [esp+548]
        mov  eax,edx
        xor  eax, [esp+544]
        or  eax,ebx
        mov  [esp+1892],ebx
        mov  ebx, [esp+544]
        or  esi, [esp+544]
        xor  edx, [esp+556]
        mov  [esp+1912],edx
        xor  eax, [esp+552]
        mov  [esp+1908],eax
        xor  ebp,esi
        mov  [esp+1904],ebp
        mov  [esp+1900],edi
        mov  [esp+1896],ecx
        mov  [esp+540],ebx
        mov  [esp+1888],ebx
        add  dword [esp+1704],BYTE 64
        mov  edi, [esp+1856]
        mov  esi, [esp+1704]
        mov  ebp, [esp+1860]
        mov  eax, [esp+1864]
        mov  edx, [esp+1704]
        mov  ecx, [esp+1872]
        mov  ebx, [esp+1704]
        xor  edi, [esi+32]
        mov  [esp+1700],edi
        xor  ebp, [esi+36]
        mov  [esp+1696],ebp
        xor  eax, [esi+40]
        mov  esi, [esp+1868]
        mov  [esp+1692],eax
        xor  esi, [edx+44]
        xor  ecx, [edx+48]
        mov  edx, [esp+1876]
        mov  edi, [esp+1880]
        mov  [esp+1688],ecx
        mov  ecx, [esp+1884]
        mov  ebp, [esp+1820]
        mov  eax, [esp+1816]
        xor  edx, [ebx+52]
        xor  edi, [ebx+56]
        mov  [esp+1684],edi
        xor  ecx, [ebx+60]
        xor  ebp, [ebx+28]
        mov  [esp+1668],ebp
        mov  [esp+528],ecx
        xor  [esp+528],ebp
        mov  [esp+1664],edx
        xor  [esp+1664],eax
        mov  ebx, [ebx+24]
        xor  [esp+1664],ebx
        mov  [esp+524],edi
        mov  edi, [esp+1664]
        mov  eax, [esp+1812]
        mov  ebp, [esp+1704]
        xor  [esp+524],edi
        mov  edi, [esp+1704]
        xor  eax, [ebp+20]
        mov  [esp+520],eax
        xor  [esp+520],edx
        mov  edx, [esp+1808]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1804]
        mov  edx, [esp+1696]
        xor  [esp+1660],ebx
        mov  ebx, [esp+1800]
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        mov  [esp+536],eax
        xor  [esp+536],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1796]
        xor  [esp+1656],esi
        xor  eax, [edi+4]
        mov  [esp+1680],eax
        mov  [esp+532],eax
        xor  [esp+532],edx
        mov  edx, [edi]
        mov  edi, [esp+1700]
        xor  ecx, [esp+1792]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+536]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+532]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+536]
        xor  edx,ecx
        xor  [esp+528],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+524],eax
        xor  eax,edi
        or  eax, [esp+532]
        xor  [esp+520],eax
        mov  eax,ebp
        or  eax, [esp+528]
        mov  esi,ebp
        and  esi, [esp+520]
        xor  esi,eax
        mov  [esp+516],esi
        xor  [esp+516],ebx
        mov  ecx,ebp
        or  ecx, [esp+524]
        xor  ecx, [esp+528]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+512],ebx
        xor  ebx,ecx
        mov  ecx, [esp+516]
        xor  [esp+512],edi
        mov  edx,ebp
        or  edx, [esp+520]
        mov  eax,ebp
        and  eax, [esp+524]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+536]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+528]
        mov  [esp+1884],ecx
        mov  ecx, [esp+516]
        not  ebx
        xor  ebx, [esp+532]
        mov  esi,eax
        xor  edx,ebx
        or  edx, [esp+516]
        mov  eax,edx
        xor  eax, [esp+512]
        or  eax,ebx
        mov  [esp+1860],ebx
        mov  ebx, [esp+512]
        or  esi, [esp+512]
        xor  ebp,esi
        mov  esi, [esp+1684]
        mov  [esp+1868],edi
        mov  edi, [esp+1688]
        mov  [esp+1872],ebp
        mov  ebp, [esp+1692]
        xor  eax, [esp+520]
        mov  [esp+1876],eax
        mov  eax, [esp+1700]
        xor  edx, [esp+524]
        mov  [esp+1880],edx
        mov  edx, [esp+1656]
        mov  [esp+1864],ecx
        mov  [esp+1856],ebx
        xor  esi, [esp+1668]
        xor  edi, [esp+1672]
        xor  ebp, [esp+1676]
        mov  [esp+508],ebp
        xor  eax, [esp+1680]
        mov  [esp+504],eax
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx, [esp+508]
        xor  edx,eax
        xor  esi,edx
        mov  [esp+500],esi
        mov  eax, [esp+504]
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  ecx, [esp+1656]
        or  eax, [esp+1656]
        mov  [esp+496],eax
        xor  eax, [esp+1652]
        or  eax, [esp+504]
        xor  edi,eax
        mov  [esp+492],edi
        mov  eax,ebp
        or  eax,esi
        mov  esi,ebp
        and  esi,edi
        mov  edi, [esp+1652]
        xor  [esp+496],edx
        xor  esi,eax
        mov  [esp+488],esi
        xor  [esp+488],ecx
        mov  ecx,ebp
        or  ecx, [esp+496]
        xor  ecx, [esp+500]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+484],ebx
        xor  ebx,ecx
        mov  ecx, [esp+488]
        xor  [esp+484],edi
        mov  edx,ebp
        or  edx, [esp+492]
        mov  eax,ebp
        and  eax, [esp+496]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+508]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+504]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+484]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+488]
        mov  eax,edx
        xor  eax, [esp+484]
        or  eax,ebx
        xor  ecx, [esp+500]
        mov  [esp+1820],ecx
        xor  edx, [esp+496]
        mov  [esp+1816],edx
        xor  eax, [esp+492]
        mov  [esp+1812],eax
        xor  ebp,esi
        mov  [esp+1808],ebp
        mov  ebp, [esp+488]
        mov  eax, [esp+484]
        mov  ecx, [esp+1984]
        add  dword [esp+1704],BYTE 64
        mov  edx, [esp+1704]
        mov  [esp+1796],ebx
        mov  ebx, [esp+1988]
        mov  esi, [esp+1992]
        mov  [esp+1804],edi
        mov  edi, [esp+2000]
        mov  [esp+1800],ebp
        mov  ebp, [esp+1704]
        mov  [esp+1792],eax
        mov  eax, [esp+2008]
        xor  ecx, [edx+32]
        mov  [esp+1700],ecx
        xor  ebx, [edx+36]
        mov  [esp+1696],ebx
        xor  esi, [edx+40]
        mov  [esp+1692],esi
        mov  esi, [esp+1996]
        xor  esi, [edx+44]
        xor  edi, [edx+48]
        mov  edx, [esp+2004]
        mov  ecx, [esp+2012]
        mov  ebx, [esp+1948]
        mov  [esp+1688],edi
        xor  edx, [ebp+52]
        xor  eax, [ebp+56]
        mov  [esp+1684],eax
        xor  ecx, [ebp+60]
        xor  ebx, [ebp+28]
        mov  [esp+1668],ebx
        mov  edi, [esp+1944]
        mov  [esp+472],ecx
        xor  [esp+472],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [ebp+24]
        xor  [esp+1664],ebp
        mov  [esp+468],eax
        mov  eax, [esp+1664]
        mov  edi, [esp+1940]
        mov  ebx, [esp+1704]
        mov  ebp, [esp+1936]
        xor  [esp+468],eax
        xor  edi, [ebx+20]
        mov  [esp+1672],edi
        mov  [esp+464],edi
        xor  [esp+464],edx
        mov  [esp+1660],esi
        xor  [esp+1660],ebp
        mov  eax, [ebx+16]
        mov  ebp, [esp+1688]
        mov  edx, [esp+1932]
        xor  [esp+1660],eax
        xor  ebp, [esp+1660]
        xor  edx, [ebx+12]
        mov  [esp+1676],edx
        mov  [esp+480],edx
        mov  ebx, [esp+1696]
        xor  [esp+480],esi
        mov  esi, [esp+1928]
        mov  edi, [esp+1704]
        mov  [esp+1656],ebx
        mov  ebx, [esp+1692]
        mov  edx, [esp+1924]
        mov  eax, [esp+1704]
        xor  [esp+1656],esi
        mov  esi, [esp+1696]
        xor  ecx, [esp+1920]
        mov  edi, [edi+8]
        xor  [esp+1656],edi
        xor  edx, [eax+4]
        mov  [esp+1680],edx
        mov  [esp+476],edx
        xor  [esp+476],esi
        mov  edi, [eax]
        xor  ecx,edi
        mov  edi, [esp+1700]
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+480]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+476]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+480]
        xor  edx,ecx
        xor  [esp+472],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+468],eax
        xor  eax,edi
        or  eax, [esp+476]
        xor  [esp+464],eax
        mov  eax,ebp
        or  eax, [esp+472]
        mov  esi,ebp
        and  esi, [esp+464]
        xor  esi,eax
        mov  [esp+460],esi
        xor  [esp+460],ebx
        mov  ecx,ebp
        or  ecx, [esp+468]
        xor  ecx, [esp+472]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+456],ebx
        xor  ebx,ecx
        mov  ecx, [esp+460]
        xor  [esp+456],edi
        mov  edx,ebp
        or  edx, [esp+464]
        mov  eax,ebp
        and  eax, [esp+468]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+480]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+456]
        xor  ebp,esi
        mov  [esp+2000],ebp
        mov  ebp, [esp+460]
        not  ebx
        xor  ebx, [esp+476]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+460]
        mov  eax,edx
        xor  eax, [esp+456]
        or  eax,ebx
        xor  eax, [esp+464]
        mov  [esp+2004],eax
        mov  eax, [esp+456]
        xor  edx, [esp+468]
        mov  [esp+2008],edx
        mov  edx, [esp+1684]
        xor  ecx, [esp+472]
        mov  [esp+2012],ecx
        mov  [esp+1996],edi
        mov  [esp+1992],ebp
        mov  [esp+1988],ebx
        mov  [esp+1984],eax
        xor  edx, [esp+1668]
        mov  [esp+444],edx
        mov  ecx, [esp+1688]
        mov  ebx, [esp+1692]
        mov  esi, [esp+1700]
        mov  edx, [esp+1656]
        mov  edi, [esp+1664]
        xor  ecx, [esp+1672]
        xor  ebx, [esp+1676]
        xor  esi, [esp+1680]
        mov  [esp+448],esi
        mov  eax,ebx
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx,ebx
        xor  edx,eax
        xor  [esp+444],edx
        mov  eax,esi
        xor  eax,edx
        or  eax, [esp+1656]
        mov  [esp+440],eax
        xor  eax, [esp+1652]
        or  eax,esi
        xor  ecx,eax
        mov  eax,ebp
        or  eax, [esp+444]
        mov  esi,ebp
        and  esi,ecx
        xor  esi,eax
        mov  eax, [esp+1656]
        mov  edx, [esp+1652]
        mov  [esp+452],ebx
        xor  [esp+440],edi
        mov  [esp+436],ecx
        mov  [esp+432],esi
        xor  [esp+432],eax
        mov  ecx,ebp
        or  ecx, [esp+440]
        xor  ecx, [esp+444]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+428],ebx
        xor  ebx,ecx
        mov  ecx, [esp+432]
        xor  [esp+428],edx
        mov  edx,ebp
        or  edx, [esp+436]
        mov  eax,ebp
        and  eax, [esp+440]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+452]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+448]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+428]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+432]
        mov  eax,edx
        xor  eax, [esp+428]
        xor  ecx, [esp+444]
        mov  [esp+1948],ecx
        mov  ecx, [esp+432]
        or  eax,ebx
        mov  [esp+1924],ebx
        mov  ebx, [esp+428]
        mov  [esp+1932],edi
        mov  edi, [esp+540]
        xor  ebp,esi
        add  dword [esp+1704],BYTE 64
        mov  esi, [esp+1704]
        mov  [esp+1936],ebp
        mov  ebp, [esp+1892]
        xor  eax, [esp+436]
        mov  [esp+1940],eax
        mov  eax, [esp+1896]
        xor  edx, [esp+440]
        mov  [esp+1944],edx
        mov  edx, [esp+1704]
        mov  [esp+1928],ecx
        mov  ecx, [esp+1904]
        mov  [esp+1920],ebx
        mov  ebx, [esp+1704]
        xor  edi, [esi+32]
        mov  [esp+1700],edi
        xor  ebp, [esi+36]
        mov  [esp+1696],ebp
        xor  eax, [esi+40]
        mov  esi, [esp+1900]
        mov  [esp+1692],eax
        xor  esi, [edx+44]
        xor  ecx, [edx+48]
        mov  edx, [esp+1908]
        mov  edi, [esp+1912]
        mov  [esp+1688],ecx
        xor  edx, [ebx+52]
        xor  edi, [ebx+56]
        mov  [esp+1684],edi
        mov  ecx, [esp+1916]
        mov  ebp, [esp+1788]
        mov  eax, [esp+1784]
        xor  ecx, [ebx+60]
        xor  ebp, [ebx+28]
        mov  [esp+1668],ebp
        mov  [esp+416],ecx
        xor  [esp+416],ebp
        mov  [esp+1664],edx
        xor  [esp+1664],eax
        mov  ebx, [ebx+24]
        xor  [esp+1664],ebx
        mov  [esp+412],edi
        mov  edi, [esp+1664]
        mov  eax, [esp+1780]
        mov  ebp, [esp+1704]
        xor  [esp+412],edi
        mov  edi, [esp+1704]
        xor  eax, [ebp+20]
        mov  [esp+408],eax
        xor  [esp+408],edx
        mov  edx, [esp+1776]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1772]
        xor  [esp+1660],ebx
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        mov  edx, [esp+1696]
        mov  ebx, [esp+1768]
        mov  [esp+424],eax
        xor  [esp+424],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1764]
        xor  [esp+1656],esi
        xor  eax, [edi+4]
        mov  [esp+1680],eax
        mov  [esp+420],eax
        xor  [esp+420],edx
        mov  edx, [edi]
        mov  edi, [esp+1700]
        xor  ecx, [esp+1760]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+424]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+420]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+424]
        xor  edx,ecx
        xor  [esp+416],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+412],eax
        xor  eax,edi
        or  eax, [esp+420]
        xor  [esp+408],eax
        mov  eax,ebp
        or  eax, [esp+416]
        mov  esi,ebp
        and  esi, [esp+408]
        xor  esi,eax
        mov  [esp+404],esi
        xor  [esp+404],ebx
        mov  ecx,ebp
        or  ecx, [esp+412]
        xor  ecx, [esp+416]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+400],ebx
        xor  ebx,ecx
        mov  ecx, [esp+404]
        xor  [esp+400],edi
        mov  edx,ebp
        or  edx, [esp+408]
        mov  eax,ebp
        and  eax, [esp+412]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+424]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+416]
        mov  [esp+1916],ecx
        mov  ecx, [esp+404]
        not  ebx
        xor  ebx, [esp+420]
        mov  esi,eax
        xor  edx,ebx
        or  edx, [esp+404]
        mov  eax,edx
        xor  eax, [esp+400]
        or  eax,ebx
        mov  [esp+1892],ebx
        mov  ebx, [esp+400]
        or  esi, [esp+400]
        xor  ebp,esi
        mov  esi, [esp+1684]
        mov  [esp+1900],edi
        mov  edi, [esp+1688]
        mov  [esp+1904],ebp
        mov  ebp, [esp+1692]
        xor  edx, [esp+412]
        mov  [esp+1912],edx
        xor  eax, [esp+408]
        mov  [esp+1908],eax
        mov  [esp+1896],ecx
        mov  [esp+1888],ebx
        xor  esi, [esp+1668]
        xor  edi, [esp+1672]
        xor  ebp, [esp+1676]
        mov  [esp+396],ebp
        mov  ecx, [esp+1700]
        mov  edx, [esp+1656]
        xor  ecx, [esp+1680]
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  edx, [esp+396]
        xor  edx,eax
        xor  esi,edx
        mov  eax,ecx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  [esp+1736],ecx
        or  eax, [esp+1656]
        mov  [esp+388],eax
        xor  eax, [esp+1652]
        or  eax,ecx
        mov  ecx, [esp+1656]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+392],esi
        xor  edi,eax
        mov  [esp+384],edi
        mov  eax,ebp
        or  eax,esi
        mov  esi,ebp
        and  esi,edi
        mov  edi, [esp+1652]
        xor  [esp+388],edx
        xor  esi,eax
        mov  [esp+380],esi
        xor  [esp+380],ecx
        mov  ebx,ebp
        or  ebx, [esp+388]
        xor  ebx, [esp+392]
        mov  edx,ebx
        or  edx,esi
        xor  edx,ebp
        mov  [esp+376],edx
        xor  [esp+376],edi
        mov  ecx,ebp
        or  ecx, [esp+384]
        mov  eax,ebp
        and  eax, [esp+388]
        xor  eax,ecx
        mov  edi,eax
        or  eax,esi
        xor  edx,ebx
        xor  edx,eax
        mov  eax, [esp+164]
        not  edi
        xor  edi, [esp+396]
        not  edx
        xor  [eax],edx
        mov  esi, [esp+1736]
        mov  ecx, [esp+380]
        mov  eax,edi
        not  eax
        mov  ebx,eax
        or  ebx, [esp+376]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,esi
        or  edx, [esp+380]
        mov  eax,edx
        xor  eax, [esp+376]
        xor  ecx, [esp+392]
        mov  [esp+1788],ecx
        xor  edx, [esp+388]
        mov  [esp+1784],edx
        mov  edx, [esp+380]
        mov  ecx, [esp+376]
        or  eax,esi
        mov  [esp+1764],esi
        mov  esi, [esp+1920]
        xor  ebp,ebx
        add  dword [esp+1704],BYTE 64
        mov  ebx, [esp+1704]
        mov  [esp+1772],edi
        mov  edi, [esp+1924]
        mov  [esp+1776],ebp
        mov  ebp, [esp+1928]
        xor  eax, [esp+384]
        mov  [esp+1780],eax
        mov  eax, [esp+1936]
        mov  [esp+1768],edx
        mov  edx, [esp+1940]
        mov  [esp+1760],ecx
        mov  ecx, [esp+1944]
        xor  esi, [ebx+32]
        mov  [esp+1700],esi
        mov  esi, [esp+1932]
        xor  edi, [ebx+36]
        mov  [esp+1696],edi
        xor  ebp, [ebx+40]
        mov  [esp+1692],ebp
        xor  esi, [ebx+44]
        xor  eax, [ebx+48]
        mov  [esp+1688],eax
        xor  edx, [ebx+52]
        xor  ecx, [ebx+56]
        mov  [esp+1684],ecx
        mov  ecx, [esp+1948]
        mov  edi, [esp+1820]
        xor  ecx, [ebx+60]
        xor  edi, [ebx+28]
        mov  [esp+1668],edi
        mov  ebp, [esp+1816]
        mov  [esp+364],ecx
        xor  [esp+364],edi
        mov  [esp+1664],edx
        xor  [esp+1664],ebp
        mov  eax, [ebx+24]
        mov  ebx, [esp+1684]
        xor  [esp+1664],eax
        mov  edi, [esp+1664]
        mov  eax, [esp+1812]
        mov  ebp, [esp+1704]
        mov  [esp+360],ebx
        xor  [esp+360],edi
        mov  edi, [esp+1704]
        xor  eax, [ebp+20]
        mov  [esp+356],eax
        xor  [esp+356],edx
        mov  edx, [esp+1808]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1804]
        xor  [esp+1660],ebx
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        mov  edx, [esp+1696]
        mov  ebx, [esp+1800]
        mov  [esp+372],eax
        xor  [esp+372],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1796]
        xor  [esp+1656],esi
        xor  eax, [edi+4]
        mov  [esp+1680],eax
        mov  [esp+368],eax
        xor  [esp+368],edx
        mov  edx, [edi]
        mov  edi, [esp+1700]
        xor  ecx, [esp+1792]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+372]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+368]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+372]
        xor  edx,ecx
        xor  [esp+364],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+360],eax
        xor  eax,edi
        or  eax, [esp+368]
        xor  [esp+356],eax
        mov  eax,ebp
        or  eax, [esp+364]
        mov  esi,ebp
        and  esi, [esp+356]
        xor  esi,eax
        mov  [esp+352],esi
        xor  [esp+352],ebx
        mov  ecx,ebp
        or  ecx, [esp+360]
        xor  ecx, [esp+364]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+348],ebx
        xor  ebx,ecx
        mov  ecx, [esp+352]
        xor  [esp+348],edi
        mov  edx,ebp
        or  edx, [esp+356]
        mov  eax,ebp
        and  eax, [esp+360]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+372]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+364]
        mov  [esp+1948],ecx
        mov  ecx, [esp+352]
        not  ebx
        xor  ebx, [esp+368]
        mov  esi,eax
        xor  edx,ebx
        or  edx, [esp+352]
        mov  eax,edx
        xor  eax, [esp+348]
        or  eax,ebx
        mov  [esp+1924],ebx
        mov  ebx, [esp+348]
        or  esi, [esp+348]
        xor  ebp,esi
        mov  esi, [esp+1684]
        mov  [esp+1932],edi
        mov  edi, [esp+1688]
        mov  [esp+1936],ebp
        mov  ebp, [esp+1692]
        xor  edx, [esp+360]
        mov  [esp+1944],edx
        xor  eax, [esp+356]
        mov  [esp+1940],eax
        mov  [esp+1928],ecx
        mov  [esp+1920],ebx
        xor  esi, [esp+1668]
        xor  edi, [esp+1672]
        xor  ebp, [esp+1676]
        mov  [esp+344],ebp
        mov  eax, [esp+1700]
        mov  edx, [esp+1656]
        xor  eax, [esp+1680]
        mov  [esp+340],eax
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  edx, [esp+344]
        xor  edx,eax
        mov  eax, [esp+340]
        xor  esi,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  ecx, [esp+1656]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+336],esi
        or  eax, [esp+1656]
        mov  [esp+332],eax
        xor  eax, [esp+1652]
        or  eax, [esp+340]
        xor  edi,eax
        mov  [esp+328],edi
        mov  eax,ebp
        or  eax,esi
        mov  esi,ebp
        and  esi,edi
        mov  edi, [esp+1652]
        xor  [esp+332],edx
        xor  esi,eax
        mov  [esp+324],esi
        xor  [esp+324],ecx
        mov  ecx,ebp
        or  ecx, [esp+332]
        xor  ecx, [esp+336]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+320],ebx
        xor  ebx,ecx
        mov  ecx, [esp+324]
        xor  [esp+320],edi
        mov  edx,ebp
        or  edx, [esp+328]
        mov  eax,ebp
        and  eax, [esp+332]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+344]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+340]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+320]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+324]
        mov  eax,edx
        xor  eax, [esp+320]
        xor  ecx, [esp+336]
        mov  [esp+1820],ecx
        xor  ebp,esi
        mov  [esp+1808],ebp
        mov  ebp, [esp+324]
        or  eax,ebx
        xor  eax, [esp+328]
        mov  [esp+1812],eax
        mov  eax, [esp+320]
        mov  ecx, [esp+1952]
        xor  edx, [esp+332]
        mov  [esp+1816],edx
        add  dword [esp+1704],BYTE 64
        mov  edx, [esp+1704]
        mov  [esp+1796],ebx
        mov  ebx, [esp+1956]
        mov  esi, [esp+1960]
        mov  [esp+1804],edi
        mov  edi, [esp+1968]
        mov  [esp+1800],ebp
        mov  ebp, [esp+1704]
        mov  [esp+1792],eax
        mov  eax, [esp+1976]
        xor  ecx, [edx+32]
        mov  [esp+1700],ecx
        xor  ebx, [edx+36]
        mov  [esp+1696],ebx
        xor  esi, [edx+40]
        mov  [esp+1692],esi
        mov  esi, [esp+1964]
        xor  esi, [edx+44]
        xor  edi, [edx+48]
        mov  edx, [esp+1972]
        mov  [esp+1688],edi
        xor  edx, [ebp+52]
        xor  eax, [ebp+56]
        mov  [esp+1684],eax
        mov  ecx, [esp+1980]
        mov  ebx, [esp+1852]
        mov  edi, [esp+1848]
        xor  ecx, [ebp+60]
        xor  ebx, [ebp+28]
        mov  [esp+1668],ebx
        mov  [esp+308],ecx
        xor  [esp+308],ebx
        mov  [esp+1664],edx
        xor  [esp+1664],edi
        mov  ebp, [ebp+24]
        xor  [esp+1664],ebp
        mov  [esp+304],eax
        mov  eax, [esp+1664]
        mov  edi, [esp+1844]
        mov  ebx, [esp+1704]
        mov  ebp, [esp+1840]
        xor  [esp+304],eax
        xor  edi, [ebx+20]
        mov  [esp+1672],edi
        mov  [esp+300],edi
        xor  [esp+300],edx
        mov  [esp+1660],esi
        xor  [esp+1660],ebp
        mov  eax, [ebx+16]
        mov  ebp, [esp+1688]
        mov  edx, [esp+1836]
        xor  [esp+1660],eax
        xor  ebp, [esp+1660]
        xor  edx, [ebx+12]
        mov  [esp+1676],edx
        mov  ebx, [esp+1696]
        mov  [esp+316],edx
        xor  [esp+316],esi
        mov  esi, [esp+1832]
        mov  edi, [esp+1704]
        mov  [esp+1656],ebx
        mov  ebx, [esp+1692]
        mov  edx, [esp+1828]
        mov  eax, [esp+1704]
        xor  [esp+1656],esi
        mov  esi, [esp+1696]
        xor  ecx, [esp+1824]
        mov  edi, [edi+8]
        xor  [esp+1656],edi
        xor  edx, [eax+4]
        mov  [esp+1680],edx
        mov  [esp+312],edx
        xor  [esp+312],esi
        mov  edi, [eax]
        xor  ecx,edi
        mov  edi, [esp+1700]
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+316]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+312]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+316]
        xor  edx,ecx
        xor  [esp+308],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+304],eax
        xor  eax,edi
        or  eax, [esp+312]
        xor  [esp+300],eax
        mov  eax,ebp
        or  eax, [esp+308]
        mov  esi,ebp
        and  esi, [esp+300]
        xor  esi,eax
        mov  [esp+296],esi
        xor  [esp+296],ebx
        mov  ecx,ebp
        or  ecx, [esp+304]
        xor  ecx, [esp+308]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+292],ebx
        xor  ebx,ecx
        mov  ecx, [esp+296]
        xor  [esp+292],edi
        mov  edx,ebp
        or  edx, [esp+300]
        mov  eax,ebp
        and  eax, [esp+304]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+316]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+292]
        xor  ebp,esi
        mov  [esp+1968],ebp
        mov  ebp, [esp+296]
        not  ebx
        xor  ebx, [esp+312]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+296]
        mov  eax,edx
        xor  eax, [esp+292]
        or  eax,ebx
        xor  eax, [esp+300]
        mov  [esp+1972],eax
        mov  eax, [esp+292]
        xor  edx, [esp+304]
        mov  [esp+1976],edx
        mov  edx, [esp+1684]
        xor  ecx, [esp+308]
        mov  [esp+1980],ecx
        mov  [esp+1964],edi
        mov  [esp+1960],ebp
        mov  [esp+1956],ebx
        mov  [esp+1952],eax
        xor  edx, [esp+1668]
        mov  [esp+280],edx
        mov  ecx, [esp+1688]
        mov  ebx, [esp+1692]
        mov  esi, [esp+1700]
        mov  edx, [esp+1656]
        mov  edi, [esp+1664]
        xor  ecx, [esp+1672]
        xor  ebx, [esp+1676]
        xor  esi, [esp+1680]
        mov  [esp+284],esi
        mov  eax,ebx
        not  eax
        mov  ebp,eax
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        or  edx,ebx
        xor  edx,eax
        xor  [esp+280],edx
        mov  eax,esi
        xor  eax,edx
        or  eax, [esp+1656]
        mov  [esp+276],eax
        xor  eax, [esp+1652]
        or  eax,esi
        xor  ecx,eax
        mov  eax,ebp
        or  eax, [esp+280]
        mov  esi,ebp
        and  esi,ecx
        xor  esi,eax
        mov  eax, [esp+1656]
        mov  edx, [esp+1652]
        mov  [esp+288],ebx
        xor  [esp+276],edi
        mov  [esp+272],ecx
        mov  [esp+268],esi
        xor  [esp+268],eax
        mov  ecx,ebp
        or  ecx, [esp+276]
        xor  ecx, [esp+280]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+264],ebx
        xor  ebx,ecx
        mov  ecx, [esp+268]
        xor  [esp+264],edx
        mov  edx,ebp
        or  edx, [esp+272]
        mov  eax,ebp
        and  eax, [esp+276]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+288]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+284]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+264]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+268]
        mov  eax,edx
        xor  eax, [esp+264]
        xor  ecx, [esp+280]
        mov  [esp+1852],ecx
        mov  ecx, [esp+268]
        or  eax,ebx
        mov  [esp+1828],ebx
        mov  ebx, [esp+264]
        mov  [esp+1836],edi
        mov  edi, [esp+1984]
        xor  ebp,esi
        add  dword [esp+1704],BYTE 64
        mov  esi, [esp+1704]
        mov  [esp+1840],ebp
        mov  ebp, [esp+1988]
        xor  eax, [esp+272]
        mov  [esp+1844],eax
        mov  eax, [esp+1992]
        xor  edx, [esp+276]
        mov  [esp+1848],edx
        mov  edx, [esp+1704]
        mov  [esp+1832],ecx
        mov  ecx, [esp+2000]
        mov  [esp+1824],ebx
        mov  ebx, [esp+1704]
        xor  edi, [esi+32]
        mov  [esp+1700],edi
        xor  ebp, [esi+36]
        mov  [esp+1696],ebp
        xor  eax, [esi+40]
        mov  esi, [esp+1996]
        mov  [esp+1692],eax
        xor  esi, [edx+44]
        xor  ecx, [edx+48]
        mov  edx, [esp+2004]
        mov  edi, [esp+2008]
        mov  [esp+1688],ecx
        xor  edx, [ebx+52]
        xor  edi, [ebx+56]
        mov  [esp+1684],edi
        mov  ecx, [esp+2012]
        mov  ebp, [esp+1884]
        mov  eax, [esp+1880]
        xor  ecx, [ebx+60]
        xor  ebp, [ebx+28]
        mov  [esp+1668],ebp
        mov  [esp+252],ecx
        xor  [esp+252],ebp
        mov  [esp+1664],edx
        xor  [esp+1664],eax
        mov  ebx, [ebx+24]
        xor  [esp+1664],ebx
        mov  [esp+248],edi
        mov  edi, [esp+1664]
        mov  eax, [esp+1876]
        mov  ebp, [esp+1704]
        xor  [esp+248],edi
        mov  edi, [esp+1704]
        xor  eax, [ebp+20]
        mov  [esp+244],eax
        xor  [esp+244],edx
        mov  edx, [esp+1872]
        mov  [esp+1672],eax
        mov  [esp+1660],esi
        xor  [esp+1660],edx
        mov  ebx, [ebp+16]
        mov  ebp, [esp+1688]
        mov  eax, [esp+1868]
        xor  [esp+1660],ebx
        xor  ebp, [esp+1660]
        xor  eax, [edi+12]
        mov  [esp+1676],eax
        mov  edx, [esp+1696]
        mov  ebx, [esp+1864]
        mov  [esp+260],eax
        xor  [esp+260],esi
        mov  [esp+1656],edx
        xor  [esp+1656],ebx
        mov  esi, [edi+8]
        mov  ebx, [esp+1692]
        mov  eax, [esp+1860]
        xor  [esp+1656],esi
        xor  eax, [edi+4]
        mov  [esp+1680],eax
        mov  [esp+256],eax
        xor  [esp+256],edx
        mov  edx, [edi]
        mov  edi, [esp+1700]
        xor  ecx, [esp+1856]
        xor  ecx,edx
        mov  [esp+1652],ecx
        xor  edi,ecx
        mov  ecx, [esp+260]
        not  ecx
        mov  eax,ecx
        or  eax,edi
        xor  ebp,eax
        mov  eax, [esp+256]
        xor  ebx, [esp+1656]
        mov  edx,ebx
        or  edx, [esp+260]
        xor  edx,ecx
        xor  [esp+252],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+248],eax
        xor  eax,edi
        or  eax, [esp+256]
        xor  [esp+244],eax
        mov  eax,ebp
        or  eax, [esp+252]
        mov  esi,ebp
        and  esi, [esp+244]
        xor  esi,eax
        mov  [esp+240],esi
        xor  [esp+240],ebx
        mov  ecx,ebp
        or  ecx, [esp+248]
        xor  ecx, [esp+252]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+236],ebx
        xor  ebx,ecx
        mov  ecx, [esp+240]
        xor  [esp+236],edi
        mov  edx,ebp
        or  edx, [esp+244]
        mov  eax,ebp
        and  eax, [esp+248]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+260]
        or  eax,esi
        xor  ebx,eax
        mov  eax,edi
        not  eax
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  ecx, [esp+252]
        mov  [esp+2012],ecx
        mov  ecx, [esp+240]
        not  ebx
        xor  ebx, [esp+256]
        mov  esi,eax
        xor  edx,ebx
        or  edx, [esp+240]
        mov  eax,edx
        xor  eax, [esp+236]
        or  eax,ebx
        mov  [esp+1988],ebx
        mov  ebx, [esp+236]
        or  esi, [esp+236]
        xor  ebp,esi
        mov  esi, [esp+1684]
        mov  [esp+1996],edi
        mov  edi, [esp+1688]
        mov  [esp+2000],ebp
        mov  ebp, [esp+1692]
        xor  edx, [esp+248]
        mov  [esp+2008],edx
        xor  eax, [esp+244]
        mov  [esp+2004],eax
        mov  [esp+1992],ecx
        mov  [esp+1984],ebx
        xor  esi, [esp+1668]
        xor  edi, [esp+1672]
        xor  ebp, [esp+1676]
        mov  [esp+232],ebp
        mov  eax, [esp+1700]
        mov  edx, [esp+1656]
        xor  eax, [esp+1680]
        mov  [esp+228],eax
        mov  eax,ebp
        not  eax
        mov  ebp,eax
        or  edx, [esp+232]
        xor  edx,eax
        mov  eax, [esp+228]
        xor  esi,edx
        xor  eax,edx
        mov  edx, [esp+1664]
        mov  ecx, [esp+1656]
        or  ebp, [esp+1652]
        xor  ebp, [esp+1660]
        mov  [esp+224],esi
        or  eax, [esp+1656]
        mov  [esp+220],eax
        xor  eax, [esp+1652]
        or  eax, [esp+228]
        xor  edi,eax
        mov  [esp+216],edi
        mov  eax,ebp
        or  eax,esi
        mov  esi,ebp
        and  esi,edi
        mov  edi, [esp+1652]
        xor  [esp+220],edx
        xor  esi,eax
        mov  [esp+212],esi
        xor  [esp+212],ecx
        mov  ecx,ebp
        or  ecx, [esp+220]
        xor  ecx, [esp+224]
        mov  ebx,ecx
        or  ebx,esi
        xor  ebx,ebp
        mov  [esp+208],ebx
        xor  ebx,ecx
        mov  ecx, [esp+212]
        xor  [esp+208],edi
        mov  edx,ebp
        or  edx, [esp+216]
        mov  eax,ebp
        and  eax, [esp+220]
        xor  eax,edx
        mov  edi,eax
        not  edi
        xor  edi, [esp+232]
        or  eax,esi
        xor  ebx,eax
        not  ebx
        xor  ebx, [esp+228]
        mov  eax,edi
        not  eax
        mov  esi,eax
        or  esi, [esp+208]
        or  ecx,edi
        xor  ecx,eax
        mov  edx,ecx
        xor  edx,ebx
        or  edx, [esp+212]
        mov  eax,edx
        xor  eax, [esp+208]
        xor  ecx, [esp+224]
        mov  [esp+1884],ecx
        xor  ebp,esi
        mov  [esp+1872],ebp
        mov  ebp, [esp+212]
        or  eax,ebx
        xor  eax, [esp+216]
        mov  [esp+1876],eax
        mov  eax, [esp+208]
        mov  [esp+1856],eax
        mov  eax, [esp+136]
        xor  edx, [esp+220]
        mov  [esp+1880],edx
        mov  [esp+1868],edi
        mov  [esp+1864],ebp
        mov  [esp+1860],ebx
        test  eax,eax
        jne near L247
        mov  edx, [esp+144]
        mov  ecx, [esp+2044]
        mov  dword [esp+204],-1
        mov  dword [esp+200],0
        mov  [esp+112],edx
        mov  [esp+108],ecx
        ;ALIGN 1<<4
	db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00 ; leal   0x0(%esi),%esi
	db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;  leal   0x0(%edi,1),%edi

L457:
        mov  ebx, [esp+1712]
        mov  esi, [ebx+28]
        mov  ebx, [esp+1708]
        mov  edi, [esp+1712]
        mov  edx, [esp+1712]
        xor  esi, [ebx+28]
        mov  [esp+188],esi
        mov  ebp, [edi+24]
        xor  ebp, [ebx+24]
        mov  [esp+184],ebp
        mov  eax, [edi+20]
        xor  eax, [ebx+20]
        mov  [esp+180],eax
        mov  edi, [edi+16]
        mov  ecx, [edx+12]
        xor  edi, [ebx+16]
        xor  ecx, [ebx+12]
        mov  [esp+196],ecx
        mov  ebx, [edx+8]
        mov  esi, [esp+1708]
        mov  ebp, [edx+4]
        not  ecx
        xor  ebx, [esi+8]
        xor  ebp, [esi+4]
        mov  [esp+192],ebp
        mov  eax, [edx]
        xor  eax, [esi]
        mov  ebp,eax
        mov  eax,ecx
        or  eax,ebp
        xor  edi,eax
        mov  eax, [esp+192]
        mov  edx,ebx
        or  edx, [esp+196]
        xor  edx,ecx
        xor  [esp+188],edx
        xor  eax,edx
        or  eax,ebx
        xor  [esp+184],eax
        xor  eax,ebp
        or  eax, [esp+192]
        xor  [esp+180],eax
        mov  eax,edi
        or  eax, [esp+188]
        mov  esi,edi
        and  esi, [esp+180]
        xor  esi,eax
        mov  edx,edi
        or  edx, [esp+180]
        mov  eax,edi
        and  eax, [esp+184]
        xor  eax,edx
        mov  edx, [esp+196]
        mov  [esp+168],eax
        not  dword [esp+168]
        xor  [esp+168],edx
        mov  edx, [esp+168]
        mov  [esp+176],esi
        xor  [esp+176],ebx
        mov  ebx,edi
        or  ebx, [esp+184]
        xor  ebx, [esp+188]
        mov  ecx,ebx
        or  ecx,esi
        xor  ecx,edi
        or  eax,esi
        mov  esi,ecx
        xor  esi,ebx
        xor  esi,eax
        mov  eax, [esp+176]
        mov  ebx, [esp+184]
        mov  [esp+172],ecx
        xor  [esp+172],ebp
        not  esi
        xor  esi, [esp+192]
        not  edx
        mov  ebp,edx
        or  eax, [esp+168]
        xor  eax,edx
        mov  edx,esi
        xor  edx,eax
        or  edx, [esp+176]
        mov  [esp+28],edx
        xor  [esp+28],ebx
        mov  ebx, [esp+172]
        or  ebp, [esp+172]
        xor  ebp,edi
        mov  edi, [esp+1716]
        mov  ecx,eax
        xor  ecx, [esp+188]
        xor  ebx,edx
        or  ebx,esi
        xor  ebx, [esp+180]
        mov  [edi+224],ecx
        mov  eax, [esp+28]
        mov  [edi+192],eax
        mov  [edi+160],ebx
        mov  [edi+128],ebp
        mov  edx, [esp+168]
        mov  [edi+96],edx
        mov  eax, [esp+176]
        mov  [edi+64],eax
        mov  [edi+32],esi
        mov  edx, [esp+172]
        mov  [edi],edx
        mov  edi, [esp+108]
        mov  edx, [esp+144]
        mov  eax, [edi+224]
        mov  edi, [esp+200]
        xor  eax, [edx+edi*4+224]
        xor  eax,ecx
        mov  ecx, [esp+1716]
        xor  eax, [ecx+-288]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  edi, [esp+108]
        mov  eax, [edi+192]
        mov  edi, [esp+112]
        xor  eax, [edi+192]
        xor  eax, [esp+28]
        xor  eax, [ecx+-320]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  edx, [esp+108]
        mov  eax, [edx+160]
        xor  eax, [edi+160]
        xor  eax,ebx
        xor  eax, [ecx+-352]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  eax, [edx+128]
        xor  eax, [edi+128]
        xor  eax,ebp
        xor  eax, [ecx+-384]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  eax, [edx+96]
        xor  eax, [edi+96]
        xor  eax, [esp+168]
        xor  eax, [ecx+-416]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  eax, [edx+64]
        xor  eax, [edi+64]
        xor  eax, [esp+176]
        xor  eax, [ecx+-448]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  eax, [edx+32]
        xor  eax, [edi+32]
        xor  eax,esi
        xor  eax, [ecx+-480]
        not  eax
        and  [esp+204],eax
        je near L467
        mov  eax, [edx]
        xor  eax, [edi]
        xor  eax, [esp+172]
        xor  eax, [ecx+-512]
        not  eax
        and  [esp+204],eax
        je short L467
        add  edi,BYTE 4
        mov  [esp+112],edi
        add  edx,BYTE 4
        mov  [esp+108],edx
        inc  dword [esp+200]
        add  dword [esp+1708],BYTE 32
        add  dword [esp+1712],BYTE 32
        add  ecx,BYTE 4
        mov  [esp+1716],ecx
        cmp  dword [esp+200],BYTE 7
        jle near L457
        mov  edi, [esp+2032]
        mov  esi, [esp+1732]
        add  esi,256
        cld
        mov  ecx,64
        rep movsd
        mov  eax, [esp+204]
        jmp L513
        ;; ALIGN 1<<4 ; IF < 7
	nop
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
L467:
        inc  dword [esp+1648]
        mov  ebx, [esp+1648]
        test  bl,1
        je L476
        mov  edx, [esp+1728]
        cmp  dword [edx],BYTE 0
        je near L49
        ;; ALIGN 1<<4
	db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00 ; leal   0x0(%esi),%esi
	db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ;  leal   0x0(%edi,1),%edi

L480:
        mov  eax, [edx]
        add  edx,BYTE 4
        not  dword [eax]
        cmp  dword [edx],BYTE 0
        jne short L480
        jmp L49
        ;ALIGN 1<<4 ; IF < 7
L476:
        mov  esi, [esp+1648]
        test  esi,2
        je L482
        mov  edi, [esp+1728]
        mov  edx, [esp+1728]
        add  edx,BYTE 116
        cmp  dword [edi+116],BYTE 0
        je near L49
        ;; ALIGN 4
	nop
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
L486:
        mov  eax, [edx]
        add  edx,BYTE 4
        not  dword [eax]
        cmp  dword [edx],BYTE 0
        jne L486
        jmp L49
        ;; ALIGN 1<<4 ; IF < 7
L482:
        mov  ebp, [esp+1648]
        test  ebp,4
        je L488
        mov  eax, [esp+1728]
        mov  edx, [esp+1728]
        add  edx,232
        cmp  dword [eax+232],BYTE 0
        je  near L49
	jmp short L492
        ALIGN 1<<4
	
L492:
        mov  eax, [edx]
        add  edx,BYTE 4
        not  dword [eax]
        cmp  dword [edx],BYTE 0
        jne short L492
        jmp L49
        ;;ALIGN 1<<4 ; IF < 7
L488:
        mov  edx, [esp+1648]
        test  dl,8
        je L494
        mov  ecx, [esp+1728]
        mov  edx, [esp+1728]
        add  edx,348
        cmp  dword [ecx+348],BYTE 0
        je near L49
        ;ALIGN 4
	mov esi,esi
L498:
        mov  eax, [edx]
        add  edx,BYTE 4
        not  dword [eax]
        cmp  dword [edx],BYTE 0
        jne L498
        jmp L49
        ;; ALIGN 1<<4 ; IF < 7
L494:
        mov  ebx, [esp+1648]
        test  bl,16
        je L500
        mov  esi, [esp+1728]
        mov  edx, [esp+1728]
        add  edx,464
        cmp  dword [esi+464],BYTE 0
        je near L49
        ;;ALIGN 4
	mov esi,esi
L504:
        mov  eax, [edx]
        add  edx,BYTE 4
        not  dword [eax]
        cmp  dword [edx],BYTE 0
        jne L504
        jmp L49
        ;;ALIGN 1<<4 ; IF < 7
L500:
        mov  edi, [esp+1648]
        test  edi,32
        je L50
        mov  ebp, [esp+1728]
        mov  edx, [esp+1728]
        add  edx,580
        cmp  dword [ebp+580],BYTE 0
        je near L49
	jmp short L510
        ALIGN 1<<4
L510:
        mov  eax, [edx]
        add  edx,BYTE 4
        not  dword [eax]
        cmp  dword [edx],BYTE 0
        jne L510
        jmp L49
        ;;ALIGN 1<<4 ; IF < 7
L50:
        xor  eax,eax
L513:
        pop  ebx
        pop  esi
        pop  edi
        pop  ebp
        add  esp,2012
        ret

        ALIGN 32
GLOBAL csc_unit_func_6b_i
csc_unit_func_6b_i: 
        push  ebp
        mov  ebp,esp
        sub  esp,BYTE 60
        push  edi
        push  esi
        push  ebx
        mov  eax, [ebp+16]
        mov  [ebp+-20],eax
        test  al,15
        je L521
        add  eax,BYTE 15
        and  al,240
        mov  [ebp+-20],eax
L521:
        mov  edi, [ebp+-20]
        lea  edx, [edi+512]
        mov  [ebp+-24],edx
        mov  edx, [ebp+8]
        lea  ecx, [edi+768]
        mov  [ebp+-28],ecx
        lea  ebx, [edi+1024]
        mov  [ebp+-20],ebx
        add  esp,BYTE -8
        lea  esi, [ebp+-4]
        lea  ebx, [ebp+-8]
        mov  eax, [edx+16]
        mov  [ebp+-8],eax
        mov  eax, [edx+20]
        mov  [ebp+-4],eax
        push  esi
        push  ebx
        call convert_key_from_inc_to_csc
        mov  ecx, [ebp+8]
        mov  eax, [ebp+8]
        mov  edx,1
        add  esp,BYTE 16
        lea  ebx, [ebp+-16]
        mov  ecx, [ecx+4]
        mov  [ebp+-32],ecx
        mov  eax, [eax+12]
        mov  ecx, [ebp+-4]
        mov  [ebp+-52],eax
        xor  eax,eax
        mov  [ebp+-48],ebx
        ;;ALIGN 4
	db 0x8d, 0x76, 0x00 ;       leal   0x0(%esi),%esi
L525:
        test  [ebp+-32],edx
        je L526
        mov  ebx, [ebp+-24]
        mov  dword [ebx+eax*4],-1
        jmp short L527
        ;;ALIGN 1<<4 ; IF < 7
L526:
        mov  ebx, [ebp+-24]
        mov  dword [ebx+eax*4],0
L527:
        test  [ebp+-52],edx
        je L528
        mov  ebx, [ebp+-28]
        mov  dword [ebx+eax*4],-1
        jmp short L529
        ;;ALIGN 1<<4 ; IF < 7
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
L528:
        mov  ebx, [ebp+-28]
        mov  dword [ebx+eax*4],0
L529:
        test  edx,ecx
        je L530
        mov  dword [edi+eax*4],-1
        jmp short L531
        ;;ALIGN 1<<4 ; IF < 7
L530:
        mov  dword [edi+eax*4],0
L531:
        add  edx,edx
        jne L524
        mov  edx, [ebp+8]
        mov  ecx, [ebp+8]
        mov  edx, [edx]
        mov  [ebp+-32],edx
        mov  ecx, [ecx+8]
        mov  [ebp+-52],ecx
        mov  ecx, [ebp+-8]
        mov  edx, 1
L524:
        inc  eax
        cmp  eax,BYTE 63
        jle short L525
        add  esp,BYTE -4
        push  dword 256
        push  BYTE 0
        lea  eax, [edi+256]
        push  eax
        call mymemset
        mov  edx, [ebp+-8]
        mov  eax,edx
        shr  eax,24
        mov  [ebp+-16],al
        mov  eax, [ebp+-16]
        mov  ebx, [ebp+-48]
        mov  ecx,edx
        shr  ecx,8
        and  ecx,65280
        and  eax,-16776961
        sal  edx,8
        and  edx,16711680
        or  eax,ecx
        or  eax,edx
        movzx  edx,byte [ebp+-8]
        sal  edx,24
        and  eax,16777215
        or  eax,edx
        mov  [ebp+-16],eax
        movzx  eax,byte [ebp+-1]
        mov  [ebx+4],al
        movzx  eax,byte [ebp+-2]
        sal  eax,8
        mov  edx, [ebx+4]
        xor  dh,dh
        or  edx,eax
        mov  [ebx+4],edx
        movzx  eax,byte [ebp+-3]
        sal  eax,16
        movzx  edx,dx
        or  edx,eax
        mov  [ebx+4],edx
        mov  eax,[csc_bit_order+24]
        mov  dword [edi+eax*4],-1431655766
        mov  eax,[csc_bit_order+28]
        mov  dword [edi+eax*4],-858993460
        mov  eax,[csc_bit_order+32]
        mov  dword [edi+eax*4],-252645136
        mov  eax,[csc_bit_order+36]
        mov  dword [edi+eax*4],-16711936
        mov  eax,[csc_bit_order+40]
        mov  dword [edi+eax*4],-65536
        mov  eax, [ebp+12]
        add  esp,BYTE 16
        mov  dword [ebp+-36],11
        mov  edx, [eax]
        cmp  edx,2048
        jbe L535
        mov  ebx,1
        ;ALIGN 4
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
	db 0x8d, 0xbc, 0x27, 0x00, 0x00, 0x00, 0x00 ; leal   0x0(%edi,1),%edi
L536:
        inc  dword [ebp+-36]
        mov  ecx, [ebp+-36]
        mov  eax,ebx
        sal  eax,cl
        cmp  edx,eax
        ja L536
L535:
        mov  eax, [ebp+-36]
        mov  esi, [ebp+-48]
        xor  ebx,ebx
        add  eax,BYTE -11
        mov  [ebp+-44],eax
        ;ALIGN 4
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
L541:
        mov  ecx, [csc_bit_order+ebx*4]
        mov  dword [edi+ecx*4],0
        mov  eax,ecx
        test  ecx,ecx
        jge L542
        lea  eax, [ecx+7]
L542:
        sar  eax,3
        mov  edx,7
        sub  edx,eax
        sal  eax,3
        sub  ecx,eax
        mov  eax,1
        sal  eax,cl
        not  al
        and  [edx+esi], al
        inc  ebx
        cmp  ebx,BYTE 5
        jle L541
        xor  esi,esi
        cmp  esi, [ebp+-44]
        jae L546
        mov  ebx, [ebp+-48]
        mov  edx, [ebp+-44]
        mov  dword [ebp-52], csc_bit_order+44
        mov  [ebp+-40],edx
        ALIGN 4
L548:
        mov  eax, [ebp+-52]
        mov  ecx, [eax]
        mov  dword [edi+ecx*4],0
        mov  eax,ecx
        test  ecx,ecx
        jge L549
        lea  eax, [ecx+7]
L549:
        sar  eax,3
        mov  edx,7
        sub  edx,eax
        sal  eax,3
        sub  ecx,eax
        mov  eax,1
        sal  eax,cl
        not  al
        and  [edx+ebx],al
        add  dword [ebp+-52],BYTE 4
        inc  esi
        cmp  esi, [ebp+-40]
        jb L548
L546:
        mov  edx, [ebp+-48]
        xor  esi,esi
        mov  [ebp+-52],edx
        jmp short L552
        ;ALIGN 4
	db 0x8d, 0xb6, 0x00, 0x00, 0x00, 0x00 ;  leal   0x0(%esi),%esi
	db 0x8d, 0xbf, 0x00, 0x00, 0x00, 0x00 ;  leal   0x0(%edi),%edi
L556:
        xor  ebx,ebx
        test  esi,1
        jne L558
        mov  edx,1
        ALIGN 4
L559:
        inc  ebx
        mov  eax,edx
        mov  ecx,ebx
        sal  eax,cl
        test  eax,esi
        je L559
L558:
        mov  ebx, [csc_bit_order+ebx*4+44]
        mov  ecx,ebx
        shr  ecx,3
        mov  edx,7
        sub  edx,ecx
        not  dword [edi+ebx*4]
        and  ebx,BYTE 7
        mov  ecx,ebx
        mov  ebx, [ebp+-52]
        mov  eax,1
        sal  eax,cl
        xor  [edx+ebx],al
L552:
        mov  eax, [ebp+-20]
        add  esp,BYTE -12
        push  eax
        mov  edx, [ebp+-28]
        push  edx
        mov  ecx, [ebp+-24]
        push  ecx
        mov  ebx, [ebp+-48]
        push  ebx
        push  edi
        call cscipher_bitslicer_6b_i
        add  esp,BYTE 32
        test  eax,eax
        jne L581
        mov  ecx, [ebp+-44]
        inc  esi
        mov  eax,esi
        shr  eax,cl
        test  eax,eax
        je L556
        jmp L562
        ;;ALIGN 1<<4 ; IF < 7
L581:
        xor  ebx,ebx
        cmp  eax,BYTE 1
        je L564
        ;ALIGN 4
	nop
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
L565:
        inc  ebx
        shr  eax,BYTE 1
        cmp  eax,BYTE 1
        jne L565
L564:
        mov  dword [ebp+-8],0
        mov  dword [ebp+-4],0
        mov  edx,8
        ;ALIGN 4
	db 0x8d, 0x74, 0x26, 0x00 ;    leal   0x0(%esi,1),%esi
L571:
        cmp  edx,BYTE 31
        jg L572
        mov  eax, [edi+edx*4]
        mov  ecx,ebx
        shr  eax,cl
        and  eax,BYTE 1
        mov  ecx,edx
        sal  eax,cl
        or  [ebp+-4],eax
        jmp short L570
        ;;ALIGN 1<<4 ; IF < 7
L572:
        mov  eax, [edi+edx*4]
        mov  ecx,ebx
        shr  eax,cl
        and  eax,BYTE 1
        lea  ecx, [edx+-32]
        sal  eax,cl
        or  [ebp+-8],eax
L570:
        inc  edx
        cmp  edx,BYTE 63
        jle L571
        add  esp,BYTE -8
        lea  ebx, [ebp+-4]
        push  ebx
        lea  eax, [ebp+-8]
        push  eax
        call convert_key_from_csc_to_inc
        mov  ecx, [ebp+8]
        mov  edx, [ebp+-4]
        add  esp,BYTE 16
        mov  eax, [ecx+20]
        cmp  edx,eax
        jae L575
        mov  ebx, [ebp+12]
        sub  eax,edx
        mov  [ebx],eax
        jmp short L576
        ;;ALIGN 1<<4 ; IF < 7
	db 0x8d, 0xb4, 0x26, 0x00, 0x00, 0x00, 0x00 ;  leal   0x0(%esi,1),%esi
L575:
        mov  ecx, [ebp+12]
        sub  edx,eax
        mov  [ecx],edx
L576:
        mov  eax, [ebp+-4]
        mov  ebx, [ebp+8]
        mov  [ebx+20],eax
        mov  eax, [ebp+-8]
        mov  [ebx+16],eax
        mov  eax,2
        jmp short L580
        ;;ALIGN 1<<4 ; IF < 7
	db 0x8d, 0x76, 0x00 ;        leal   0x0(%esi),%esi
L562:
        mov  ecx, [ebp+-36]
        mov  ebx, [ebp+12]
        mov  eax,1
        sal  eax,cl
        mov  [ebx],eax
        mov  edx, [ebp+8]
        add  eax, [edx+20]
        mov  [edx+20],eax
        shr  eax,cl
        test  eax,eax
        jne L578
        inc  dword [edx+16]
L578:
        mov  eax,1
L580:
        lea  esp, [ebp+-72]
        pop  ebx
        pop  esi
        pop  edi
        mov  esp,ebp
        pop  ebp
        ret

        ALIGN 32
mymemset:
	push  edi
	push  ebx
	mov   edi, [esp+0xc]
	movzx eax, BYTE [esp+0x10] ;0f b6 44 24 10
	mov   ecx, [esp+0x14]
	push  edi
	cld    
	cmp   ecx, 15
	jle   short .here
	mov   ah,al
	mov   edx,eax
	shl   eax,16
	or    eax,edx
	mov   edx,edi
	neg   edx
	and   edx,3
	mov   ebx,ecx
	sub   ebx,edx
	mov   ecx,edx
	repz  stosb
	mov   ecx,ebx
	shr   ecx,2
	repz  stosd
	mov   ecx,ebx
	and   ecx,3
.here:
	repz stosb
	pop   eax
	pop   ebx
	pop   edi
	ret    
