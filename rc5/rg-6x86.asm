.386p

;.model flat    ; some use small model, so use appropriate cmdline switch
                ; eg /mf or /ms for WASM, 

_TEXT   segment dword public use32 'CODE'
align 4


public _rc5_unit_func_6x86
_rc5_unit_func_6x86 proc near

sub esp,268
push ebp
push edi
push esi
push ebx
mov ebp,[ 292+esp]
mov [ 268+esp],ebp
mov [ 224+esp],dword ptr 0
mov eax,[ 288+esp]
 	;APP
mov [ 244+esp],ebp
mov ebx,[ 20+eax]	; ebx = l0 = Llo1
mov edx,[ 16+eax]	; edx = l1 = Lhi1
mov esi,ebx	; esi = l2 = Llo2
lea edi,[ 16777216+edx]	; edi = l3 = lhi2
mov [ 264+esp],ebx
mov [ 260+esp],edx
mov ebp,[ 4+eax]
mov [ 228+esp],ebp
mov ebp,[ 0+eax]
mov [ 232+esp],ebp
mov ebp,[ 12+eax]
mov [ 236+esp],ebp
mov ebp,[ 8+eax]
mov [ 240+esp],ebp
add ebx,-1089828067
rol ebx,29
mov [ 272+esp],ebx

lea eax,[ 354637369+ebx]
rol eax,3
mov [ 276+esp],eax

lea ecx,[eax+ebx]
mov [ 280+esp],ecx

_loaded_6x86:
mov ebx,[ 272+esp]	; 1
mov eax,[ 276+esp]
mov esi,ebx	; 1
mov ebp,eax

mov ecx,[ 280+esp]	; 1
add edx,ecx
rol edx,cl	; 2
add edi,ecx
mov [ 20+esp],eax	; 1
add ebp,-196066091
rol edi,cl	; 2 sum = 8
add eax,-196066091	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 24+esp],eax	; 1
add ebx,ecx
add eax,-1836597618	; 1
mov [ 128+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-1836597618
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 28+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 132+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,817838151
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,817838151	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 32+esp],eax	; 1
add ebx,ecx
add eax,-822693376	; 1
mov [ 136+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-822693376
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 36+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 140+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,1831742393
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,1831742393	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 40+esp],eax	; 1
add ebx,ecx
add eax,191210866	; 1
mov [ 144+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,191210866
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 44+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 148+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,-1449320661
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,-1449320661	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 48+esp],eax	; 1
add ebx,ecx
add eax,1205115108	; 1
mov [ 152+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,1205115108
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 52+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 156+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,-435416419
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,-435416419	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 56+esp],eax	; 1
add ebx,ecx
add eax,-2075947946	; 1
mov [ 160+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-2075947946
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 60+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 164+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,578487823
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,578487823	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 64+esp],eax	; 1
add ebx,ecx
add eax,-1062043704	; 1
mov [ 168+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-1062043704
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 68+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 172+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,1592392065
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,1592392065	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 72+esp],eax	; 1
add ebx,ecx
add eax,-48139462	; 1
mov [ 176+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-48139462
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 76+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 180+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,-1688670989
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,-1688670989	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 80+esp],eax	; 1
add ebx,ecx
add eax,965764780	; 1
mov [ 184+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,965764780
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 84+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 188+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,-674766747
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,-674766747	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 88+esp],eax	; 1
add ebx,ecx
add eax,1979669022	; 1
mov [ 192+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,1979669022
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 92+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 196+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,339137495
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,339137495	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 96+esp],eax	; 1
add ebx,ecx
add eax,-1301394032	; 1
mov [ 200+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-1301394032
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 100+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 204+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,1353041737
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,1353041737	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 104+esp],eax	; 1
add ebx,ecx
add eax,-287489790	; 1
mov [ 208+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,-287489790
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 108+esp],eax
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 212+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,-1928021317
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,-1928021317	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 112+esp],eax	; 1
add ebx,ecx
add eax,726414452	; 1
mov [ 216+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,726414452
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 116+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 220+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add eax,-1089828067
rol edi,cl	; 2 sum = 19 (r1 & r2)
_end_round1_6x86:
 	; addl $0xbf0a8b1d,%eax # . already done in ROUND_1_LAST
add ebp,-1089828067
add eax,edx	; 1
add ebp,edi
rol eax,3	; 1
rol ebp,3
mov ecx,eax	; 1
add ecx,edx
mov [ 16+esp],eax	; 1
add ebx,ecx
add eax,[ 20+esp]	; 1
mov [ 120+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 20+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 20+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 124+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 128+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 24+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 24+esp],eax	; 1
add ebx,ecx
add eax,[ 28+esp]	; 1
mov [ 128+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 132+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 28+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 132+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 136+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 32+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 32+esp],eax	; 1
add ebx,ecx
add eax,[ 36+esp]	; 1
mov [ 136+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 140+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 36+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 140+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 144+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 40+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 40+esp],eax	; 1
add ebx,ecx
add eax,[ 44+esp]	; 1
mov [ 144+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 148+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 44+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 148+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 152+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 48+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 48+esp],eax	; 1
add ebx,ecx
add eax,[ 52+esp]	; 1
mov [ 152+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 156+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 52+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 156+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 160+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 56+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 56+esp],eax	; 1
add ebx,ecx
add eax,[ 60+esp]	; 1
mov [ 160+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 164+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 60+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 164+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 168+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 64+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 64+esp],eax	; 1
add ebx,ecx
add eax,[ 68+esp]	; 1
mov [ 168+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 172+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 68+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 172+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 176+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 72+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 72+esp],eax	; 1
add ebx,ecx
add eax,[ 76+esp]	; 1
mov [ 176+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 180+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 76+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 180+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 184+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 80+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 80+esp],eax	; 1
add ebx,ecx
add eax,[ 84+esp]	; 1
mov [ 184+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 188+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 84+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 188+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 192+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 88+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 88+esp],eax	; 1
add ebx,ecx
add eax,[ 92+esp]	; 1
mov [ 192+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 196+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 92+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 196+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 200+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 96+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 96+esp],eax	; 1
add ebx,ecx
add eax,[ 100+esp]	; 1
mov [ 200+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 204+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 100+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 204+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 208+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 104+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 104+esp],eax	; 1
add ebx,ecx
add eax,[ 108+esp]	; 1
mov [ 208+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 212+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 108+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 212+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
add ebp,[ 216+esp]
rol edi,cl	; 2 sum = 19 (r1 & r2)
add eax,[ 112+esp]	; . pairs with roll in previous iteration
add ebp,edi	; 1
add eax,edx
rol ebp,3	; 1
rol eax,3
mov ecx,eax	; 1
add ecx,edx	;   yes, it works
mov [ 112+esp],eax	; 1
add ebx,ecx
add eax,[ 116+esp]	; 1
mov [ 216+esp],ebp
rol ebx,cl	; 2
lea ecx,[ebp+edi]
add esi,ecx	; 1
add ebp,[ 220+esp]
rol esi,cl	; 2
add eax,ebx	; . pairs with roll in previous iteration
add ebp,esi	; 1
rol eax,3
rol ebp,3	; 1
mov [ 116+esp],eax	;   yes, it works !
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
mov [ 220+esp],ebp
rol edx,cl	; 2
lea ecx,[ebp+esi]
add edi,ecx	; 1
mov [ 248+esp],ebp
rol edi,cl	; 2 sum = 19 (r1 & r2)
_end_round2_6x86:
 	; movl %ebp, 232+16(%esp) already done in ROUND_2_LAST
mov [ 256+esp],esi
mov [ 252+esp],edi
lea ebp,[ 16+esp]
add eax,[ 0*4+ebp]	; (pairs with leal)
add eax,edx	; 1
mov esi,[ 228+esp]
rol eax,3	; 1
mov ecx,edx
add esi,eax	; 1
add ecx,eax
add ebx,ecx	; 1
add eax,[ 1*4+ebp]
rol ebx,cl	; 2
add eax,ebx
mov ecx,ebx	; 1
rol eax,3
mov edi,[ 232+esp]	; 1
add edi,eax
add ecx,eax	; 1
add edx,ecx	; 1
rol edx,cl	; 2
add eax,[ 2*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 3*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 4*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 5*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 6*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 7*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 8*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 9*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 10*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 11*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 12*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 13*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 14*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 15*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 16*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 17*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 18*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 19*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 20*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 21*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 22*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 23*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
_end_round3_1_6x86:
add eax,[ 24*4+ebp]	;     A = ROTL3(S24 + A + L1);
mov ecx,edi	; 1  eA = ROTL(eA ^ eB, eB) + A;
add eax,edx
xor esi,edi	; 1
rol eax,3
rol esi,cl	; 2
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_1_6x86

mov ecx,eax	; 1 L0 = ROTL(L0 + A + L1, A + L1);
add ecx,edx	; A = ROTL3(S25 + A + L0);
xor edi,esi	; 1 eB = ROTL(eB ^ eA, eA) + A;
add ebx,ecx
rol ebx,cl	; 2
add eax,[ 25*4+ebp]
mov ecx,esi	; 1
add eax,ebx
rol edi,cl	; 2
rol eax,3
add edi,eax	; 1

cmp edi,[ 240+esp]
je _full_exit_6x86

__exit_1_6x86:
mov edx,[ 252+esp]
mov ebx,[ 256+esp]
mov eax,[ 248+esp]
lea ebp,[ 120+esp]
add eax,edx	; 1
mov ecx,edx
add eax,[ 0*4+ebp]	; 1
rol eax,3	; 1
mov esi,[ 228+esp]
add esi,eax	; 1
add ecx,eax
add ebx,ecx	; 1
add eax,[ 1*4+ebp]
rol ebx,cl	; 2
add eax,ebx
mov ecx,ebx	; 1
rol eax,3
mov edi,[ 232+esp]	; 1
add edi,eax
add ecx,eax	; 1
add edx,ecx	; 1
rol edx,cl	; 2
add eax,[ 2*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 3*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 4*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 5*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 6*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 7*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 8*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 9*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 10*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 11*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 12*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 13*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 14*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 15*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 16*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 17*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 18*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 19*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 20*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 21*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
add eax,[ 22*4+ebp]
xor esi,edi
mov ecx,edi	; 1
add eax,edx
rol esi,cl	; 2
rol eax,3
mov ecx,eax
add ecx,edx	; 1
add esi,eax
add ebx,ecx	; 1
add eax,[ 23*4+ebp]
rol ebx,cl	; 2

xor edi,esi
mov ecx,esi
rol edi,cl	; 2
add eax,ebx
rol eax,3
mov ecx,eax	; 1
add ecx,ebx
add edx,ecx	; 1
add edi,eax
rol edx,cl	; 2 sum = 13
_end_round3_2_6x86:
add eax,[ 24*4+ebp]	;    A = ROTL3(S24 + A + L1);
mov ecx,edi	; 1   eA = ROTL(eA ^ eB, eB) + A;
add eax,edx
xor esi,edi	; 1
rol eax,3
rol esi,cl	; 2
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_2_6x86

mov ecx,eax	; 1 L0 = ROTL(L0 + A + L1, A + L1);
add ecx,edx	; A = ROTL3(S25 + A + L0);
xor edi,esi	; 1 eB = ROTL(eB ^ eA, eA) + A;
add ebx,ecx
rol ebx,cl	; 2
add eax,[ 25*4+ebp]
mov ecx,esi	; 1
add eax,ebx
rol edi,cl	; 2
rol eax,3
add edi,eax	; 1

cmp edi,[ 240+esp]
jne __exit_2_6x86
mov [ 224+esp],dword ptr 1
jmp _full_exit_6x86

__exit_2_6x86:

mov edx,[ 260+esp]


add edx,33554432
jc _next_inc_6x86

_next_iter_6x86:
mov [ 260+esp],edx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_6x86
mov eax,[ 288+esp]	; pointer to rc5unitwork
mov ebx,[ 264+esp]
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_6x86

_next_inc_6x86:
add edx,65536
test edx,16711680
jnz _next_iter_6x86

sub edx,16777216
add edx,256
test edx,-256
jnz _next_iter_6x86

sub edx,65536
add edx,1
test edx,255
jnz _next_iter_6x86

 	; we should never go here, it would mean we have iterated 2^32 times ...
 	; stop the client, something went wrong
;mov 0,0	; generate a segfault

_full_exit_6x86:
mov ebp,[ 244+esp]

 	;NO_APP
mov edx,ebp
sub edx,[ 268+esp]
mov eax,[ 224+esp]
lea edx,[ 2*edx+eax]
mov eax,edx
pop ebx
pop esi
pop edi
pop ebp
add esp,268
ret
_rc5_unit_func_6x86 endp
_TEXT   ends
end

