.386p

;.model flat    ; some use small model, so use appropriate cmdline switch
                ; eg /mf or /ms for WASM, 

_TEXT   segment dword public use32 'CODE'
align 4

public _rc5_unit_func_k6
_rc5_unit_func_k6 proc near

sub esp,268
push ebp
push edi
push esi
push ebx
mov ebp,[ 292+esp]
mov [ 268+esp],ebp
mov [ 16+esp],dword ptr 0
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
_bigger_loop_k6:
add ebx,-1089828067	; 1
rol ebx,29	; 3
mov [ 272+esp],ebx	; 1

lea eax,[ 354637369+ebx]	; 1
rol eax,3	; 2
mov [ 276+esp],eax	; 1

lea ecx,[eax+ebx]	; 2
mov [ 280+esp],ecx	; 1

;.balign 4
_loaded_k6:
mov ebx,[ 272+esp]
mov esi,ebx
mov eax,[ 276+esp]
mov ebp,eax

mov ecx,[ 280+esp]
add edx,ecx
add eax,-196066091
add edi,ecx
add ebp,-196066091
rol edx,cl
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 28+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 132+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-1836597618	;   alu
add esi,ecx	; 1 alu
add ebp,-1836597618	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 32+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 136+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,817838151	;   alu
add edi,ecx	; 1 alu
add ebp,817838151	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 36+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 140+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-822693376	;   alu
add esi,ecx	; 1 alu
add ebp,-822693376	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 40+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 144+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,1831742393	;   alu
add edi,ecx	; 1 alu
add ebp,1831742393	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 44+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 148+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,191210866	;   alu
add esi,ecx	; 1 alu
add ebp,191210866	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 48+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 152+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,-1449320661	;   alu
add edi,ecx	; 1 alu
add ebp,-1449320661	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 52+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 156+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,1205115108	;   alu
add esi,ecx	; 1 alu
add ebp,1205115108	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 56+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 160+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,-435416419	;   alu
add edi,ecx	; 1 alu
add ebp,-435416419	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 60+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 164+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-2075947946	;   alu
add esi,ecx	; 1 alu
add ebp,-2075947946	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 64+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 168+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,578487823	;   alu
add edi,ecx	; 1 alu
add ebp,578487823	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 68+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 172+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-1062043704	;   alu
add esi,ecx	; 1 alu
add ebp,-1062043704	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 72+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 176+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,1592392065	;   alu
add edi,ecx	; 1 alu
add ebp,1592392065	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 76+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 180+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-48139462	;   alu
add esi,ecx	; 1 alu
add ebp,-48139462	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 80+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 184+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,-1688670989	;   alu
add edi,ecx	; 1 alu
add ebp,-1688670989	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 84+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 188+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,965764780	;   alu
add esi,ecx	; 1 alu
add ebp,965764780	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 88+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 192+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,-674766747	;   alu
add edi,ecx	; 1 alu
add ebp,-674766747	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 92+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 196+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,1979669022	;   alu
add esi,ecx	; 1 alu
add ebp,1979669022	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 96+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 200+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,339137495	;   alu
add edi,ecx	; 1 alu
add ebp,339137495	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 100+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 204+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-1301394032	;   alu
add esi,ecx	; 1 alu
add ebp,-1301394032	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 104+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 208+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,1353041737	;   alu
add edi,ecx	; 1 alu
add ebp,1353041737	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 108+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 212+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,-287489790	;   alu
add esi,ecx	; 1 alu
add ebp,-287489790	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 112+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 216+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,-1928021317	;   alu
add edi,ecx	; 1 alu
add ebp,-1928021317	;   alu
rol edi,cl	; 2 ? sum = 12
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
mov [ 116+esp],eax	; 1 st
lea ecx,[eax+edx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 220+esp],ebp	; 1 st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	; 1 st
add eax,726414452	;   alu
add esi,ecx	; 1 alu
add ebp,726414452	;   alu
rol esi,cl	; 2 ? sum = 12
add eax,ebx	; 1 alu
add ebp,esi	;   alu
rol eax,3	; 2 ?
mov [ 120+esp],eax	; 1 st
lea ecx,[eax+ebx]	;   st (will 'pair' with roll)
rol ebp,3	; 2 ?
mov [ 224+esp],ebp	; 1 st
add edx,ecx	;   alu
rol edx,cl	; 2 ?
lea ecx,[ebp+esi]	; 1 alu
add eax,-1089828067	;   alu
add edi,ecx	; 1 alu
add ebp,-1089828067	;   alu
rol edi,cl	; 2 ? sum = 12
_end_round1_k6:
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 20+esp],eax
mov [ 124+esp],ebp

lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
rol esi,cl

mov ecx,[ 276+esp]
add eax,ebx
add eax,ecx
add ebp,esi
add ebp,ecx
rol eax,3
rol ebp,3
mov [ 24+esp],eax
mov [ 128+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 28+esp]
add edi,ecx
add ebp,[ 132+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 28+esp],eax	;   st
mov [ 132+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 32+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 136+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 32+esp],eax
mov [ 136+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 36+esp]
add edi,ecx
add ebp,[ 140+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 36+esp],eax	;   st
mov [ 140+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 40+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 144+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 40+esp],eax
mov [ 144+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 44+esp]
add edi,ecx
add ebp,[ 148+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 44+esp],eax	;   st
mov [ 148+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 48+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 152+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 48+esp],eax
mov [ 152+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 52+esp]
add edi,ecx
add ebp,[ 156+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 52+esp],eax	;   st
mov [ 156+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 56+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 160+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 56+esp],eax
mov [ 160+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 60+esp]
add edi,ecx
add ebp,[ 164+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 60+esp],eax	;   st
mov [ 164+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 64+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 168+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 64+esp],eax
mov [ 168+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 68+esp]
add edi,ecx
add ebp,[ 172+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 68+esp],eax	;   st
mov [ 172+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 72+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 176+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 72+esp],eax
mov [ 176+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 76+esp]
add edi,ecx
add ebp,[ 180+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 76+esp],eax	;   st
mov [ 180+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 80+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 184+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 80+esp],eax
mov [ 184+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 84+esp]
add edi,ecx
add ebp,[ 188+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 84+esp],eax	;   st
mov [ 188+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 88+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 192+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 88+esp],eax
mov [ 192+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 92+esp]
add edi,ecx
add ebp,[ 196+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 92+esp],eax	;   st
mov [ 196+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 96+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 200+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 96+esp],eax
mov [ 200+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 100+esp]
add edi,ecx
add ebp,[ 204+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 100+esp],eax	;   st
mov [ 204+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 104+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 208+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 104+esp],eax
mov [ 208+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 108+esp]
add edi,ecx
add ebp,[ 212+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 108+esp],eax	;   st
mov [ 212+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 112+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 216+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 112+esp],eax
mov [ 216+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 116+esp]
add edi,ecx
add ebp,[ 220+esp]
rol edi,cl
add eax,edx	; 1 alu
add ebp,edi	;   alu
rol eax,3	; 2 ?
rol ebp,3	; 2 ?
mov [ 116+esp],eax	;   st
mov [ 220+esp],ebp	;   st
lea ecx,[eax+edx]	;   st
add ebx,ecx	;   alu
rol ebx,cl	; 2 ?
lea ecx,[ebp+edi]	;   st
add eax,[ 120+esp]	;   ld  alu
add esi,ecx	;   alu
add ebp,[ 224+esp]	;   ld  alu
rol esi,cl	; 2 ?
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 120+esp],eax
mov [ 224+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
mov [ 248+esp],ebp
add edi,ecx
mov [ 256+esp],esi
rol edi,cl
_end_round2_k6:
 	;movl %ebp,232+16(%esp) # already in ROUND_2_LAST
 	;movl %esi,240+16(%esp) # already in ROUND_2_LAST
mov [ 252+esp],edi
add eax,edx	; 1 A = ROTL3(S00 + A + L1);
add eax,[ 20+esp]	; 2
rol eax,3	; 2
mov esi,[ 228+esp]	; 1 eA = P_0 + A;
add esi,eax	; 1
lea ecx,[eax+edx]	; 2 L0 = ROTL(L0 + A + L1, A + L1);
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1 A = ROTL3(S01 + A + L0);
add eax,[ 24+esp]	; 2
rol eax,3	; 2
mov edi,[ 232+esp]	; 1 eB = P_1 + A;
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2 L1 = ROTL(L1 + A + L0, A + L0);
add edx,ecx	; 1
rol edx,cl	; 3 sum = 26
_round3_k6_S1_2:
add eax,edx
add eax,[ 28+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 32+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_4:
add eax,edx
add eax,[ 36+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 40+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_6:
add eax,edx
add eax,[ 44+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 48+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_8:
add eax,edx
add eax,[ 52+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 56+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_10:
add eax,edx
add eax,[ 60+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 64+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_12:
add eax,edx
add eax,[ 68+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 72+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_14:
add eax,edx
add eax,[ 76+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 80+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_16:
add eax,edx
add eax,[ 84+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 88+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_18:
add eax,edx
add eax,[ 92+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 96+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_20:
add eax,edx
add eax,[ 100+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 104+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S1_22:
add eax,edx
add eax,[ 108+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 112+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_end_round3_1_k6:
add eax,edx	; 1 A = ROTL3(S24 + A + L1);
add eax,[ 116+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1 eA = ROTL(eA ^ eB, eB) + A
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_1_k6

lea ecx,[eax+edx]	; 2 L0 = ROTL(L0 + A + L1, A + L1);
add ebx,ecx	; 1
rol ebx,cl	; 3
add eax,ebx	; 1 A = ROTL3(S25 + A + L0);
add eax,[ 120+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1 eB = ROTL(eB ^ eA, eA) + A
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1

cmp edi,[ 240+esp]
je _full_exit_k6

;.balign 4
__exit_1_k6:
mov edx,[ 252+esp]
mov ebx,[ 256+esp]
mov eax,[ 248+esp]
add eax,edx	; 1 A = ROTL3(S00 + A + L1);
add eax,[ 124+esp]	; 2
rol eax,3	; 2
mov esi,[ 228+esp]	; 1 eA = P_0 + A;
add esi,eax	; 1
lea ecx,[eax+edx]	; 2 L0 = ROTL(L0 + A + L1, A + L1);
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1 A = ROTL3(S01 + A + L0);
add eax,[ 128+esp]	; 2
rol eax,3	; 2
mov edi,[ 232+esp]	; 1 eB = P_1 + A;
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2 L1 = ROTL(L1 + A + L0, A + L0);
add edx,ecx	; 1
rol edx,cl	; 3 sum = 26
_round3_k6_S2_2:
add eax,edx
add eax,[ 132+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 136+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_4:
add eax,edx
add eax,[ 140+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 144+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_6:
add eax,edx
add eax,[ 148+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 152+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_8:
add eax,edx
add eax,[ 156+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 160+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_10:
add eax,edx
add eax,[ 164+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 168+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_12:
add eax,edx
add eax,[ 172+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 176+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_14:
add eax,edx
add eax,[ 180+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 184+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_16:
add eax,edx
add eax,[ 188+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 192+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_18:
add eax,edx
add eax,[ 196+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 200+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_20:
add eax,edx
add eax,[ 204+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 208+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_round3_k6_S2_22:
add eax,edx
add eax,[ 212+esp]
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl

add eax,ebx
add eax,[ 216+esp]
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
_end_round3_2_k6:
add eax,edx	; 1 A = ROTL3(S24 + A + L1);
add eax,[ 220+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1 eA = ROTL(eA ^ eB, eB) + A
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_2_k6

lea ecx,[eax+edx]	; 2 L0 = ROTL(L0 + A + L1, A + L1);
add ebx,ecx	; 1
rol ebx,cl	; 3
add eax,ebx	; 1 A = ROTL3(S25 + A + L0);
add eax,[ 224+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1 eB = ROTL(eB ^ eA, eA) + A
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1

cmp edi,[ 240+esp]
jne __exit_2_k6
mov [ 16+esp], dword ptr 1
jmp _full_exit_k6

;.balign 4
__exit_2_k6:

mov edx,[ 260+esp]


add edx,33554432
jc _next_inc_k6

_next_iter_k6:
mov [ 260+esp],edx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_k6
mov eax,[ 288+esp]	; pointer to rc5unitwork
mov ebx,[ 264+esp]
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_k6

;.balign 4
_next_iter2_k6:
mov [ 264+esp],ebx
mov [ 260+esp],edx
lea edi,[ 16777216+edx]
mov esi,ebx
sub [ 268+esp], dword ptr 1
jg _bigger_loop_k6
mov eax,[ 288+esp]	; pointer to rc5unitwork
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_k6

;.balign 4
_next_inc_k6:
add edx,65536
test edx,16711680
jnz _next_iter_k6

add edx,-16776960
test edx,-256
jnz _next_iter_k6

add edx,-65535
test edx,255
jnz _next_iter_k6


mov ebx,[ 264+esp]

sub edx,256
add ebx,16777216
jnc _next_iter2_k6

add ebx,65536
test ebx,16711680
jnz _next_iter2_k6

add ebx,-16776960
test ebx,-256
jnz _next_iter2_k6

add ebx,-65535
test ebx,255
jnz _next_iter2_k6

 	; Moo !
 	; We have just finished checking the last key
 	; of the rc5-64 keyspace...
 	; Not much to do here, since we have finished the block ...


;.balign 4
_full_exit_k6:
mov ebp,[ 244+esp]

 	;NO_APP
mov edx,ebp
sub edx,[ 268+esp]
mov eax,[ 16+esp]
lea edx,[ 2*edx+eax]
mov eax,edx
pop ebx
pop esi
pop edi
pop ebp
add esp,268
ret
_rc5_unit_func_k6 endp
_TEXT   ends
end

