.386p

if modelnum eq 1
  .model small
elseif modelnum eq 2
  .model flat
endif

_TEXT   segment dword public use32 'CODE'
align 4
public _rc5_unit_func_p5
_rc5_unit_func_p5 proc near

sub esp,256
push ebp
push edi
push esi
push ebx
mov ebp,[ 280+esp]
mov [ 268+esp],ebp
mov [ 16+esp],dword ptr 0
mov eax,[ 276+esp]
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

_loaded_p5:
mov eax,-1089828067	; 1
mov ebp,eax

mov ecx,eax	; 1
add ebx,eax
rol ebx,cl	; 3

add esi,ebp	; 1
add ebp,1444465436
rol esi,cl	; 3
lea eax,[ 1444465436+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 24+esp],eax
mov [ 128+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-196066091
rol edi,cl	; 3 sum = 15
lea eax,[-196066091+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 28+esp],eax
mov [ 132+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-1836597618
rol esi,cl	; 3 sum = 15
lea eax,[-1836597618+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 32+esp],eax
mov [ 136+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,817838151
rol edi,cl	; 3 sum = 15
lea eax,[ 817838151+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 36+esp],eax
mov [ 140+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-822693376
rol esi,cl	; 3 sum = 15
lea eax,[-822693376+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 40+esp],eax
mov [ 144+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,1831742393
rol edi,cl	; 3 sum = 15
lea eax,[ 1831742393+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 44+esp],eax
mov [ 148+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,191210866
rol esi,cl	; 3 sum = 15
lea eax,[ 191210866+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 48+esp],eax
mov [ 152+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-1449320661
rol edi,cl	; 3 sum = 15
lea eax,[-1449320661+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 52+esp],eax
mov [ 156+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,1205115108
rol esi,cl	; 3 sum = 15
lea eax,[ 1205115108+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 56+esp],eax
mov [ 160+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-435416419
rol edi,cl	; 3 sum = 15
lea eax,[-435416419+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 60+esp],eax
mov [ 164+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-2075947946
rol esi,cl	; 3 sum = 15
lea eax,[-2075947946+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 64+esp],eax
mov [ 168+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,578487823
rol edi,cl	; 3 sum = 15
lea eax,[ 578487823+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 68+esp],eax
mov [ 172+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-1062043704
rol esi,cl	; 3 sum = 15
lea eax,[-1062043704+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 72+esp],eax
mov [ 176+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,1592392065
rol edi,cl	; 3 sum = 15
lea eax,[ 1592392065+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 76+esp],eax
mov [ 180+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-48139462
rol esi,cl	; 3 sum = 15
lea eax,[-48139462+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 80+esp],eax
mov [ 184+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-1688670989
rol edi,cl	; 3 sum = 15
lea eax,[-1688670989+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 84+esp],eax
mov [ 188+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,965764780
rol esi,cl	; 3 sum = 15
lea eax,[ 965764780+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 88+esp],eax
mov [ 192+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-674766747
rol edi,cl	; 3 sum = 15
lea eax,[-674766747+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 92+esp],eax
mov [ 196+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,1979669022
rol esi,cl	; 3 sum = 15
lea eax,[ 1979669022+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 96+esp],eax
mov [ 200+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,339137495
rol edi,cl	; 3 sum = 15
lea eax,[ 339137495+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 100+esp],eax
mov [ 204+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-1301394032
rol esi,cl	; 3 sum = 15
lea eax,[-1301394032+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 104+esp],eax
mov [ 208+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,1353041737
rol edi,cl	; 3 sum = 15
lea eax,[ 1353041737+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 108+esp],eax
mov [ 212+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,-287489790
rol esi,cl	; 3 sum = 15
lea eax,[-287489790+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 112+esp],eax
mov [ 216+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-1928021317
rol edi,cl	; 3 sum = 15
lea eax,[-1928021317+eax+edx]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 116+esp],eax
mov [ 220+esp],ebp	; 1
add ebx,ecx
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1  free slot, how to fill it ?
add esi,ecx	; 1
add ebp,726414452
rol esi,cl	; 3 sum = 15
lea eax,[ 726414452+eax+ebx]	; 1
add ebp,esi
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
mov [ 120+esp],eax
mov [ 224+esp],ebp	; 1
add edx,ecx
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1  free slot, how to fill it ?
add edi,ecx	; 1
add ebp,-1089828067
rol edi,cl	; 3 sum = 15
_end_round1_p5:
lea eax,[-1089828067+edx+eax]	; 1
add ebp,edi
rol eax,3	; 2
rol ebp,3	; 2

lea ecx,[eax+edx]	; 1
mov [ 20+esp],eax
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 124+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3
add eax,ebx	; 1
add ebp,esi
add eax,[ 24+esp]	; 2
add ebp,[ 128+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 24+esp],eax
add edi,ecx	; 1
mov [ 128+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 28+esp]	; 2
add ebp,[ 132+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 28+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 132+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 32+esp]	; 2
add ebp,[ 136+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 32+esp],eax
add edi,ecx	; 1
mov [ 136+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 36+esp]	; 2
add ebp,[ 140+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 36+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 140+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 40+esp]	; 2
add ebp,[ 144+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 40+esp],eax
add edi,ecx	; 1
mov [ 144+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 44+esp]	; 2
add ebp,[ 148+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 44+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 148+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 48+esp]	; 2
add ebp,[ 152+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 48+esp],eax
add edi,ecx	; 1
mov [ 152+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 52+esp]	; 2
add ebp,[ 156+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 52+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 156+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 56+esp]	; 2
add ebp,[ 160+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 56+esp],eax
add edi,ecx	; 1
mov [ 160+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 60+esp]	; 2
add ebp,[ 164+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 60+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 164+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 64+esp]	; 2
add ebp,[ 168+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 64+esp],eax
add edi,ecx	; 1
mov [ 168+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 68+esp]	; 2
add ebp,[ 172+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 68+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 172+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 72+esp]	; 2
add ebp,[ 176+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 72+esp],eax
add edi,ecx	; 1
mov [ 176+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 76+esp]	; 2
add ebp,[ 180+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 76+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 180+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 80+esp]	; 2
add ebp,[ 184+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 80+esp],eax
add edi,ecx	; 1
mov [ 184+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 84+esp]	; 2
add ebp,[ 188+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 84+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 188+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 88+esp]	; 2
add ebp,[ 192+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 88+esp],eax
add edi,ecx	; 1
mov [ 192+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 92+esp]	; 2
add ebp,[ 196+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 92+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 196+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 96+esp]	; 2
add ebp,[ 200+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 96+esp],eax
add edi,ecx	; 1
mov [ 200+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 100+esp]	; 2
add ebp,[ 204+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 100+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 204+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 104+esp]	; 2
add ebp,[ 208+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 104+esp],eax
add edi,ecx	; 1
mov [ 208+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 108+esp]	; 2
add ebp,[ 212+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 108+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 212+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 112+esp]	; 2
add ebp,[ 216+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 112+esp],eax
add edi,ecx	; 1
mov [ 216+esp],ebp
rol edi,cl	; 3 sum = 16
add eax,edx	; 1
add ebp,edi
add eax,[ 116+esp]	; 2
add ebp,[ 220+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+edx]	; 1
mov [ 116+esp],eax
add ebx,ecx	; 1  free slot, how to fill it ?
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 1
mov [ 220+esp],ebp
add esi,ecx	; 1
add edx,ebx
rol esi,cl	; 3 sum = 17
add eax,ebx	; 1
add ebp,esi
add eax,[ 120+esp]	; 2
add ebp,[ 224+esp]
rol eax,3	; 2
rol ebp,3	; 2
lea ecx,[eax+ebx]	; 1
add edx,eax
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 1
mov [ 120+esp],eax
add edi,ecx	; 1
mov [ 224+esp],ebp
rol edi,cl	; 3 sum = 16
_end_round2_p5:
mov [ 248+esp],ebp
mov [ 256+esp],esi
mov [ 252+esp],edi
add eax,edx	; 1 A = ROTL3(S00 + A + L1);
mov ebp,[ 20+esp]
add eax,ebp	; 1
mov esi,[ 228+esp]	;   eA = P_0 + A;
rol eax,3	; 2
add esi,eax	; 1
lea ecx,[eax+edx]	;   L0 = ROTL(L0 + A + L1, A + L1);
add ebx,ecx	; 1
mov ebp,[ 24+esp]
rol ebx,cl	; 3

add eax,ebx	; 1 A = ROTL3(S03 + A + L0);
add eax,ebp
rol eax,3	; 2
mov edi,[ 232+esp]	; 1 eB = P_1 + A;
lea ecx,[eax+ebx]	;   L1 = ROTL(L1 + A + L0, A + L0);
add edi,eax	; 1
add edx,ecx
mov ebp,[ 28+esp]	; 1
rol edx,cl	; 3 sum = 18
_round3_p5_S1_2:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 32+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 36+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_4:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 40+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 44+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_6:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 48+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 52+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_8:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 56+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 60+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_10:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 64+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 68+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_12:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 72+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 76+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_14:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 80+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 84+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_16:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 88+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 92+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_18:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 96+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 100+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_20:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 104+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 108+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S1_22:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 112+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 116+esp]
rol edx,cl	; 3 sum = 24
_end_round3_1_p5:
add eax,edx	; 1 A = ROTL3(S24 + A + L1);
mov ecx,edi	; eA = ROTL(eA ^ eB, eB) + A
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_1_p5

lea ecx,[eax+edx]	; 1 L0 = ROTL(L0 + A + L1, A + L1);
mov ebp,[ 120+esp]
add ebx,ecx	; 1
rol ebx,cl	; 3
add eax,ebx	; 1 A = ROTL3(S25 + A + L0);
mov ecx,esi	; eB = ROTL(eB ^ eA, eA) + A
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1

cmp edi,[ 240+esp]
je _full_exit_p5

__exit_1_p5:
mov edx,[ 252+esp]
mov ebx,[ 256+esp]
mov eax,[ 248+esp]
add eax,edx	; 1 A = ROTL3(S00 + A + L1);
mov ebp,[ 124+esp]
add eax,ebp	; 1
mov esi,[ 228+esp]	;   eA = P_0 + A;
rol eax,3	; 2
add esi,eax	; 1
lea ecx,[eax+edx]	;   L0 = ROTL(L0 + A + L1, A + L1);
add ebx,ecx	; 1
mov ebp,[ 128+esp]
rol ebx,cl	; 3

add eax,ebx	; 1 A = ROTL3(S03 + A + L0);
add eax,ebp
rol eax,3	; 2
mov edi,[ 232+esp]	; 1 eB = P_1 + A;
lea ecx,[eax+ebx]	;   L1 = ROTL(L1 + A + L0, A + L0);
add edi,eax	; 1
add edx,ecx
mov ebp,[ 132+esp]	; 1
rol edx,cl	; 3 sum = 18
_round3_p5_S2_2:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 136+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 140+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_4:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 144+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 148+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_6:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 152+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 156+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_8:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 160+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 164+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_10:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 168+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 172+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_12:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 176+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 180+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_14:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 184+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 188+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_16:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 192+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 196+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_18:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 200+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 204+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_20:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 208+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 212+esp]
rol edx,cl	; 3 sum = 24
_round3_p5_S2_22:
add eax,edx	; 1
mov ecx,edi
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]
add ebx,ecx	; 1
mov ebp,[ 216+esp]
rol ebx,cl	; 3

add eax,ebx	; 1
mov ecx,esi
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]
add edx,ecx	; 1
mov ebp,[ 220+esp]
rol edx,cl	; 3 sum = 24
_end_round3_2_p5:
add eax,edx	; 1 A = ROTL3(S24 + A + L1);
mov ecx,edi	; eA = ROTL(eA ^ eB, eB) + A
add eax,ebp	; 1
xor esi,edi
rol eax,3	; 2
rol esi,cl	; 3
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_2_p5

lea ecx,[eax+edx]	; 1 L0 = ROTL(L0 + A + L1, A + L1);
mov ebp,[ 224+esp]
add ebx,ecx	; 1
rol ebx,cl	; 3
add eax,ebx	; 1 A = ROTL3(S25 + A + L0);
mov ecx,esi	; eB = ROTL(eB ^ eA, eA) + A
add eax,ebp	; 1
xor edi,esi
rol eax,3	; 2
rol edi,cl	; 3
add edi,eax	; 1

cmp edi,[ 240+esp]
jne __exit_2_p5
mov [ 16+esp],dword ptr 1
jmp _full_exit_p5

__exit_2_p5:

mov ebx,[ 264+esp]
mov edx,[ 260+esp]


add edx,33554432
jc _next_inc_p5

mov [ 260+esp],edx
mov esi,ebx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_p5
mov eax,[ 276+esp]	; pointer to rc5unitwork
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_p5

_next_inc_p5:
add edx,65536
test edx,16711680
jnz _next_iter2_p5

sub edx,16777216
add edx,256
test edx,-256
jnz _next_iter2_p5

sub edx,65536
add edx,1
test edx,255
jnz _next_iter2_p5

sub edx,256
add ebx,16777216
jnc _next_iter_p5

add ebx,65536
test ebx,16711680
jnz _next_iter_p5

sub ebx,16777216
add ebx,256
test ebx,-256
jnz _next_iter_p5

sub ebx,65536
add ebx,1


_next_iter_p5:
mov [ 264+esp],ebx
_next_iter2_p5:
mov [ 260+esp],edx
mov esi,ebx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_p5
mov eax,[ 276+esp]	; pointer to rc5unitwork
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)

_full_exit_p5:
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
add esp,256
ret
_rc5_unit_func_p5 endp
_TEXT   ends
end

