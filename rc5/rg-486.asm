.386p
.model flat
_TEXT   segment dword public use32 'CODE'
align 4
public _rc5_unit_func_486
_rc5_unit_func_486 proc near

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
add ebx,-1089828067	; 1
rol ebx,29	; 3
mov [ 272+esp],ebx	; 1

lea eax,[ 354637369+ebx]	; 1
rol eax,3	; 2
mov [ 276+esp],eax	; 1

lea ecx,[eax+ebx]	; 2
mov [ 280+esp],ecx	; 1

_loaded_486:
mov ebx,[ 272+esp]	; 1
mov esi,ebx	; 1
mov eax,[ 276+esp]	; 1
mov ebp,eax	; 1

mov ecx,[ 280+esp]
add edx,ecx	; 1
rol edx,cl	; 3
add edi,ecx	; 1
rol edi,cl	; 3 sum = 12
lea eax,[-196066091+eax+edx]	; 2
lea ebp,[-196066091+ebp+edi]	; 2
rol eax,3	; 2
mov [ 28+esp],eax	; 1
rol ebp,3	; 2
mov [ 132+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-1836597618+eax+ebx]	; 2
lea ebp,[-1836597618+ebp+esi]	; 2
rol eax,3	; 2
mov [ 32+esp],eax	; 1
rol ebp,3	; 2
mov [ 136+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[ 817838151+eax+edx]	; 2
lea ebp,[ 817838151+ebp+edi]	; 2
rol eax,3	; 2
mov [ 36+esp],eax	; 1
rol ebp,3	; 2
mov [ 140+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-822693376+eax+ebx]	; 2
lea ebp,[-822693376+ebp+esi]	; 2
rol eax,3	; 2
mov [ 40+esp],eax	; 1
rol ebp,3	; 2
mov [ 144+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[ 1831742393+eax+edx]	; 2
lea ebp,[ 1831742393+ebp+edi]	; 2
rol eax,3	; 2
mov [ 44+esp],eax	; 1
rol ebp,3	; 2
mov [ 148+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[ 191210866+eax+ebx]	; 2
lea ebp,[ 191210866+ebp+esi]	; 2
rol eax,3	; 2
mov [ 48+esp],eax	; 1
rol ebp,3	; 2
mov [ 152+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[-1449320661+eax+edx]	; 2
lea ebp,[-1449320661+ebp+edi]	; 2
rol eax,3	; 2
mov [ 52+esp],eax	; 1
rol ebp,3	; 2
mov [ 156+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[ 1205115108+eax+ebx]	; 2
lea ebp,[ 1205115108+ebp+esi]	; 2
rol eax,3	; 2
mov [ 56+esp],eax	; 1
rol ebp,3	; 2
mov [ 160+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[-435416419+eax+edx]	; 2
lea ebp,[-435416419+ebp+edi]	; 2
rol eax,3	; 2
mov [ 60+esp],eax	; 1
rol ebp,3	; 2
mov [ 164+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-2075947946+eax+ebx]	; 2
lea ebp,[-2075947946+ebp+esi]	; 2
rol eax,3	; 2
mov [ 64+esp],eax	; 1
rol ebp,3	; 2
mov [ 168+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[ 578487823+eax+edx]	; 2
lea ebp,[ 578487823+ebp+edi]	; 2
rol eax,3	; 2
mov [ 68+esp],eax	; 1
rol ebp,3	; 2
mov [ 172+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-1062043704+eax+ebx]	; 2
lea ebp,[-1062043704+ebp+esi]	; 2
rol eax,3	; 2
mov [ 72+esp],eax	; 1
rol ebp,3	; 2
mov [ 176+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[ 1592392065+eax+edx]	; 2
lea ebp,[ 1592392065+ebp+edi]	; 2
rol eax,3	; 2
mov [ 76+esp],eax	; 1
rol ebp,3	; 2
mov [ 180+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-48139462+eax+ebx]	; 2
lea ebp,[-48139462+ebp+esi]	; 2
rol eax,3	; 2
mov [ 80+esp],eax	; 1
rol ebp,3	; 2
mov [ 184+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[-1688670989+eax+edx]	; 2
lea ebp,[-1688670989+ebp+edi]	; 2
rol eax,3	; 2
mov [ 84+esp],eax	; 1
rol ebp,3	; 2
mov [ 188+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[ 965764780+eax+ebx]	; 2
lea ebp,[ 965764780+ebp+esi]	; 2
rol eax,3	; 2
mov [ 88+esp],eax	; 1
rol ebp,3	; 2
mov [ 192+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[-674766747+eax+edx]	; 2
lea ebp,[-674766747+ebp+edi]	; 2
rol eax,3	; 2
mov [ 92+esp],eax	; 1
rol ebp,3	; 2
mov [ 196+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[ 1979669022+eax+ebx]	; 2
lea ebp,[ 1979669022+ebp+esi]	; 2
rol eax,3	; 2
mov [ 96+esp],eax	; 1
rol ebp,3	; 2
mov [ 200+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[ 339137495+eax+edx]	; 2
lea ebp,[ 339137495+ebp+edi]	; 2
rol eax,3	; 2
mov [ 100+esp],eax	; 1
rol ebp,3	; 2
mov [ 204+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-1301394032+eax+ebx]	; 2
lea ebp,[-1301394032+ebp+esi]	; 2
rol eax,3	; 2
mov [ 104+esp],eax	; 1
rol ebp,3	; 2
mov [ 208+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[ 1353041737+eax+edx]	; 2
lea ebp,[ 1353041737+ebp+edi]	; 2
rol eax,3	; 2
mov [ 108+esp],eax	; 1
rol ebp,3	; 2
mov [ 212+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[-287489790+eax+ebx]	; 2
lea ebp,[-287489790+ebp+esi]	; 2
rol eax,3	; 2
mov [ 112+esp],eax	; 1
rol ebp,3	; 2
mov [ 216+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
lea eax,[-1928021317+eax+edx]	; 2
lea ebp,[-1928021317+ebp+edi]	; 2
rol eax,3	; 2
mov [ 116+esp],eax	; 1
rol ebp,3	; 2
mov [ 220+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 22
lea eax,[ 726414452+eax+ebx]	; 2
lea ebp,[ 726414452+ebp+esi]	; 2
rol eax,3	; 2
mov [ 120+esp],eax	; 1
rol ebp,3	; 2
mov [ 224+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 22
_end_round1_486:
lea eax,[-1089828067+eax+edx]	; 2
lea ebp,[-1089828067+ebp+edi]	; 2
rol eax,3	; 2
mov [ 20+esp],eax	; 1
rol ebp,3	; 2
mov [ 124+esp],ebp	; 1

lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3
mov ecx,[ 276+esp]	; 1
add eax,ebx	; 1
add eax,ecx	; 1
add ebp,esi	; 1
add ebp,ecx	; 1
rol eax,3	; 2
mov [ 24+esp],eax	; 1
rol ebp,3	; 2
mov [ 128+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 23
add eax,edx	; 1
add eax,[ 28+esp]	; 2
add ebp,edi	; 1
add ebp,[ 132+esp]	; 2
rol eax,3	; 2
mov [ 28+esp],eax	; 1
rol ebp,3	; 2
mov [ 132+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 32+esp]	; 2
add ebp,esi	; 1
add ebp,[ 136+esp]	; 2
rol eax,3	; 2
mov [ 32+esp],eax	; 1
rol ebp,3	; 2
mov [ 136+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 36+esp]	; 2
add ebp,edi	; 1
add ebp,[ 140+esp]	; 2
rol eax,3	; 2
mov [ 36+esp],eax	; 1
rol ebp,3	; 2
mov [ 140+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 40+esp]	; 2
add ebp,esi	; 1
add ebp,[ 144+esp]	; 2
rol eax,3	; 2
mov [ 40+esp],eax	; 1
rol ebp,3	; 2
mov [ 144+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 44+esp]	; 2
add ebp,edi	; 1
add ebp,[ 148+esp]	; 2
rol eax,3	; 2
mov [ 44+esp],eax	; 1
rol ebp,3	; 2
mov [ 148+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 48+esp]	; 2
add ebp,esi	; 1
add ebp,[ 152+esp]	; 2
rol eax,3	; 2
mov [ 48+esp],eax	; 1
rol ebp,3	; 2
mov [ 152+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 52+esp]	; 2
add ebp,edi	; 1
add ebp,[ 156+esp]	; 2
rol eax,3	; 2
mov [ 52+esp],eax	; 1
rol ebp,3	; 2
mov [ 156+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 56+esp]	; 2
add ebp,esi	; 1
add ebp,[ 160+esp]	; 2
rol eax,3	; 2
mov [ 56+esp],eax	; 1
rol ebp,3	; 2
mov [ 160+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 60+esp]	; 2
add ebp,edi	; 1
add ebp,[ 164+esp]	; 2
rol eax,3	; 2
mov [ 60+esp],eax	; 1
rol ebp,3	; 2
mov [ 164+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 64+esp]	; 2
add ebp,esi	; 1
add ebp,[ 168+esp]	; 2
rol eax,3	; 2
mov [ 64+esp],eax	; 1
rol ebp,3	; 2
mov [ 168+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 68+esp]	; 2
add ebp,edi	; 1
add ebp,[ 172+esp]	; 2
rol eax,3	; 2
mov [ 68+esp],eax	; 1
rol ebp,3	; 2
mov [ 172+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 72+esp]	; 2
add ebp,esi	; 1
add ebp,[ 176+esp]	; 2
rol eax,3	; 2
mov [ 72+esp],eax	; 1
rol ebp,3	; 2
mov [ 176+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 76+esp]	; 2
add ebp,edi	; 1
add ebp,[ 180+esp]	; 2
rol eax,3	; 2
mov [ 76+esp],eax	; 1
rol ebp,3	; 2
mov [ 180+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 80+esp]	; 2
add ebp,esi	; 1
add ebp,[ 184+esp]	; 2
rol eax,3	; 2
mov [ 80+esp],eax	; 1
rol ebp,3	; 2
mov [ 184+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 84+esp]	; 2
add ebp,edi	; 1
add ebp,[ 188+esp]	; 2
rol eax,3	; 2
mov [ 84+esp],eax	; 1
rol ebp,3	; 2
mov [ 188+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 88+esp]	; 2
add ebp,esi	; 1
add ebp,[ 192+esp]	; 2
rol eax,3	; 2
mov [ 88+esp],eax	; 1
rol ebp,3	; 2
mov [ 192+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 92+esp]	; 2
add ebp,edi	; 1
add ebp,[ 196+esp]	; 2
rol eax,3	; 2
mov [ 92+esp],eax	; 1
rol ebp,3	; 2
mov [ 196+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 96+esp]	; 2
add ebp,esi	; 1
add ebp,[ 200+esp]	; 2
rol eax,3	; 2
mov [ 96+esp],eax	; 1
rol ebp,3	; 2
mov [ 200+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 100+esp]	; 2
add ebp,edi	; 1
add ebp,[ 204+esp]	; 2
rol eax,3	; 2
mov [ 100+esp],eax	; 1
rol ebp,3	; 2
mov [ 204+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 104+esp]	; 2
add ebp,esi	; 1
add ebp,[ 208+esp]	; 2
rol eax,3	; 2
mov [ 104+esp],eax	; 1
rol ebp,3	; 2
mov [ 208+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 108+esp]	; 2
add ebp,edi	; 1
add ebp,[ 212+esp]	; 2
rol eax,3	; 2
mov [ 108+esp],eax	; 1
rol ebp,3	; 2
mov [ 212+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 112+esp]	; 2
add ebp,esi	; 1
add ebp,[ 216+esp]	; 2
rol eax,3	; 2
mov [ 112+esp],eax	; 1
rol ebp,3	; 2
mov [ 216+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
add eax,edx	; 1
add eax,[ 116+esp]	; 2
add ebp,edi	; 1
add ebp,[ 220+esp]	; 2
rol eax,3	; 2
mov [ 116+esp],eax	; 1
rol ebp,3	; 2
mov [ 220+esp],ebp	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3
lea ecx,[ebp+edi]	; 2
add esi,ecx	; 1
rol esi,cl	; 3 sum = 24
add eax,ebx	; 1
add eax,[ 120+esp]	; 2
add ebp,esi	; 1
add ebp,[ 224+esp]	; 2
rol eax,3	; 2
mov [ 120+esp],eax	; 1
rol ebp,3	; 2
mov [ 224+esp],ebp	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3
lea ecx,[ebp+esi]	; 2
add edi,ecx	; 1
rol edi,cl	; 3 sum = 24
_end_round2_486:
mov [ 248+esp],ebp
mov [ 256+esp],esi
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
_round3_486_S1_2:
add eax,edx	; 1
add eax,[ 28+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 32+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_4:
add eax,edx	; 1
add eax,[ 36+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 40+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_6:
add eax,edx	; 1
add eax,[ 44+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 48+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_8:
add eax,edx	; 1
add eax,[ 52+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 56+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_10:
add eax,edx	; 1
add eax,[ 60+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 64+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_12:
add eax,edx	; 1
add eax,[ 68+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 72+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_14:
add eax,edx	; 1
add eax,[ 76+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 80+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_16:
add eax,edx	; 1
add eax,[ 84+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 88+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_18:
add eax,edx	; 1
add eax,[ 92+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 96+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_20:
add eax,edx	; 1
add eax,[ 100+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 104+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S1_22:
add eax,edx	; 1
add eax,[ 108+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 112+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_end_round3_1_486:
add eax,edx	; 1 A = ROTL3(S24 + A + L1);
add eax,[ 116+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1 eA = ROTL(eA ^ eB, eB) + A
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_1_486

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
je _full_exit_486

__exit_1_486:
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
_round3_486_S2_2:
add eax,edx	; 1
add eax,[ 132+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 136+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_4:
add eax,edx	; 1
add eax,[ 140+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 144+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_6:
add eax,edx	; 1
add eax,[ 148+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 152+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_8:
add eax,edx	; 1
add eax,[ 156+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 160+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_10:
add eax,edx	; 1
add eax,[ 164+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 168+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_12:
add eax,edx	; 1
add eax,[ 172+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 176+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_14:
add eax,edx	; 1
add eax,[ 180+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 184+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_16:
add eax,edx	; 1
add eax,[ 188+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 192+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_18:
add eax,edx	; 1
add eax,[ 196+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 200+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_20:
add eax,edx	; 1
add eax,[ 204+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 208+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_round3_486_S2_22:
add eax,edx	; 1
add eax,[ 212+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1
lea ecx,[eax+edx]	; 2
add ebx,ecx	; 1
rol ebx,cl	; 3

add eax,ebx	; 1
add eax,[ 216+esp]	; 2
rol eax,3	; 2
mov ecx,esi	; 1
xor edi,esi	; 1
rol edi,cl	; 3
add edi,eax	; 1
lea ecx,[eax+ebx]	; 2
add edx,ecx	; 1
rol edx,cl	; 3 sum = 34
_end_round3_2_486:
add eax,edx	; 1 A = ROTL3(S24 + A + L1);
add eax,[ 220+esp]	; 2
rol eax,3	; 2
mov ecx,edi	; 1 eA = ROTL(eA ^ eB, eB) + A
xor esi,edi	; 1
rol esi,cl	; 3
add esi,eax	; 1

cmp esi,[ 236+esp]
jne __exit_2_486

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
jne __exit_2_486
mov [ 16+esp],dword ptr 1
jmp _full_exit_486

__exit_2_486:

mov edx,[ 260+esp]


add edx,33554432
jc _next_inc_486

_next_iter_486:
mov [ 260+esp],edx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_486
mov eax,[ 288+esp]	; pointer to rc5unitwork
mov ebx,[ 264+esp]
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_486

_next_inc_486:
add edx,65536
test edx,16711680
jnz _next_iter_486

sub edx,16777216
add edx,256
test edx,-256
jnz _next_iter_486

sub edx,65536
add edx,1
test edx,255
jnz _next_iter_486

 	; we should never go here, it would mean we have iterated 2^32 times ...
 	; stop the client, something went wrong
;mov 0,0	; generate a segfault

_full_exit_486:
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
_rc5_unit_func_486 endp
_TEXT   ends
end

