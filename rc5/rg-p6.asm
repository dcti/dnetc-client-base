.386p
.model flat
_TEXT   segment dword public use32 'CODE'
align 4
public _rc5_unit_func_p6
_rc5_unit_func_p6 proc near

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
add ebx,-1089828067
rol ebx,29
mov [ 272+esp],ebx

lea eax,[ 354637369+ebx]
rol eax,3
mov [ 276+esp],eax

lea ecx,[eax+ebx]
mov [ 280+esp],ecx

_loaded_p6:

mov ebx,[ 272+esp]
mov eax,[ 276+esp]
mov esi,ebx
mov ebp,eax

mov ecx,[ 280+esp]
add edx,ecx
rol edx,cl
add edi,ecx
lea eax,[-196066091+eax+edx]
rol edi,cl
lea ebp,[-196066091+ebp+edi]
rol eax,3
rol ebp,3
mov [ 28+esp],eax
mov [ 132+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-1836597618+eax+ebx]
rol esi,cl
lea ebp,[-1836597618+ebp+esi]
rol eax,3
rol ebp,3
mov [ 32+esp],eax
mov [ 136+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[ 817838151+eax+edx]
rol edi,cl
lea ebp,[ 817838151+ebp+edi]
rol eax,3
rol ebp,3
mov [ 36+esp],eax
mov [ 140+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-822693376+eax+ebx]
rol esi,cl
lea ebp,[-822693376+ebp+esi]
rol eax,3
rol ebp,3
mov [ 40+esp],eax
mov [ 144+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[ 1831742393+eax+edx]
rol edi,cl
lea ebp,[ 1831742393+ebp+edi]
rol eax,3
rol ebp,3
mov [ 44+esp],eax
mov [ 148+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[ 191210866+eax+ebx]
rol esi,cl
lea ebp,[ 191210866+ebp+esi]
rol eax,3
rol ebp,3
mov [ 48+esp],eax
mov [ 152+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[-1449320661+eax+edx]
rol edi,cl
lea ebp,[-1449320661+ebp+edi]
rol eax,3
rol ebp,3
mov [ 52+esp],eax
mov [ 156+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[ 1205115108+eax+ebx]
rol esi,cl
lea ebp,[ 1205115108+ebp+esi]
rol eax,3
rol ebp,3
mov [ 56+esp],eax
mov [ 160+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[-435416419+eax+edx]
rol edi,cl
lea ebp,[-435416419+ebp+edi]
rol eax,3
rol ebp,3
mov [ 60+esp],eax
mov [ 164+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-2075947946+eax+ebx]
rol esi,cl
lea ebp,[-2075947946+ebp+esi]
rol eax,3
rol ebp,3
mov [ 64+esp],eax
mov [ 168+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[ 578487823+eax+edx]
rol edi,cl
lea ebp,[ 578487823+ebp+edi]
rol eax,3
rol ebp,3
mov [ 68+esp],eax
mov [ 172+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-1062043704+eax+ebx]
rol esi,cl
lea ebp,[-1062043704+ebp+esi]
rol eax,3
rol ebp,3
mov [ 72+esp],eax
mov [ 176+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[ 1592392065+eax+edx]
rol edi,cl
lea ebp,[ 1592392065+ebp+edi]
rol eax,3
rol ebp,3
mov [ 76+esp],eax
mov [ 180+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-48139462+eax+ebx]
rol esi,cl
lea ebp,[-48139462+ebp+esi]
rol eax,3
rol ebp,3
mov [ 80+esp],eax
mov [ 184+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[-1688670989+eax+edx]
rol edi,cl
lea ebp,[-1688670989+ebp+edi]
rol eax,3
rol ebp,3
mov [ 84+esp],eax
mov [ 188+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[ 965764780+eax+ebx]
rol esi,cl
lea ebp,[ 965764780+ebp+esi]
rol eax,3
rol ebp,3
mov [ 88+esp],eax
mov [ 192+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[-674766747+eax+edx]
rol edi,cl
lea ebp,[-674766747+ebp+edi]
rol eax,3
rol ebp,3
mov [ 92+esp],eax
mov [ 196+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[ 1979669022+eax+ebx]
rol esi,cl
lea ebp,[ 1979669022+ebp+esi]
rol eax,3
rol ebp,3
mov [ 96+esp],eax
mov [ 200+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[ 339137495+eax+edx]
rol edi,cl
lea ebp,[ 339137495+ebp+edi]
rol eax,3
rol ebp,3
mov [ 100+esp],eax
mov [ 204+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-1301394032+eax+ebx]
rol esi,cl
lea ebp,[-1301394032+ebp+esi]
rol eax,3
rol ebp,3
mov [ 104+esp],eax
mov [ 208+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[ 1353041737+eax+edx]
rol edi,cl
lea ebp,[ 1353041737+ebp+edi]
rol eax,3
rol ebp,3
mov [ 108+esp],eax
mov [ 212+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[-287489790+eax+ebx]
rol esi,cl
lea ebp,[-287489790+ebp+esi]
rol eax,3
rol ebp,3
mov [ 112+esp],eax
mov [ 216+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[-1928021317+eax+edx]
rol edi,cl
lea ebp,[-1928021317+ebp+edi]
rol eax,3
rol ebp,3
mov [ 116+esp],eax
mov [ 220+esp],ebp
lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add esi,ecx
lea eax,[ 726414452+eax+ebx]
rol esi,cl
lea ebp,[ 726414452+ebp+esi]
rol eax,3
rol ebp,3
mov [ 120+esp],eax
mov [ 224+esp],ebp
lea ecx,[eax+ebx]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add edi,ecx
lea eax,[-1089828067+eax+edx]	; wrap with start of ROUND2
rol edi,cl
_end_round1_p6:
 	;leal 0xbf0a8b1d(%edx,%eax),  %eax   # already in ROUND1_LAST
lea ebp,[-1089828067+edi+ebp]
rol eax,3
rol ebp,3
mov [ 20+esp],eax
mov [ 124+esp],ebp

lea ecx,[eax+edx]
add eax,[ 276+esp]
add ebx,ecx
rol ebx,cl

lea ecx,[ebp+edi]
add ebp,[ 276+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 24+esp],eax
mov [ 128+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 28+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 132+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 28+esp],eax
mov [ 132+esp],ebp
lea ecx,[eax+edx]
add eax,[ 32+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 136+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 32+esp],eax
mov [ 136+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 36+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 140+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 36+esp],eax
mov [ 140+esp],ebp
lea ecx,[eax+edx]
add eax,[ 40+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 144+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 40+esp],eax
mov [ 144+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 44+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 148+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 44+esp],eax
mov [ 148+esp],ebp
lea ecx,[eax+edx]
add eax,[ 48+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 152+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 48+esp],eax
mov [ 152+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 52+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 156+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 52+esp],eax
mov [ 156+esp],ebp
lea ecx,[eax+edx]
add eax,[ 56+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 160+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 56+esp],eax
mov [ 160+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 60+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 164+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 60+esp],eax
mov [ 164+esp],ebp
lea ecx,[eax+edx]
add eax,[ 64+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 168+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 64+esp],eax
mov [ 168+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 68+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 172+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 68+esp],eax
mov [ 172+esp],ebp
lea ecx,[eax+edx]
add eax,[ 72+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 176+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 72+esp],eax
mov [ 176+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 76+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 180+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 76+esp],eax
mov [ 180+esp],ebp
lea ecx,[eax+edx]
add eax,[ 80+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 184+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 80+esp],eax
mov [ 184+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 84+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 188+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 84+esp],eax
mov [ 188+esp],ebp
lea ecx,[eax+edx]
add eax,[ 88+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 192+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 88+esp],eax
mov [ 192+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 92+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 196+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 92+esp],eax
mov [ 196+esp],ebp
lea ecx,[eax+edx]
add eax,[ 96+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 200+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 96+esp],eax
mov [ 200+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 100+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 204+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 100+esp],eax
mov [ 204+esp],ebp
lea ecx,[eax+edx]
add eax,[ 104+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 208+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 104+esp],eax
mov [ 208+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 108+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 212+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 108+esp],eax
mov [ 212+esp],ebp
lea ecx,[eax+edx]
add eax,[ 112+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 216+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 112+esp],eax
mov [ 216+esp],ebp
lea ecx,[eax+ebx]
add eax,[ 116+esp]
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add ebp,[ 220+esp]
add edi,ecx
rol edi,cl
add eax,edx
add ebp,edi
rol eax,3
rol ebp,3
mov [ 116+esp],eax
mov [ 220+esp],ebp
lea ecx,[eax+edx]
add eax,[ 120+esp]
add ebx,ecx
rol ebx,cl
lea ecx,[ebp+edi]
add ebp,[ 224+esp]
add esi,ecx
rol esi,cl
add eax,ebx
add ebp,esi
rol eax,3
rol ebp,3
mov [ 120+esp],eax
mov [ 224+esp],ebp
lea ecx,[eax+ebx]
 	; addl ((25+1)*4)+4+16(%esp),%eax
add edx,ecx
rol edx,cl
lea ecx,[ebp+esi]
add eax,[ 20+esp]	; wrap with first part of ROUND3
add edi,ecx
rol edi,cl
_end_round2_p6:
mov [ 248+esp],ebp
mov [ 256+esp],esi
mov [ 252+esp],edi
 	;addl ((0)*4)+4+16(%esp),%eax # already in ROUND_2_LAST
add eax,edx
rol eax,3
mov esi,[ 228+esp]
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 24+esp]
add eax,ebx
rol eax,3
mov edi,[ 232+esp]
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 28+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 32+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 36+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 40+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 44+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 48+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 52+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 56+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 60+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 64+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 68+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 72+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 76+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 80+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 84+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 88+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 92+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 96+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 100+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 104+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 108+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 112+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
_end_round3_1_p6:
add eax,[ 116+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax

cmp esi,[ 236+esp]
jne __exit_1_p6

lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
add eax,[ 120+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax

cmp edi,[ 240+esp]
je _full_exit_p6

__exit_1_p6:
mov edx,[ 252+esp]
mov ebx,[ 256+esp]
mov eax,[ 248+esp]
add eax,[ 124+esp]
add eax,edx
rol eax,3
mov esi,[ 228+esp]
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 128+esp]
add eax,ebx
rol eax,3
mov edi,[ 232+esp]
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 132+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 136+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 140+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 144+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 148+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 152+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 156+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 160+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 164+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 168+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 172+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 176+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 180+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 184+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 188+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 192+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 196+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 200+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 204+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 208+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
add eax,[ 212+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
lea ecx,[eax+edx]
add esi,eax
add ebx,ecx
rol ebx,cl

add eax,[ 216+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
lea ecx,[eax+ebx]
add edi,eax
add edx,ecx
rol edx,cl
_end_round3_2_p6:
add eax,[ 220+esp]
add eax,edx
rol eax,3
mov ecx,edi
xor esi,edi
rol esi,cl
add esi,eax

cmp esi,[ 236+esp]
jne __exit_2_p6

lea ecx,[eax+edx]
add ebx,ecx
rol ebx,cl
add eax,[ 224+esp]
add eax,ebx
rol eax,3
mov ecx,esi
xor edi,esi
rol edi,cl
add edi,eax

cmp edi,[ 240+esp]
jne __exit_2_p6
mov [ 16+esp],dword ptr 1
jmp _full_exit_p6

__exit_2_p6:

mov edx,[ 260+esp]


add edx,33554432
jc _next_inc_p6

_next_iter_p6:
mov [ 260+esp],edx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_p6
mov eax,[ 288+esp]	; pointer to rc5unitwork
mov ebx,[ 264+esp]
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_p6

_next_inc_p6:
add edx,65536
test edx,16711680
jnz _next_iter_p6

sub edx,16777216
add edx,256
test edx,-256
jnz _next_iter_p6

sub edx,65536
add edx,1
test edx,255
jnz _next_iter_p6

 	; we should never go here, it would mean we have iterated 2^32 times ...
 	; stop the client, something went wrong
;mov 0,0	; generate a segfault

_full_exit_p6:
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
_rc5_unit_func_p6 endp
_TEXT   ends
end

