.386p
.model flat
_TEXT   segment dword public use32 'CODE'
align 4
public _rc5_unit_func_k5
_rc5_unit_func_k5 proc near

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

_loaded_k5:
mov ebx,[ 272+esp]	; 1 ld
mov eax,[ 276+esp]	;   ld
mov esi,ebx	; 1 alu
mov ebp,eax	;   alu
mov ecx,[ 280+esp]	;   ld
add edx,ecx	; 1 alu
add edi,ecx	;   alu
rol edx,cl	; 1 alu
rol edi,cl	;   alu  sum = 4
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[-196066091+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-196066091+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 28+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 132+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-1836597618+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-1836597618+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 32+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 136+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[ 817838151+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 817838151+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 36+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 140+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-822693376+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-822693376+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 40+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 144+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[ 1831742393+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 1831742393+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 44+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 148+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[ 191210866+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 191210866+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 48+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 152+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[-1449320661+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-1449320661+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 52+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 156+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[ 1205115108+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 1205115108+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 56+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 160+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[-435416419+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-435416419+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 60+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 164+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-2075947946+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-2075947946+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 64+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 168+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[ 578487823+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 578487823+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 68+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 172+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-1062043704+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-1062043704+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 72+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 176+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[ 1592392065+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 1592392065+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 76+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 180+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-48139462+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-48139462+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 80+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 184+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[-1688670989+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-1688670989+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 84+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 188+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[ 965764780+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 965764780+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 88+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 192+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[-674766747+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-674766747+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 92+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 196+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[ 1979669022+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 1979669022+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 96+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 200+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[ 339137495+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 339137495+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 100+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 204+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-1301394032+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-1301394032+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 104+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 208+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[ 1353041737+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 1353041737+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 108+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 212+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[-287489790+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-287489790+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 112+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 216+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
 	;roll %cl,  %edi     1 alu previous iteration
add ebx,edx	; alu
lea eax,[-1928021317+edx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[-1928021317+edi+ebp]	; ld
add esi,edi	; alu
rol ebp,3	; 1 alu
mov [ 116+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; alu
rol ebx,cl	; 1 alu
mov [ 220+esp],ebp	; st
lea ecx,[ebp+edi]	;  ld
add esi,ebp	; alu
rol esi,cl	; 1 alu
add edx,ebx	; alu
lea eax,[ 726414452+ebx+eax]	;   ld
rol eax,3	; 1 alu
lea ebp,[ 726414452+esi+ebp]	; ld
add edi,esi	; alu
rol ebp,3	; 1 alu
mov [ 120+esp],eax	; st
lea ecx,[eax+ebx]	;  ld
add edx,eax	; alu
rol edx,cl	; 1 alu
mov [ 224+esp],ebp	; st
lea ecx,[ebp+esi]	; ld
add edi,ebp	; alu
rol edi,cl	; 1 alu sum = 8
_end_round1_k5:
 	;roll %cl,   %edi     1 alu previous iteration (round 1)
add eax,edx	;   alu
add eax,-1089828067	; 1 alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	; alu
add ebp,-1089828067	; 1 alu
rol eax,3	; alu
mov [ 20+esp],eax	; st
lea ecx,[eax+edx]	; ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 276+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 124+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 276+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 24+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 128+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 28+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 132+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 28+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 32+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 132+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 136+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 32+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 136+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 36+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 140+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 36+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 40+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 140+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 144+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 40+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 144+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 44+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 148+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 44+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 48+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 148+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 152+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 48+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 152+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 52+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 156+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 52+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 56+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 156+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 160+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 56+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 160+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 60+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 164+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 60+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 64+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 164+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 168+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 64+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 168+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 68+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 172+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 68+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 72+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 172+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 176+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 72+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 176+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 76+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 180+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 76+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 80+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 180+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 184+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 80+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 184+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 84+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 188+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 84+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 88+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 188+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 192+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 88+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 192+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 92+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 196+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 92+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 96+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 196+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 200+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 96+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 200+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 100+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 204+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 100+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 104+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 204+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 208+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 104+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 208+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 108+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 212+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 108+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 112+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 212+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 216+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 112+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 216+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
 	;roll %cl,     %edi    3 alu previous iteration
add eax,edx	;   alu
add eax,[ 116+esp]	;  ld  - alu
lea ebx,[edx+ebx]	; ld
add ebp,edi	;     - alu
add ebp,[ 220+esp]	;  ld  - alu
rol eax,3	;   alu
mov [ 116+esp],eax	;   st
lea ecx,[eax+edx]	;   ld
add ebx,eax	; 2 alu
rol ebp,3	; alu
add eax,[ 120+esp]	; ld  - alu
lea esi,[edi+esi]	; ld
mov [ 220+esp],ebp	;  st
rol ebx,cl	;  alu
lea ecx,[ebp+edi]	;  ld
add eax,ebx	; 2 alu
add esi,ebp	;   alu
add ebp,[ 224+esp]	; ld  - alu
rol esi,cl	;  alu
lea edx,[ebx+edx]	;  ld
rol eax,3	; 1 alu
add ebp,esi	;  alu
lea edi,[esi+edi]	; ld
rol ebp,3	; 1 alu
mov [ 120+esp],eax	; st
lea ecx,[eax+ebx]	; ld
add edx,eax	; 1 alu
mov [ 224+esp],ebp	; st
rol edx,cl	; alu
lea ecx,[ebp+esi]	; 1 ld
add edi,ebp	; alu
rol edi,cl	; .. alu sum = 11
_end_round2_k5:
mov [ 248+esp],ebp
mov [ 256+esp],esi
mov [ 252+esp],edi
add eax,edx	; 2 alu
add eax,[ 20+esp]	; ld  - alu
mov esi,[ 228+esp]	; ld
add ebx,edx	; alu
rol eax,3	; 1 alu
lea ecx,[eax+edx]	; 1 ld
add ebx,eax	; alu
rol ebx,cl	; 2 alu
add esi,eax	; alu
lea eax,[eax+ebx]	; ld
add eax,[ 24+esp]	; ld  - alu
mov edi,[ 232+esp]	;  ld
rol eax,3	; 1 alu
lea ebp,[ebx+edx]	; ld
lea edx,[eax+ebp]	; 1 ld
lea ecx,[eax+ebx]	; ld
rol edx,cl	; 1 alu
add edi,eax	; alu sum = 9
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 28+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 32+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 36+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 40+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 44+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 48+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 52+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 56+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 60+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 64+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 68+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 72+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 76+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 80+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 84+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 88+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 92+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 96+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 100+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 104+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 108+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 112+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
_end_round3_1_k5:
add eax,edx
mov ebp,[ 116+esp]
mov ecx,edi
add eax,ebp
xor esi,edi
rol eax,3
rol esi,cl
add esi,eax

cmp esi,[ 236+esp]
jne __exit_1_k5

lea ecx,[eax+edx]
mov ebp,[ 120+esp]
add ebx,ecx
rol ebx,cl
add eax,ebx
mov ecx,esi
add eax,ebp
xor edi,esi
rol eax,3
rol edi,cl
add edi,eax

cmp edi,[ 240+esp]
je _full_exit_k5

__exit_1_k5:
mov edx,[ 252+esp]
mov ebx,[ 256+esp]
mov eax,[ 248+esp]
add eax,edx	; 2 alu
add eax,[ 124+esp]	; ld  - alu
mov esi,[ 228+esp]	; ld
add ebx,edx	; alu
rol eax,3	; 1 alu
lea ecx,[eax+edx]	; 1 ld
add ebx,eax	; alu
rol ebx,cl	; 2 alu
add esi,eax	; alu
lea eax,[eax+ebx]	; ld
add eax,[ 128+esp]	; ld  - alu
mov edi,[ 232+esp]	;  ld
rol eax,3	; 1 alu
lea ebp,[ebx+edx]	; ld
lea edx,[eax+ebp]	; 1 ld
lea ecx,[eax+ebx]	; ld
rol edx,cl	; 1 alu
add edi,eax	; alu sum = 9
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 132+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 136+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 140+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 144+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 148+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 152+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 156+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 160+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 164+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 168+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 172+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 176+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 180+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 184+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 188+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 192+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 196+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 200+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 204+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 208+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
 	;roll %cl,  %edx    1  alu (previous iteration)
 	;addl %eax, %edi       alu (previous iteration)
mov ebp,[ 212+esp]	;   ld
lea ebp,[ebp+edx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor esi,edi	;   alu
rol eax,3	; 1 alu
mov ecx,edi	;   alu
lea ebp,[edx+ebx]	;   ld
rol esi,cl	; 1 alu
lea ebx,[eax+ebp]	;   ld
lea ecx,[eax+edx]	;   ld
rol ebx,cl	; 1 alu
add esi,eax	;   alu

mov ebp,[ 216+esp]	;   ld
lea ebp,[ebp+ebx]	; 1 ld
add eax,ebp	;   alu (data forwarding)
xor edi,esi	;   alu
rol eax,3	; 1 alu
mov ecx,esi	;   alu
lea ebp,[ebx+edx]	;   ld
rol edi,cl	; 1 alu
lea edx,[eax+ebp]	;   ld
lea ecx,[eax+ebx]	;   ld
rol edx,cl	; 1 alu
add edi,eax	;   alu sum = 8
_end_round3_2_k5:
add eax,edx
mov ebp,[ 220+esp]
mov ecx,edi
add eax,ebp
xor esi,edi
rol eax,3
rol esi,cl
add esi,eax

cmp esi,[ 236+esp]
jne __exit_2_k5

lea ecx,[eax+edx]
mov ebp,[ 224+esp]
add ebx,ecx
rol ebx,cl
add eax,ebx
mov ecx,esi
add eax,ebp
xor edi,esi
rol eax,3
rol edi,cl
add edi,eax

cmp edi,[ 240+esp]
jne __exit_2_k5
mov [ 16+esp],dword ptr 1
jmp _full_exit_k5

__exit_2_k5:

mov edx,[ 260+esp]


add edx,33554432
jc _next_inc_k5

_next_iter_k5:
mov [ 260+esp],edx
lea edi,[ 16777216+edx]
sub [ 268+esp], dword ptr 1
jg _loaded_k5
mov eax,[ 288+esp]	; pointer to rc5unitwork
mov ebx,[ 264+esp]
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)
jmp _full_exit_k5

_next_inc_k5:
add edx,65536
test edx,16711680
jnz _next_iter_k5

sub edx,16777216
add edx,256
test edx,-256
jnz _next_iter_k5

sub edx,65536
add edx,1
test edx,255
jnz _next_iter_k5

 	; we should never go here, it would mean we have iterated 2^32 times ...
 	; stop the client, something went wrong
;mov 0,0	; generate a segfault

_full_exit_k5:
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
_rc5_unit_func_k5 endp
_TEXT   ends
end

