.386p
.model flat
_TEXT   segment dword public use32 'CODE'
align 4
public _rc5_unit_func_p5
_rc5_unit_func_p5 proc near

sub esp,264
push ebp
push edi
push esi
push ebx
mov ebp,[ 288+esp]
mov [ 272+esp],ebp
mov [ 16+esp], dword ptr 0
mov eax,[ 284+esp]
 	;APP
mov [ 276+esp],ebp
mov ebx,[ 20+eax]	; ebx = l0 = Llo1
mov edx,[ 16+eax]	; edx = l1 = Lhi1
mov [ 24+esp],ebx
mov [ 20+esp],edx
mov ebp,[ 4+eax]
mov [ 40+esp],ebp
mov ebp,[ 0+eax]
mov [ 44+esp],ebp
mov ebp,[ 12+eax]
mov [ 256+esp],ebp
mov ebp,[ 8+eax]
mov [ 260+esp],ebp

_loaded_p5:
mov esi,[ 24+esp]	; 1
mov ebx,354637369
add esi,-1089828067	; 1   Spare slot (not that it matters here)  BRF
rol esi,29	; 1
add ebx,esi	; 1
mov ecx,esi
rol ebx,3	; 1
add ecx,ebx	; 1
mov [ 32+esp],ebx
mov [ 36+esp],esi	; 1
mov [ 28+esp],ecx	;  sum = 7 every 2147483648 loops or on subroutine
 	;        entry.  The latter happens more often.  BRF

_next_key:
mov edi,[ 20+esp]	; 1
mov ebx,[ 32+esp]
add edi,ecx
mov edx,edi	; 1
add edi,16777216
rol edi,cl	; 4
rol edx,cl	; 4
lea ebp,[-196066091+ebx+edi]	; 1
mov ecx,edi
rol ebp,3	; 1
lea ebx,[-196066091+ebx+edx]	; 1
add ecx,ebp
rol ebx,3	; 1
add esi,ecx	; 1
mov eax,[ 36+esp]	;  sum = 16
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 68+esp],ebp
add eax,ecx	; 1
lea ebp,[-1836597618+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 64+esp],ebx
add edi,ecx	; 1
lea ebx,[-1836597618+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 76+esp],ebp
add edx,ecx	; 1
lea ebp,[ 817838151+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 72+esp],ebx
add esi,ecx	; 1
lea ebx,[ 817838151+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 84+esp],ebp
add eax,ecx	; 1
lea ebp,[-822693376+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 80+esp],ebx
add edi,ecx	; 1
lea ebx,[-822693376+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 92+esp],ebp
add edx,ecx	; 1
lea ebp,[ 1831742393+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 88+esp],ebx
add esi,ecx	; 1
lea ebx,[ 1831742393+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 100+esp],ebp
add eax,ecx	; 1
lea ebp,[ 191210866+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 96+esp],ebx
add edi,ecx	; 1
lea ebx,[ 191210866+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 108+esp],ebp
add edx,ecx	; 1
lea ebp,[-1449320661+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 104+esp],ebx
add esi,ecx	; 1
lea ebx,[-1449320661+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 116+esp],ebp
add eax,ecx	; 1
lea ebp,[ 1205115108+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 112+esp],ebx
add edi,ecx	; 1
lea ebx,[ 1205115108+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 124+esp],ebp
add edx,ecx	; 1
lea ebp,[-435416419+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 120+esp],ebx
add esi,ecx	; 1
lea ebx,[-435416419+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 132+esp],ebp
add eax,ecx	; 1
lea ebp,[-2075947946+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 128+esp],ebx
add edi,ecx	; 1
lea ebx,[-2075947946+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 140+esp],ebp
add edx,ecx	; 1
lea ebp,[ 578487823+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 136+esp],ebx
add esi,ecx	; 1
lea ebx,[ 578487823+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 148+esp],ebp
add eax,ecx	; 1
lea ebp,[-1062043704+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 144+esp],ebx
add edi,ecx	; 1
lea ebx,[-1062043704+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 156+esp],ebp
add edx,ecx	; 1
lea ebp,[ 1592392065+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 152+esp],ebx
add esi,ecx	; 1
lea ebx,[ 1592392065+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 164+esp],ebp
add eax,ecx	; 1
lea ebp,[-48139462+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 160+esp],ebx
add edi,ecx	; 1
lea ebx,[-48139462+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 172+esp],ebp
add edx,ecx	; 1
lea ebp,[-1688670989+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 168+esp],ebx
add esi,ecx	; 1
lea ebx,[-1688670989+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 180+esp],ebp
add eax,ecx	; 1
lea ebp,[ 965764780+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 176+esp],ebx
add edi,ecx	; 1
lea ebx,[ 965764780+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 188+esp],ebp
add edx,ecx	; 1
lea ebp,[-674766747+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 184+esp],ebx
add esi,ecx	; 1
lea ebx,[-674766747+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 196+esp],ebp
add eax,ecx	; 1
lea ebp,[ 1979669022+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 192+esp],ebx
add edi,ecx	; 1
lea ebx,[ 1979669022+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 204+esp],ebp
add edx,ecx	; 1
lea ebp,[ 339137495+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 200+esp],ebx
add esi,ecx	; 1
lea ebx,[ 339137495+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 212+esp],ebp
add eax,ecx	; 1
lea ebp,[-1301394032+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 208+esp],ebx
add edi,ecx	; 1
lea ebx,[-1301394032+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 220+esp],ebp
add edx,ecx	; 1
lea ebp,[ 1353041737+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 216+esp],ebx
add esi,ecx	; 1
lea ebx,[ 1353041737+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 228+esp],ebp
add eax,ecx	; 1
lea ebp,[-287489790+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 224+esp],ebx
add edi,ecx	; 1
lea ebx,[-287489790+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 236+esp],ebp
add edx,ecx	; 1
lea ebp,[-1928021317+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 232+esp],ebx
add esi,ecx	; 1
lea ebx,[-1928021317+ebx+edx]
rol ebx,3	; 1  sum = 14
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
mov [ 244+esp],ebp
add eax,ecx	; 1
lea ebp,[ 726414452+ebp+esi]
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 240+esp],ebx
add edi,ecx	; 1
lea ebx,[ 726414452+ebx+eax]
rol ebx,3	; 1  sum = 14
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
mov [ 252+esp],ebp
add edx,ecx	; 1
lea ebp,[-1089828067+ebp+edi]
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 248+esp],ebx
lea ebx,[-1089828067+ebx+edx]	; 1
add esi,ecx
rol ebx,3	; 1
rol esi,cl	; 4
mov [ 52+esp],ebp	; 1
mov ecx,[ 32+esp]
add ebp,ecx	; 1
lea ecx,[ebx+edx]
add ebp,esi	; 1
mov [ 48+esp],ebx
add eax,ecx	; 1   Spare slot
 	;   sum = 22
_end_round1_p5:
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 60+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 32+esp]	; 2
add ebp,[ 68+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 56+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 68+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 64+esp]	; 2
add ebp,[ 76+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 64+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 76+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 72+esp]	; 2
add ebp,[ 84+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 72+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 84+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 80+esp]	; 2
add ebp,[ 92+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 80+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 92+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 88+esp]	; 2
add ebp,[ 100+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 88+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 100+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 96+esp]	; 2
add ebp,[ 108+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 96+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 108+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 104+esp]	; 2
add ebp,[ 116+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 104+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 116+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 112+esp]	; 2
add ebp,[ 124+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 112+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 124+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 120+esp]	; 2
add ebp,[ 132+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 120+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 132+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 128+esp]	; 2
add ebp,[ 140+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 128+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 140+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 136+esp]	; 2
add ebp,[ 148+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 136+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 148+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 144+esp]	; 2
add ebp,[ 156+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 144+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 156+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 152+esp]	; 2
add ebp,[ 164+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 152+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 164+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 160+esp]	; 2
add ebp,[ 172+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 160+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 172+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 168+esp]	; 2
add ebp,[ 180+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 168+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 180+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 176+esp]	; 2
add ebp,[ 188+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 176+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 188+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 184+esp]	; 2
add ebp,[ 196+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 184+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 196+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 192+esp]	; 2
add ebp,[ 204+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 192+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 204+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 200+esp]	; 2
add ebp,[ 212+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 200+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 212+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 208+esp]	; 2
add ebp,[ 220+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 208+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 220+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 216+esp]	; 2
add ebp,[ 228+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 216+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 228+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 224+esp]	; 2
add ebp,[ 236+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 224+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 236+esp],ebp
add ebx,eax	; 1
add edi,ecx
add ebx,[ 232+esp]	; 2
add ebp,[ 244+esp]
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
add ebp,edi
mov [ 232+esp],ebx	; 1
add edx,ecx	;   sum = 16
rol ebp,3	; 1
rol edx,cl	; 4
lea ecx,[ebp+edi]	; 1
mov [ 244+esp],ebp
add ebx,edx	; 1
add esi,ecx
add ebx,[ 240+esp]	; 2
add ebp,[ 252+esp]
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add ebp,esi
mov [ 240+esp],ebx	; 1
add eax,ecx	;   sum = 16
rol ebp,3	; 1
rol eax,cl	; 4
lea ecx,[ebp+esi]	; 1
mov [ 252+esp],ebp
add ebx,eax	; 1
mov ebp,[ 248+esp]
add ebx,ebp	; 1
add edi,ecx
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[ebx+eax]	; 1
mov [ 264+esp],edi
mov [ 248+esp],ebx	; 1
add edx,ecx
rol edx,cl	; 4
mov [ 268+esp],esi	; 1
add ebx,edx	;   sum = 20

_end_round2_p5:
mov ebp,[ 48+esp]	; 1
mov esi,[ 40+esp]	;   eA = P_0 + A;
add ebp,ebx	; 1
mov ebx,[ 56+esp]
rol ebp,3	; 1
add esi,ebp	; 1
add ebx,ebp
lea ecx,[ebp+edx]	; 1   L0 = ROTL(L0 + A + L1, A + L1);
mov edi,[ 44+esp]	;   eB = P_1 + A;
add eax,ecx	; 1
 	;  Spare slot
rol eax,cl	; 4

add ebx,eax	; 1 A = ROTL3(S00 + A + L1);
mov ecx,eax	;   A = ROTL3(S03 + A + L0);
rol ebx,3	; 1
add edi,ebx	; 1
add ecx,ebx
add edx,ecx	; 1
mov ebp,[ 64+esp]
rol edx,cl	; 4 sum = 18
_round3_p5_S1_2:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 72+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 80+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_4:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 88+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 96+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_6:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 104+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 112+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_8:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 120+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 128+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_10:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 136+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 144+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_12:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 152+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 160+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_14:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 168+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 176+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_16:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 184+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 192+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_18:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 200+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 208+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_20:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 216+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 224+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S1_22:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 232+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 240+esp]
rol edx,cl	; 4   sum = 26
_end_round3_1_p5:
add ebx,edx	; 1 A = ROTL3(S24 + A + L1);
mov ecx,edi	;    eA = ROTL(eA ^ eB, eB) + A
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
add esi,ebx	; 1
mov ebp,[ 256+esp]	;     Places je in V pipe for pairing.  BRF

cmp esi,ebp	; 1
je _testC1_1_p5	;  sum = 9

_second_key:
mov edx,[ 264+esp]	; 1
mov ebx,[ 252+esp]
mov eax,[ 268+esp]	; 1
add ebx,edx
mov ebp,[ 52+esp]	; 1
mov esi,[ 40+esp]	;   eA = P_0 + A;
add ebp,ebx	; 1
mov ebx,[ 60+esp]
rol ebp,3	; 1
add esi,ebp	; 1
add ebx,ebp
lea ecx,[ebp+edx]	; 1   L0 = ROTL(L0 + A + L1, A + L1);
mov edi,[ 44+esp]	;   eB = P_1 + A;
add eax,ecx	; 1
 	;  Spare slot
rol eax,cl	; 4

add ebx,eax	; 1 A = ROTL3(S00 + A + L1);
mov ecx,eax	;   A = ROTL3(S03 + A + L0);
rol ebx,3	; 1
add edi,ebx	; 1
add ecx,ebx
add edx,ecx	; 1
mov ebp,[ 68+esp]
rol edx,cl	; 4 sum = 20
_round3_p5_S2_2:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 76+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 84+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_4:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 92+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 100+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_6:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 108+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 116+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_8:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 124+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 132+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_10:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 140+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 148+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_12:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 156+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 164+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_14:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 172+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 180+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_16:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 188+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 196+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_18:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 204+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 212+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_20:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 220+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 228+esp]
rol edx,cl	; 4   sum = 26
_round3_p5_S2_22:
add ebx,edx	; 1
mov ecx,edi
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
lea ecx,[ebx+edx]	; 1
add esi,ebx
add eax,ecx	; 1
mov ebp,[ 236+esp]
rol eax,cl	; 4

add ebx,eax	; 1
mov ecx,esi
add ebx,ebp	; 1
xor edi,esi
rol ebx,3	; 1
rol edi,cl	; 4
lea ecx,[eax+ebx]	; 1
add edi,ebx
add edx,ecx	; 1
mov ebp,[ 244+esp]
rol edx,cl	; 4   sum = 26
_end_round3_2_p5:
add ebx,edx	; 1 A = ROTL3(S24 + A + L1);
mov ecx,edi	; eA = ROTL(eA ^ eB, eB) + A
add ebx,ebp	; 1
xor esi,edi
rol ebx,3	; 1
rol esi,cl	; 4
add esi,ebx	; 1
mov ebp,[ 256+esp]	;    Places je in V pipe for pairing.  BRF

cmp esi,ebp	; 1
je _testC1_2_p5	;  sum = 9

_incr_key:
sub [ 272+esp], dword ptr 1	; 3
jz _full_exit_p5

mov dl, [ 23+esp]	; 1  All this is to try and save one clock
 	;    at the jnc below
mov ecx,[ 28+esp]	;    Costs nothing (in clocks) to try.  BRF
add dl,byte ptr 2	; 1
mov esi,[ 36+esp]
mov [ 23+esp],dl	; 1
jnc _next_key

inc byte ptr [ 22+esp]
jnz _next_key
inc byte ptr [ 21+esp]
jnz _next_key
inc byte ptr [ 20+esp]
jnz _next_key
inc byte ptr [ 27+esp]
jnz _loaded_p5
inc byte ptr [ 26+esp]
jnz _loaded_p5
inc byte ptr [ 25+esp]
jnz _loaded_p5
inc byte ptr [ 24+esp]
jmp _loaded_p5	; Wrap the keyspace

_testC1_1_p5:
lea ecx,[ebx+edx]	; 1 L0 = ROTL(L0 + A + L1, A + L1);
mov ebp,[ 248+esp]
add eax,ecx	; 1
xor edi,esi
rol eax,cl	; 4
add ebx,eax	; 1 A = ROTL3(S25 + A + L0);
mov ecx,esi	;    eB = ROTL(eB ^ eA, eA) + A
add ebx,ebp	; 1
 	;  Spare slot (not that it matters)  BRF
rol ebx,3	; 1
rol edi,cl	; 4
add edi,ebx	; 1
mov ebp,[ 260+esp]	;     Places jne in V pipe for pairing.  BRF

cmp edi,ebp	; 1
jne _second_key
jmp _done

_testC1_2_p5:
lea ecx,[ebx+edx]	; 1 L0 = ROTL(L0 + A + L1, A + L1);
mov ebp,[ 252+esp]
add eax,ecx	; 1
xor edi,esi
rol eax,cl	; 4
add ebx,eax	; 1 A = ROTL3(S25 + A + L0);
mov ecx,esi	;    eB = ROTL(eB ^ eA, eA) + A
add ebx,ebp	; 1
 	;  Spare slot (not that it matters)  BRF
rol ebx,3	; 1
rol edi,cl	; 4
add edi,ebx	; 1
mov ebp,[ 260+esp]	;     Places jne in V pipe for pairing.  BRF

cmp edi,ebp	; 1
jne _incr_key
mov [ 16+esp],dword ptr 1
jmp _done

_full_exit_p5:
add [ 23+esp],byte ptr 2
jnc _key_updated
inc byte ptr [ 22+esp]
jnz _key_updated
inc byte ptr [ 21+esp]
jnz _key_updated
inc byte ptr [ 20+esp]
jnz _key_updated
inc byte ptr [ 27+esp]
jnz _key_updated
inc byte ptr [ 26+esp]
jnz _key_updated
inc byte ptr [ 25+esp]
jnz _key_updated
inc byte ptr [ 24+esp]

_key_updated:
mov eax,[ 284+esp]	; pointer to rc5unitwork
mov ebx,[ 24+esp]
mov edx,[ 20+esp]
mov [ 20+eax],ebx	; Update real data
mov [ 16+eax],edx	; (used by caller)

_done:
mov ebp,[ 276+esp]

 	;NO_APP
mov edx,ebp
sub edx,[ 272+esp]
mov eax,[ 16+esp]
lea edx,[ 2*edx+eax]
mov eax,edx
pop ebx
pop esi
pop edi
pop ebp
add esp,264
ret
_rc5_unit_func_p5 endp
_TEXT   ends
end

