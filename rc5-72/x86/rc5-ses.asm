; RC5-72 Assembly version - 1 pipe
; Samuel Seay - Samuel@Lightspeed.cx - 2002/10/17
;
; For use by distributed.net. I do request that any
; changes to this core are also emailed to Samuel@Lightspeed.cx

[SECTION .text]

;export the global function
[GLOBAL _rc5_72_unit_func_ses]
[GLOBAL rc5_72_unit_func_ses]

;define P and Q
%define P	0xB7E15163
%define Q	0x9E3779B9

;define an entry for S's value
%define SVal(x)	(P+(Q*x))

;defines of where things are in the stack
%define ebx_save	ebp+0		;4 bytes
%define ebp_save	esp+4		;4 bytes
%define edi_save	ebp+8		;4 bytes
%define counter		ebp+12		;4 bytes
%define itersleft       ebp+16          ; 4 bytes
%define LBox		ebp+20		;3 entries, 12 bytes
%define SBox		ebp+32		;26 entries, 104 bytes
%define stacksize	136		;make sure to modify if added values

;variables on the stack we want
%define scratch         [ebp+stacksize+12] ;not actually used by us.
%define keystodo	[ebp+stacksize+8]
%define workunit	[ebp+stacksize+4]

;define the workunit entries
%define workunitplain(x)	[edi+((x)*4)]		;0=hi, 1=low
%define workunitcypher(x)	[edi+8+((x)*4)]		;0=hi, 1=low
%define workunitL0(x)		[edi+16+((x)*4)]	;0=hi, 1=mid, 2=low
%define workunitL0Byte(x,y)	[edi+16+(((x)*4)+y)]
%define workunitcount		[edi+28]		;16bit, not 32bit
%define workunitcheck(x)	[edi+32+((x)*4)]	;0=hi, 1=mid, 2=low

;define S() and L()
%define S(x)	[SBox + ((x)*4)]
%define L(x)	[LBox + ((x)*4)]

;A = S[i] = ROTL(SVal[i]+(A+B),3);
;B = L[i] = ROTL(rc5_72workunit->L0(j)+(A+B),(A+B));
%macro StartROTLLoop1 2
	mov edx, workunitL0(%2)
	lea eax, [SVal(%1)+eax+ebx]
	rol eax, 3
	mov S(%1), eax

	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L(%1), ebx
%endmacro

;A = S[i] = ROTL(SVal[i]+(A+B),3);
;B = L[j] = ROTL(L[j]+(A+B),(A+B));
%macro ROTLLoop1 2
	mov edx, L(%2)
	lea eax, [SVal(%1)+eax+ebx]
	rol eax, 3
	mov S(%1), eax

	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L(%2), ebx
%endmacro

;A = S[i] = ROTL(S[i]+(A+B),3);
;B = L[j] = ROTL(L[j]+(A+B),(A+B));
%macro ROTLLoop 2
	mov edx, L(%2)
	add eax, S(%1)
	add eax, ebx
	rol eax, 3
	mov S(%1), eax

	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L(%2), ebx
%endmacro

;A = ROTL(A^B,B)+S[2*i];
;B = ROTL(B^A,A)+S[2*i+1];
%macro ROTLLoop12 1
	;A = ROTL(A^B,B)+S[2*i];
	mov ecx, ebx
	xor eax, ebx
	rol eax, cl
	add eax, dword S(2 * %1)

	;B = ROTL(B^A,A)+S[2*i+1];
	mov ecx, eax
	xor ebx, ecx
	rol ebx, cl
	add ebx,  dword S((2 * %1) + 1)
%endmacro




align 16

;start the function
_rc5_72_unit_func_ses:
rc5_72_unit_func_ses:
	;get some stack space
	sub esp, stacksize

	;save ebp
	mov [ebp_save], ebp

	;change it
	mov ebp, esp

	;save ebx and edi
	mov [ebx_save], ebx
	mov [edi_save], edi
        mov dword [counter], 0
	
	;get the structure
	mov edi, workunit

	;find the number of iters to run (for 1-pipe, 1 key==1 iter)
	mov ebx,keystodo
	mov eax,dword [ebx]
	inc eax			; add 1, since we start with a dec.
	mov [itersleft],eax

timesliceloop:
	;decrement the amount of iterations, loop if not at the end
	dec dword [itersleft]
	jz near foundnothing

	;do our loops

	;A=0, B=0;
	;A = S[i] = ROTL(SVal[i]+(A+B),3);
	;B = L[i] = ROTL(rc5_72workunit->L0(j)+(A+B),(A+B));
	mov ebx, workunitL0(2)
	mov eax, 0xBF0A8B1D	;rol SVal(0), 3

	add ebx, eax
	rol ebx, 0x1D
	mov L(0), ebx

	StartROTLLoop1 1, 1
	StartROTLLoop1 2, 0

	;finish off the first loop
	%assign i 3
	%assign j 0
	%rep 23
		ROTLLoop1 i, j

		%assign i i+1
		%assign j j+1

		%if j==3
			%assign j 0
		%endif
	%endrep


	;A = S[0] = ROTL(S[0]+(A+B),3);
	;B = L[2] = ROTL(L[2]+(A+B),(A+B));
	mov edx, L(2)
	lea eax, [eax+ebx+0xBF0A8B1D]	;S[0] is always 0xBF0A8B1D
	rol eax, 3
	mov S(0), eax
	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L(2), ebx

	;do the other 2 loops
	%assign i 1
	%assign j 0
	%rep 51
		ROTLLoop i, j

		%assign i i+1
		%assign j j+1

		%if j==3
			%assign j 0
		%endif

		%if i==26
			%assign i 0
		%endif
	%endrep


	;A = unitwork->plain.lo + S(0)
	;B = unitwork->plain.hi + S(1)
	mov eax, workunitplain(1)
	mov ebx, workunitplain(0)
	add eax, S(0)
	add ebx, S(1)

	%assign i 1
	%rep 12
		ROTLLoop12 i
		%assign i i+1
	%endrep

	;if (A == rc5_72unitwork->cypher.lo)
	cmp eax, workunitcypher(1)
	jne continuechecks

	inc dword workunitcount

	;copy the values to the check entries
	mov eax, workunitL0(0)
	mov ecx, workunitL0(1)
	mov edx, workunitL0(2)

	mov workunitcheck(0), eax
	mov workunitcheck(1), ecx
	mov workunitcheck(2), edx

	;if (B == rc5_72unitwork->cypher.hi) return;
	cmp ebx, workunitcypher(0)
	je near foundsuccess

continuechecks:
	;now a massive if statement
        inc dword [counter]
	
	;key.hi = (key.hi + 0x01) & 0x000000FF;
	;if (!key.hi)
	inc byte workunitL0Byte(0,0)
	jnz near timesliceloop

	;key.mid = key.mid + 0x01000000;
	;if (!(key.mid & 0xFF000000u))
	inc byte workunitL0Byte(1,3)
	jnz near timesliceloop

	;key.mid = (key.mid + 0x00010000) & 0x00FFFFFF;
	;if (!(key.mid & 0x00FF0000))
	inc byte workunitL0Byte(1,2)
	jnz near timesliceloop

	;key.mid = (key.mid + 0x00000100) & 0x0000FFFF;
	;if (!(key.mid & 0x0000FF00))
	inc byte workunitL0Byte(1,1)
	jnz near timesliceloop

	;key.mid = (key.mid + 0x00000001) & 0x000000FF;
	;if (!(key.mid & 0x000000FF))
	inc byte workunitL0Byte(1,0)
	jnz near timesliceloop

	;key.lo = key.lo + 0x01000000;
	;if (!(key.lo & 0xFF000000u))
	inc byte workunitL0Byte(2,3)
	jnz near timesliceloop

	;key.lo = (key.lo + 0x00010000) & 0x00FFFFFF;
	;if (!(key.lo & 0x00FF0000))
	inc byte workunitL0Byte(2,2)
	jnz near timesliceloop

	;key.lo = (key.lo + 0x00000100) & 0x0000FFFF;
	;if (!(key.lo & 0x0000FF00))
	inc byte workunitL0Byte(2,1)
	jnz near timesliceloop

	;key.lo = (key.lo + 0x00000001) & 0x000000FF;
	inc byte workunitL0Byte(2,0)
	jmp timesliceloop


foundsuccess:
	mov eax,[counter]
	mov ebx,keystodo
	mov dword [ebx],eax       ; pass back how many keys done
	
	mov eax,2		; return code = RESULT_FOUND
	jmp short endloop

foundnothing:
	mov eax,1       	; return code = RESULT_NOTHING

endloop:
	;restore ebx and edi
	mov ebx, [ebx_save]
	mov edi, [edi_save]

	;restore ebp
	mov ebp, [ebp_save]

	;remove our stack
	add esp, stacksize

	;return
	ret