; RC5-72 Assembly version - 2 pipe
; Samuel Seay - Samuel@Lightspeed.cx - 2002/10/19
;
; For use by distributed.net. I do request that any
; changes to this core are also emailed to Samuel@Lightspeed.cx

; $Id: rc5-ses-2.asm,v 1.3 2002/10/20 20:52:10 andreasb Exp $

%ifdef __OMF__ ; Borland and Watcom compilers/linkers
[SECTION _TEXT FLAT USE32 align=16 CLASS=CODE]
%else
[SECTION .text]
%endif

;export the global function
[GLOBAL _rc5_72_unit_func_ses_2]
[GLOBAL rc5_72_unit_func_ses_2]

;define P and Q
%define P	0xB7E15163
%define Q	0x9E3779B9

;define an entry for S's value
%define SVal(x)	(P+(Q*x))

;defines of where things are in the stack
%define ebx_save	esp+0		;4 bytes
%define ebp_save	esp+4		;4 bytes
%define edi_save	esp+8		;4 bytes
%define esi_save	esp+12		;4 bytes
%define counter		esp+16		;4 bytes
%define itersleft	esp+20		;4 bytes
%define LBox1		esp+24		;3 entries, 12 bytes
%define SBox1		esp+36		;26 entries, 104 bytes
%define LBox2		esp+140		;3 entries, 12 bytes
%define SBox2		esp+152		;26 entries, 104 bytes
%define stacksize	256		;make sure to modify if added values

;variables on the stack we want
%define scratchspace	[esp+stacksize+12]
%define keystodo	[esp+stacksize+8]
%define workunit	[esp+stacksize+4]

;define the workunit entries
%define workunitplain(x)	[ebp+((x)*4)]		;0=hi, 1=low
%define workunitcypher(x)	[ebp+8+((x)*4)]		;0=hi, 1=low
%define workunitL0(x)		[ebp+16+((x)*4)]	;0=hi, 1=mid, 2=low
%define workunitL0Byte(x,y)	[ebp+16+(((x)*4)+y)]
%define workunitcount		[ebp+28]		;16bit, not 32bit
%define workunitcheck(x)	[ebp+32+((x)*4)]	;0=hi, 1=mid, 2=low

;define S() and L()
%define S1(x)	[SBox1 + ((x)*4)]
%define L1(x)	[LBox1 + ((x)*4)]
%define S2(x)	[SBox2 + ((x)*4)]
%define L2(x)	[LBox2 + ((x)*4)]

;eax = A1   ebx = B1
;edi = A2   esi = B2


;A1 = S1[i] = ROTL(SVal[i]+(A1+B1),3);
;B1 = L1[i] = ROTL(rc5_72workunit->L0(j)+(A1+B1),(A1+B1));
;A2 = S2[i] = ROTL(SVal[i]+(A2+B2),3);
;B2 = L2[i] = ROTL(rc5_72workunit->L0(j)+(A2+B2),(A2+B2));
%macro StartROTLLoop1 3
	mov edx, workunitL0(%2)
	lea eax, [SVal(%1)+eax+ebx]
	rol eax, 3
	mov S1(%1), eax
	mov S2(%1), eax
	mov edi, eax

	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L1(%1), ebx

	lea esi, [edx+ecx+%3]
	rol esi, cl
	mov L2(%1), esi
%endmacro

;A1 = S1[i] = ROTL(SVal[i]+(1A+B1),3);
;B1 = L1[j] = ROTL(L1[j]+(A1+B1),(A1+B1));
;A2 = S2[i] = ROTL(SVal[i]+(A2+B2),3);
;B2 = L2[j] = ROTL(L2[j]+(A2+B2),(A2+B2));
%macro ROTLLoop1 2
	lea eax, [SVal(%1)+eax+ebx]
	rol eax, 3
	mov S1(%1), eax

	mov edx, L1(%2)
	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L1(%2), ebx


	lea edi, [SVal(%1)+edi+esi]
	rol edi, 3
	mov S2(%1), edi

	mov edx, L2(%2)
	lea ecx, [edi+esi]
	lea esi, [edx+ecx]
	rol esi, cl
	mov L2(%2), esi
%endmacro

;A1 = S1[i] = ROTL(S1[i]+(A1+B1),3);
;B1 = L1[j] = ROTL(L1[j]+(A1+B1),(A1+B1));
;A2 = S2[i] = ROTL(S2[i]+(A2+B2),3);
;B2 = L2[j] = ROTL(L2[j]+(A2+B2),(A2+B2));
%macro ROTLLoop 2
	mov edx, L1(%2)
	add eax, S1(%1)
	add eax, ebx
	rol eax, 3
	mov S1(%1), eax

	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L1(%2), ebx


	mov edx, L2(%2)
	add edi, S2(%1)
	add edi, esi
	rol edi, 3
	mov S2(%1), edi

	lea ecx, [edi+esi]
	lea esi, [edx+ecx]
	rol esi, cl
	mov L2(%2), esi
%endmacro

;A1 = ROTL(A1^B1,B1)+S1[2*i];
;B1 = ROTL(B1^A1,A1)+S1[2*i+1];
;A2 = ROTL(A2^B2,B2)+S2[2*i];
;B2 = ROTL(B2^A2,A2)+S2[2*i+1];
%macro ROTLLoop12 1
	;A1 = ROTL(A1^B1,B1)+S1[2*i];
	mov ecx, ebx
	xor eax, ebx
	rol eax, cl
	add eax, dword S1(2 * %1)

	;B1 = ROTL(B1^A1,A1)+S1[2*i+1];
	mov ecx, eax
	xor ebx, ecx
	rol ebx, cl
	add ebx, dword S1((2 * %1) + 1)


	;A2 = ROTL(A2^B2,B2)+S2[2*i];
	mov ecx, esi
	xor edi, esi
	rol edi, cl
	add edi, dword S2(2 * %1)

	;B2 = ROTL(B2^A2,A2)+S2[2*i+1];
	mov ecx, edi
	xor esi, ecx
	rol esi, cl
	add esi, dword S2((2 * %1) + 1)
%endmacro




align 16

;start the function
_rc5_72_unit_func_ses_2:
rc5_72_unit_func_ses_2:
	;get some stack space
	sub esp, stacksize

	;save ebp
	mov [ebp_save], ebp

	;save ebx and edi
	mov [ebx_save], ebx
	mov [edi_save], edi
	mov [esi_save], esi
	mov [counter], dword 0

	;get the structure
	mov ebp, workunit

	;compute how many iterations to do.
	mov ebx,keystodo
	mov eax,dword [ebx]
	shr eax,1    		; divide keycount by 2, since 2 pipes
	inc eax			; add 1 since we start with a dec.
	mov dword [itersleft],eax

timesliceloop:
	;decrement the amount of time, loop if not at the end
	dec dword [itersleft]
	jz near foundnothing

	;do our loops

	;A=0, B=0;
	;A1 = S1[i] = ROTL(SVal[i]+(A1+B1),3);
	;B1 = L1[i] = ROTL(rc5_72workunit->L0(j)+(A1+B1),(A1+B1));
	;A2 = S2[i] = ROTL(SVal[i]+(A2+B2),3);
	;B2 = L2[i] = ROTL(rc5_72workunit->L0(j)+(A2+B2),(A2+B2));
	mov ebx, workunitL0(2)
	mov eax, 0xBF0A8B1D	;rol SVal(0), 3
	mov edi, eax

	add ebx, eax
	rol ebx, 0x1D
	mov L1(0), ebx
	mov L2(0), ebx
	mov esi, ebx

	StartROTLLoop1 1, 1, 0
	StartROTLLoop1 2, 0, 1

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


	;A1 = S1[0] = ROTL(S1[0]+(A1+B1),3);
	;B1 = L1[2] = ROTL(L1[2]+(A1+B1),(A1+B1));
	;A2 = S2[0] = ROTL(S2[0]+(A2+B2),3);
	;B2 = L2[2] = ROTL(L2[2]+(A2+B2),(A2+B2));
	mov edx, L1(2)
	lea eax, [eax+ebx+0xBF0A8B1D]	;S[0] is always 0xBF0A8B1D
	rol eax, 3
	mov S1(0), eax
	lea ecx, [eax+ebx]
	lea ebx, [edx+ecx]
	rol ebx, cl
	mov L1(2), ebx

	mov edx, L2(2)
	lea edi, [edi+esi+0xBF0A8B1D]	;S[0] is always 0xBF0A8B1D
	rol edi, 3
	mov S2(0), edi
	lea ecx, [edi+esi]
	lea esi, [edx+ecx]
	rol esi, cl
	mov L2(2), esi

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


	;A1 = unitwork->plain.lo + S1(0)
	;B1 = unitwork->plain.hi + S1(1)
	;A2 = unitwork->plain.lo + S2(0)
	;B2 = unitwork->plain.hi + S2(1)
	mov eax, workunitplain(1)
	mov edi, eax
	mov ebx, workunitplain(0)
	mov esi, ebx
	add eax, S1(0)
	add ebx, S1(1)
	add edi, S2(0)
	add esi, S2(1)

	%assign i 1
	%rep 12
		ROTLLoop12 i
		%assign i i+1
	%endrep


	;if (A1 == rc5_72unitwork->cypher.lo)
	cmp eax, workunitcypher(1)
	jne near continuechecks1

	inc dword workunitcount

	;copy the values to the check entries
	mov eax, workunitL0(0)
	mov ecx, workunitL0(1)
	mov edx, workunitL0(2)

	mov workunitcheck(0), eax
	mov workunitcheck(1), ecx
	mov workunitcheck(2), edx


	;if (B1 == rc5_72unitwork->cypher.hi) return;
	cmp ebx, workunitcypher(0)
	je near foundsuccess


continuechecks1:
	;if (A2	 == rc5_72unitwork->cypher.lo)
	cmp edi, workunitcypher(1)
	jne near continuechecks2

	inc dword workunitcount

	;copy the values to the check entries
	mov eax, workunitL0(0)
	inc eax
	mov ecx, workunitL0(1)
	mov edx, workunitL0(2)

	mov workunitcheck(0), eax
	mov workunitcheck(1), ecx
	mov workunitcheck(2), edx

	;if (B2 == rc5_72unitwork->cypher.hi) return;
	cmp esi, workunitcypher(0)
	jne near continuechecks2

	inc dword [counter]
	jmp near foundsuccess

continuechecks2:
	;now a massive if statement
	add dword [counter], 2

	;key.hi = (key.hi + 0x02) & 0x000000FF;
	;if (!key.hi)
	add byte workunitL0Byte(0,0), 2
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
	mov eax,dword [counter]
	mov ebx,keystodo
	mov dword [ebx],eax	;  pass back how many keys done

        mov eax,2	;  return code = RESULT_FOUND
        jmp short endloop

foundnothing:
	mov eax,1		;  return code = RESULT_NOTHING
	
endloop:
	;restore ebx and edi
	mov ebx, [ebx_save]
	mov edi, [edi_save]
	mov esi, [esi_save]

	;restore ebp
	mov ebp, [ebp_save]

	;remove our stack
	add esp, stacksize

	;return
	ret
