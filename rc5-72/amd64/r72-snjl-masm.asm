; line 1+1 r72-snjl.asm











text	SEGMENT

public rc5_72_unit_func_snjl

; line 21+1 r72-snjl.asm

; line 24+1 r72-snjl.asm



; line 30+1 r72-snjl.asm

; line 35+1 r72-snjl.asm


; line 61+1 r72-snjl.asm


; line 74+1 r72-snjl.asm


; line 99+1 r72-snjl.asm


; line 152+1 r72-snjl.asm

; line 187+1 r72-snjl.asm


align 16
rc5_72_unit_func_snjl proc near

 sub rsp, 452


 mov [rsp+436],rcx
 mov [rsp+444],rdx
 mov rax, rcx


 mov [rsp+388], rsi
 mov [rsp+396], rdi
; line 211+1 r72-snjl.asm


 mov [rsp+380], rbp
 mov [rsp+372], rbx
 mov [rsp+404], r12
 mov [rsp+412], r13
 mov [rsp+420], r14
 mov [rsp+428], r15

 mov edx, [rax+4]
 mov edi, [rax+0]

 mov ebx, [rax+12]
 mov ecx, [rax+8]

 mov [rsp+352], edx
 mov [rsp+356], edi
 mov edx, [rax+16]

 mov [rsp+360], ebx
 mov [rsp+364], ecx
 mov edi, [rsi]

 imul edi, 2863311531
 mov ecx, [rax+20]
 mov ebx, [rax+24]

 mov [rsp+368], edi
 mov r10d, edx
 mov r9d, ecx
 mov r8d, ebx

 mov [rsp+316+((2)*4)], edx
 mov [rsp+316+((1)*4)], ecx
 mov [rsp+316+((0)*4)], ebx

 add dl, 1
 bswap ecx
 bswap ebx

 adc ecx, 0
 adc ebx, 0

 bswap ecx
 bswap ebx

 mov r13d, edx
 mov r12d, ecx
 mov r11d, ebx

 mov [rsp+328+((2)*4)], edx
 mov [rsp+328+((1)*4)], ecx
 mov [rsp+328+((0)*4)], ebx

 add dl, 1
 bswap ecx
 bswap ebx

 adc ecx, 0
 adc ebx, 0

 bswap ecx
 bswap ebx

 mov [rsp+0], edx
 mov r15d, ecx
 mov r14d, ebx

 mov [rsp+340+((2)*4)], edx
 mov [rsp+340+((1)*4)], ecx
 mov [rsp+340+((0)*4)], ebx


align 16
key_setup_1:
 mov esi, r8d
 mov edi, r11d
 mov ebp, r14d

 mov eax, 0BF0A8B1Dh
 mov ebx, eax
 mov edx, eax

 mov [rsp+4+((0)*4)], eax
 mov [rsp+108+((0)*4)], eax
 mov [rsp+212+((0)*4)], eax

 add esi, eax
 add edi, eax
 add ebp, eax

 rol esi, 01Dh
 rol edi, 01Dh
 rol ebp, 01Dh

 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(1))]
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(1))]
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(1))]

 mov r8d, esi
 mov r11d, edi
 mov r14d, ebp

 rol eax, 3
; line 314+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((1)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(1+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((1)*4)], ebx
 mov [rsp+212+((1)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(1+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(1+1))]

 mov r12d, edi
 mov r15d, ebp
; line 315+1 r72-snjl.asm
 rol eax, 3
; line 315+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((2)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(2+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((2)*4)], ebx
 mov [rsp+212+((2)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(2+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(2+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 316+1 r72-snjl.asm
 rol eax, 3
; line 316+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((3)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(3+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((3)*4)], ebx
 mov [rsp+212+((3)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(3+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(3+1))]

 mov r11d, edi
 mov r14d, ebp
; line 317+1 r72-snjl.asm
 rol eax, 3
; line 317+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((4)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(4+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((4)*4)], ebx
 mov [rsp+212+((4)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(4+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(4+1))]

 mov r12d, edi
 mov r15d, ebp
; line 318+1 r72-snjl.asm
 rol eax, 3
; line 318+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((5)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(5+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((5)*4)], ebx
 mov [rsp+212+((5)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(5+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(5+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 319+1 r72-snjl.asm
 rol eax, 3
; line 319+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((6)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(6+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((6)*4)], ebx
 mov [rsp+212+((6)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(6+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(6+1))]

 mov r11d, edi
 mov r14d, ebp
; line 320+1 r72-snjl.asm
 rol eax, 3
; line 320+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((7)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(7+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((7)*4)], ebx
 mov [rsp+212+((7)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(7+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(7+1))]

 mov r12d, edi
 mov r15d, ebp
; line 321+1 r72-snjl.asm
 rol eax, 3
; line 321+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((8)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(8+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((8)*4)], ebx
 mov [rsp+212+((8)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(8+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(8+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 322+1 r72-snjl.asm
 rol eax, 3
; line 322+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((9)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(9+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((9)*4)], ebx
 mov [rsp+212+((9)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(9+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(9+1))]

 mov r11d, edi
 mov r14d, ebp
; line 323+1 r72-snjl.asm
 rol eax, 3
; line 323+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((10)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(10+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((10)*4)], ebx
 mov [rsp+212+((10)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(10+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(10+1))]

 mov r12d, edi
 mov r15d, ebp
; line 324+1 r72-snjl.asm
 rol eax, 3
; line 324+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((11)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(11+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((11)*4)], ebx
 mov [rsp+212+((11)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(11+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(11+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 325+1 r72-snjl.asm
 rol eax, 3
; line 325+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((12)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(12+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((12)*4)], ebx
 mov [rsp+212+((12)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(12+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(12+1))]

 mov r11d, edi
 mov r14d, ebp
; line 326+1 r72-snjl.asm
 rol eax, 3
; line 326+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((13)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(13+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((13)*4)], ebx
 mov [rsp+212+((13)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(13+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(13+1))]

 mov r12d, edi
 mov r15d, ebp
; line 327+1 r72-snjl.asm
 rol eax, 3
; line 327+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((14)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(14+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((14)*4)], ebx
 mov [rsp+212+((14)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(14+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(14+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 328+1 r72-snjl.asm
 rol eax, 3
; line 328+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((15)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(15+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((15)*4)], ebx
 mov [rsp+212+((15)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(15+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(15+1))]

 mov r11d, edi
 mov r14d, ebp
; line 329+1 r72-snjl.asm
 rol eax, 3
; line 329+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((16)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(16+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((16)*4)], ebx
 mov [rsp+212+((16)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(16+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(16+1))]

 mov r12d, edi
 mov r15d, ebp
; line 330+1 r72-snjl.asm
 rol eax, 3
; line 330+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((17)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(17+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((17)*4)], ebx
 mov [rsp+212+((17)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(17+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(17+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 331+1 r72-snjl.asm
 rol eax, 3
; line 331+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((18)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(18+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((18)*4)], ebx
 mov [rsp+212+((18)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(18+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(18+1))]

 mov r11d, edi
 mov r14d, ebp
; line 332+1 r72-snjl.asm
 rol eax, 3
; line 332+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((19)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(19+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((19)*4)], ebx
 mov [rsp+212+((19)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(19+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(19+1))]

 mov r12d, edi
 mov r15d, ebp
; line 333+1 r72-snjl.asm
 rol eax, 3
; line 333+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((20)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(20+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((20)*4)], ebx
 mov [rsp+212+((20)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(20+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(20+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 334+1 r72-snjl.asm
 rol eax, 3
; line 334+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((21)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(21+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((21)*4)], ebx
 mov [rsp+212+((21)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(21+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(21+1))]

 mov r11d, edi
 mov r14d, ebp
; line 335+1 r72-snjl.asm
 rol eax, 3
; line 335+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((22)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(22+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((22)*4)], ebx
 mov [rsp+212+((22)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(22+1))]

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(22+1))]

 mov r12d, edi
 mov r15d, ebp
; line 336+1 r72-snjl.asm
 rol eax, 3
; line 336+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((23)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(23+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((23)*4)], ebx
 mov [rsp+212+((23)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(23+1))]

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(23+1))]

 mov r13d, edi
 mov [rsp+0], ebp
; line 337+1 r72-snjl.asm
 rol eax, 3
; line 337+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((24)*4)], eax

 add esi, eax
 rol esi, cl
 lea eax, [eax + esi + (0B7E15163h+09E3779B9h*(24+1))]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((24)*4)], ebx
 mov [rsp+212+((24)*4)], edx

 rol edi, cl
 lea ebx, [ebx + edi + (0B7E15163h+09E3779B9h*(24+1))]

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 lea edx, [edx + ebp + (0B7E15163h+09E3779B9h*(24+1))]

 mov r11d, edi
 mov r14d, ebp
; line 338+1 r72-snjl.asm

 rol eax, 3
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]

 add esi, r9d
 mov [rsp+4+((25)*4)], eax

 mov [rsp+108+((25)*4)], ebx
 mov [rsp+212+((25)*4)], edx
 add esi, eax

 add eax, [rsp+4+((0)*4)]
 add edi, ebx
 add ebp, edx

 rol esi, cl
 mov ecx, edi
 add edi, r12d

 add ebx, [rsp+108+((0)*4)]
 add eax, esi
 mov r9d, esi

 rol edi, cl
 mov ecx, ebp
 add ebp, r15d

 add edx, [rsp+212+((0)*4)]
 add ebx, edi
 mov r12d, edi

 rol ebp, cl

 mov r15d, ebp
 add edx, ebp

key_setup_2:

 rol eax, 3
; line 379+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((0)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((0+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((0)*4)], ebx
 mov [rsp+212+((0)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((0+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((0+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 380+1 r72-snjl.asm
 rol eax, 3
; line 380+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((1)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((1+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((1)*4)], ebx
 mov [rsp+212+((1)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((1+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((1+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 381+1 r72-snjl.asm
 rol eax, 3
; line 381+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((2)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((2+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((2)*4)], ebx
 mov [rsp+212+((2)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((2+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((2+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 382+1 r72-snjl.asm
 rol eax, 3
; line 382+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((3)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((3+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((3)*4)], ebx
 mov [rsp+212+((3)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((3+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((3+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 383+1 r72-snjl.asm
 rol eax, 3
; line 383+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((4)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((4+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((4)*4)], ebx
 mov [rsp+212+((4)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((4+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((4+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 384+1 r72-snjl.asm
 rol eax, 3
; line 384+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((5)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((5+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((5)*4)], ebx
 mov [rsp+212+((5)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((5+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((5+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 385+1 r72-snjl.asm
 rol eax, 3
; line 385+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((6)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((6+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((6)*4)], ebx
 mov [rsp+212+((6)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((6+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((6+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 386+1 r72-snjl.asm
 rol eax, 3
; line 386+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((7)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((7+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((7)*4)], ebx
 mov [rsp+212+((7)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((7+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((7+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 387+1 r72-snjl.asm
 rol eax, 3
; line 387+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((8)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((8+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((8)*4)], ebx
 mov [rsp+212+((8)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((8+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((8+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 388+1 r72-snjl.asm
 rol eax, 3
; line 388+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((9)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((9+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((9)*4)], ebx
 mov [rsp+212+((9)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((9+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((9+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 389+1 r72-snjl.asm
 rol eax, 3
; line 389+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((10)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((10+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((10)*4)], ebx
 mov [rsp+212+((10)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((10+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((10+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 390+1 r72-snjl.asm
 rol eax, 3
; line 390+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((11)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((11+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((11)*4)], ebx
 mov [rsp+212+((11)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((11+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((11+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 391+1 r72-snjl.asm
 rol eax, 3
; line 391+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((12)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((12+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((12)*4)], ebx
 mov [rsp+212+((12)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((12+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((12+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 392+1 r72-snjl.asm
 rol eax, 3
; line 392+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((13)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((13+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((13)*4)], ebx
 mov [rsp+212+((13)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((13+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((13+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 393+1 r72-snjl.asm
 rol eax, 3
; line 393+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((14)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((14+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((14)*4)], ebx
 mov [rsp+212+((14)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((14+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((14+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 394+1 r72-snjl.asm
 rol eax, 3
; line 394+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((15)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((15+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((15)*4)], ebx
 mov [rsp+212+((15)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((15+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((15+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 395+1 r72-snjl.asm
 rol eax, 3
; line 395+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((16)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((16+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((16)*4)], ebx
 mov [rsp+212+((16)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((16+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((16+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 396+1 r72-snjl.asm
 rol eax, 3
; line 396+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((17)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((17+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((17)*4)], ebx
 mov [rsp+212+((17)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((17+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((17+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 397+1 r72-snjl.asm
 rol eax, 3
; line 397+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((18)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((18+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((18)*4)], ebx
 mov [rsp+212+((18)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((18+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((18+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 398+1 r72-snjl.asm
 rol eax, 3
; line 398+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((19)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((19+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((19)*4)], ebx
 mov [rsp+212+((19)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((19+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((19+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 399+1 r72-snjl.asm
 rol eax, 3
; line 399+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((20)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((20+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((20)*4)], ebx
 mov [rsp+212+((20)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((20+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((20+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 400+1 r72-snjl.asm
 rol eax, 3
; line 400+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((21)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((21+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((21)*4)], ebx
 mov [rsp+212+((21)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((21+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((21+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 401+1 r72-snjl.asm
 rol eax, 3
; line 401+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((22)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((22+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((22)*4)], ebx
 mov [rsp+212+((22)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((22+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((22+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 402+1 r72-snjl.asm
 rol eax, 3
; line 402+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((23)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((23+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((23)*4)], ebx
 mov [rsp+212+((23)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((23+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((23+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 403+1 r72-snjl.asm
 rol eax, 3
; line 403+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((24)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((24+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((24)*4)], ebx
 mov [rsp+212+((24)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((24+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((24+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 404+1 r72-snjl.asm

 rol eax, 3
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((25)*4)], eax

 mov [rsp+108+((25)*4)], ebx
 mov [rsp+212+((25)*4)], edx
 add esi, eax

 add eax, [rsp+4+((0)*4)]
 add edi, ebx
 add ebp, edx

 rol esi, cl
 mov ecx, edi
 add edi, r11d

 add ebx, [rsp+108+((0)*4)]
 add eax, esi
 mov r8d, esi

 rol edi, cl
 mov ecx, ebp
 add ebp, r14d

 add edx, [rsp+212+((0)*4)]
 add ebx, edi
 mov r11d, edi

 rol ebp, cl

 mov r14d, ebp
 add edx, ebp

key_setup_3:

 rol eax, 3
; line 444+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((0)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((0+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((0)*4)], ebx
 mov [rsp+212+((0)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((0+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((0+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 445+1 r72-snjl.asm
 rol eax, 3
; line 445+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((1)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((1+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((1)*4)], ebx
 mov [rsp+212+((1)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((1+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((1+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 446+1 r72-snjl.asm
 rol eax, 3
; line 446+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((2)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((2+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((2)*4)], ebx
 mov [rsp+212+((2)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((2+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((2+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 447+1 r72-snjl.asm
 rol eax, 3
; line 447+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((3)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((3+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((3)*4)], ebx
 mov [rsp+212+((3)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((3+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((3+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 448+1 r72-snjl.asm
 rol eax, 3
; line 448+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((4)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((4+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((4)*4)], ebx
 mov [rsp+212+((4)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((4+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((4+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 449+1 r72-snjl.asm
 rol eax, 3
; line 449+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((5)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((5+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((5)*4)], ebx
 mov [rsp+212+((5)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((5+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((5+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 450+1 r72-snjl.asm
 rol eax, 3
; line 450+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((6)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((6+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((6)*4)], ebx
 mov [rsp+212+((6)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((6+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((6+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 451+1 r72-snjl.asm
 rol eax, 3
; line 451+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((7)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((7+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((7)*4)], ebx
 mov [rsp+212+((7)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((7+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((7+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 452+1 r72-snjl.asm
 rol eax, 3
; line 452+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((8)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((8+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((8)*4)], ebx
 mov [rsp+212+((8)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((8+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((8+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 453+1 r72-snjl.asm
 rol eax, 3
; line 453+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((9)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((9+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((9)*4)], ebx
 mov [rsp+212+((9)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((9+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((9+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 454+1 r72-snjl.asm
 rol eax, 3
; line 454+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((10)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((10+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((10)*4)], ebx
 mov [rsp+212+((10)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((10+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((10+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 455+1 r72-snjl.asm
 rol eax, 3
; line 455+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((11)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((11+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((11)*4)], ebx
 mov [rsp+212+((11)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((11+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((11+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 456+1 r72-snjl.asm
 rol eax, 3
; line 456+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((12)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((12+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((12)*4)], ebx
 mov [rsp+212+((12)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((12+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((12+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 457+1 r72-snjl.asm
 rol eax, 3
; line 457+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((13)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((13+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((13)*4)], ebx
 mov [rsp+212+((13)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((13+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((13+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 458+1 r72-snjl.asm
 rol eax, 3
; line 458+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((14)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((14+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((14)*4)], ebx
 mov [rsp+212+((14)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((14+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((14+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 459+1 r72-snjl.asm
 rol eax, 3
; line 459+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((15)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((15+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((15)*4)], ebx
 mov [rsp+212+((15)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((15+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((15+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 460+1 r72-snjl.asm
 rol eax, 3
; line 460+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((16)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((16+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((16)*4)], ebx
 mov [rsp+212+((16)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((16+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((16+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 461+1 r72-snjl.asm
 rol eax, 3
; line 461+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((17)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((17+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((17)*4)], ebx
 mov [rsp+212+((17)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((17+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((17+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 462+1 r72-snjl.asm
 rol eax, 3
; line 462+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((18)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((18+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((18)*4)], ebx
 mov [rsp+212+((18)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((18+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((18+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 463+1 r72-snjl.asm
 rol eax, 3
; line 463+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((19)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((19+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((19)*4)], ebx
 mov [rsp+212+((19)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((19+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((19+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 464+1 r72-snjl.asm
 rol eax, 3
; line 464+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((20)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((20+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((20)*4)], ebx
 mov [rsp+212+((20)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((20+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((20+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 465+1 r72-snjl.asm
 rol eax, 3
; line 465+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((21)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((21+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((21)*4)], ebx
 mov [rsp+212+((21)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((21+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((21+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 466+1 r72-snjl.asm
 rol eax, 3
; line 466+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r10d
 mov [rsp+4+((22)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((22+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r13d

 add ebp, edx
 mov [rsp+108+((22)*4)], ebx
 mov [rsp+212+((22)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((22+1)*4)]
 add eax, esi

 mov r10d, esi
 mov ecx, ebp
 add ebp, [rsp+0]

 rol ebp, cl
 add edx, [rsp+212+((22+1)*4)]
 add ebx, edi

 mov r13d, edi
 mov [rsp+0], ebp
 add edx, ebp
; line 467+1 r72-snjl.asm
 rol eax, 3
; line 467+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r8d
 mov [rsp+4+((23)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((23+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r11d

 add ebp, edx
 mov [rsp+108+((23)*4)], ebx
 mov [rsp+212+((23)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((23+1)*4)]
 add eax, esi

 mov r8d, esi
 mov ecx, ebp
 add ebp, r14d

 rol ebp, cl
 add edx, [rsp+212+((23+1)*4)]
 add ebx, edi

 mov r11d, edi
 mov r14d, ebp
 add edx, ebp
; line 468+1 r72-snjl.asm
 rol eax, 3
; line 468+0 r72-snjl.asm
 rol ebx, 3
 rol edx, 3

 lea ecx, [eax + esi]
 add esi, r9d
 mov [rsp+4+((24)*4)], eax

 add esi, eax
 rol esi, cl
 add eax, [rsp+4+((24+1)*4)]

 add edi, ebx
 mov ecx, edi
 add edi, r12d

 add ebp, edx
 mov [rsp+108+((24)*4)], ebx
 mov [rsp+212+((24)*4)], edx

 rol edi, cl
 add ebx, [rsp+108+((24+1)*4)]
 add eax, esi

 mov r9d, esi
 mov ecx, ebp
 add ebp, r15d

 rol ebp, cl
 add edx, [rsp+212+((24+1)*4)]
 add ebx, edi

 mov r12d, edi
 mov r15d, ebp
 add edx, ebp
; line 469+1 r72-snjl.asm

 rol eax, 3
 rol ebx, 3
 rol edx, 3

 mov [rsp+4+((25)*4)], eax
 mov [rsp+108+((25)*4)], ebx
 mov [rsp+212+((25)*4)], edx







encryption:

 mov eax, [rsp+352]
 mov esi, [rsp+356]
 mov ebx, eax

 mov edx, eax
 mov edi, esi
 mov ebp, esi

 add eax, [rsp+4+((0)*4)]
 add ebx, [rsp+108+((0)*4)]
 add edx, [rsp+212+((0)*4)]

 add esi, [rsp+4+((1)*4)]
 add edi, [rsp+108+((1)*4)]
 add ebp, [rsp+212+((1)*4)]

 mov ecx, esi
; line 502+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*1)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*1)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*1)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*1+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*1+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*1+1)*4)]
; line 503+1 r72-snjl.asm
 mov ecx, esi
; line 503+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*2)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*2)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*2)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*2+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*2+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*2+1)*4)]
; line 504+1 r72-snjl.asm
 mov ecx, esi
; line 504+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*3)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*3)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*3)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*3+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*3+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*3+1)*4)]
; line 505+1 r72-snjl.asm
 mov ecx, esi
; line 505+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*4)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*4)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*4)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*4+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*4+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*4+1)*4)]
; line 506+1 r72-snjl.asm
 mov ecx, esi
; line 506+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*5)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*5)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*5)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*5+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*5+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*5+1)*4)]
; line 507+1 r72-snjl.asm
 mov ecx, esi
; line 507+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*6)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*6)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*6)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*6+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*6+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*6+1)*4)]
; line 508+1 r72-snjl.asm
 mov ecx, esi
; line 508+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*7)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*7)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*7)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*7+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*7+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*7+1)*4)]
; line 509+1 r72-snjl.asm
 mov ecx, esi
; line 509+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*8)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*8)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*8)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*8+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*8+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*8+1)*4)]
; line 510+1 r72-snjl.asm
 mov ecx, esi
; line 510+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*9)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*9)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*9)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*9+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*9+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*9+1)*4)]
; line 511+1 r72-snjl.asm
 mov ecx, esi
; line 511+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*10)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*10)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*10)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*10+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*10+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*10+1)*4)]
; line 512+1 r72-snjl.asm
 mov ecx, esi
; line 512+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*11)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*11)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*11)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*11+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*11+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*11+1)*4)]
; line 513+1 r72-snjl.asm
 mov ecx, esi
; line 513+0 r72-snjl.asm
 xor eax, esi
 xor ebx, edi

 rol eax, cl
 mov ecx, edi
 add eax, [rsp+4+((2*12)*4)]

 xor edx, ebp
 rol ebx, cl
 mov ecx, ebp

 add ebx, [rsp+108+((2*12)*4)]
 rol edx, cl
 mov ecx, eax

 add edx, [rsp+212+((2*12)*4)]
 xor esi, eax
 xor edi, ebx

 rol esi, cl
 mov ecx, ebx
 add esi, [rsp+4+((2*12+1)*4)]

 xor ebp, edx
 rol edi, cl
 mov ecx, edx

 add edi, [rsp+108+((2*12+1)*4)]
 rol ebp, cl

 add ebp, [rsp+212+((2*12+1)*4)]
; line 514+1 r72-snjl.asm

test_key_1:
 cmp eax, [rsp+360]
 mov rax, [rsp+436]

 jne short test_key_2

 inc dword ptr [rax+28]

 mov ecx, [rsp+316+((2)*4)]
 mov [rax+32], ecx

 mov ecx, [rsp+316+((1)*4)]
 mov [rax+36], ecx

 mov ecx, [rsp+316+((0)*4)]
 mov [rax+40], ecx

 cmp esi, [rsp+364]
 jne short test_key_2

 xor ecx, ecx

 jmp finished_found

align 16
test_key_2:
 cmp ebx, [rsp+360]

 jne short test_key_3

 mov esi, [rsp+328+((2)*4)]
 mov ecx, [rsp+328+((1)*4)]
 mov ebx, [rsp+328+((0)*4)]

 inc dword ptr [rax+28]

 mov [rax+32], esi
 mov [rax+36], ecx
 mov [rax+40], ebx

 cmp edi, [rsp+364]
 jne short test_key_3

 xor ecx, ecx
 dec ecx

 jmp finished_found

align 16
test_key_3:
 cmp edx, [rsp+360]
 mov edx, [rax+16]

 jne short inc_key

 mov esi, [rsp+340+((2)*4)]
 mov ecx, [rsp+340+((1)*4)]
 mov ebx, [rsp+340+((0)*4)]

 inc dword ptr [rax+28]

 mov [rax+32], esi
 mov [rax+36], ecx
 mov [rax+40], ebx

 cmp ebp, [rsp+364]

 jne short inc_key

 mov ecx, -2

 jmp finished_found


align 16
inc_key:
 cmp dl, 0FDh
 mov ecx, [rax+20]
 mov ebx, [rax+24]

 jae complex_incr

 add dl, 3

 mov [rax+16], edx
 mov r10d, edx
 mov [rsp+316+((2)*4)], edx
 inc edx

 mov r13d, edx
 mov [rsp+328+((2)*4)], edx
 inc edx

 mov [rsp+0], edx
 mov [rsp+340+((2)*4)], edx
 dec dword ptr [rsp+368]

 mov r9d, ecx
 mov r12d, ecx
 mov r15d, ecx

 mov r8d, ebx
 mov r11d, ebx
 mov r14d, ebx

 jnz key_setup_1

 xor eax, eax
 jmp finished

align 16
complex_incr:
 add dl, 3
 bswap ecx
 bswap ebx

 adc ecx, 0
 adc ebx, 0

 bswap ecx
 bswap ebx

 mov r10d, edx
 mov r9d, ecx
 mov r8d, ebx

 mov [rsp+316+((2)*4)], edx
 mov [rsp+316+((1)*4)], ecx
 mov [rsp+316+((0)*4)], ebx

 mov [rax+16], edx
 mov [rax+20], ecx
 mov [rax+24], ebx

 add dl, 1
 bswap ecx
 bswap ebx

 adc ecx, 0
 adc ebx, 0

 bswap ecx
 bswap ebx

 mov r13d, edx
 mov r12d, ecx
 mov r11d, ebx

 mov [rsp+328+((2)*4)], edx
 mov [rsp+328+((1)*4)], ecx
 mov [rsp+328+((0)*4)], ebx

 add dl, 1
 bswap ecx
 bswap ebx

 adc ecx, 0
 adc ebx, 0
 dec dword ptr [rsp+368]

 bswap ecx
 bswap ebx

 mov [rsp+0], edx
 mov r15d, ecx
 mov r14d, ebx

 mov [rsp+340+((2)*4)], edx
 mov [rsp+340+((1)*4)], ecx
 mov [rsp+340+((0)*4)], ebx

 jnz key_setup_1

 xor eax, eax
 jmp short finished
finished_found:
 mov rsi, [rsp+444]
 add ecx, [rsp+368]
 add ecx, [rsp+368]
 add ecx, [rsp+368]
 sub [rsi], ecx

 xor eax, eax
 inc eax
finished:
 inc eax



 mov rsi, [rsp+388]
 mov rdi, [rsp+396]


 mov rbp, [rsp+380]
 mov rbx, [rsp+372]
 mov r12, [rsp+404]
 mov r13, [rsp+412]
 mov r14, [rsp+420]
 mov r15, [rsp+428]


 add rsp, 452

 ret
rc5_72_unit_func_snjl endp
text	ENDS
END

