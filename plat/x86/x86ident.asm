;
; x86 Processor identification for rc5 distributed.net effort
; Written in a dark and stormy night (Jan 16, 1998) by
; Cyrus Patel <cyp@fb14.uni-mainz.de>
;
; $Id: x86ident.asm,v 1.1.2.1 2001/01/21 15:09:27 cyp Exp $
;
; correctly identifies almost every 386+ processor with the
; following exceptions:
; a) The code to use Cyrix Device ID registers for differentiate between a 
;    Cyrix 486, 6x86, etc is in place but disabled. Even so, it would not
;    detect correctly if CPUID is enabled _and_ reassigned.
; b) It may not correctly identify a Rise CPU if cpuid is disabled/reassigned
; c) It identifies an Intel 486 as an AMD 486 (chips are identical) and not
;    vice-versa because the Intel:0400 CPUID is actually a real CPUID value.
; d) It _does_ id the Transmeta Crusoe (its Family:5,Model:4) :)
;
; x86ident:  return u32
;            hiword = manufacturer ID (also for 386/486)
;                     (BX from cpuid for simplicity)
;                    [bxBXdxDXcxCX]
;                     GenuineIntel = 'eG' = 0x6547
;                     GenuineTMx86 = 'MT' = 0x4D54 <-- NOTE!
;                     AuthenticAMD = 'uA' = 0x7541
;                     CyrixInstead = 'yC' = 0x7943
;                     NexGenDriven = 'eN' = 0x654E
;                     CentaurHauls = 'eC' = 0x6543
;                     UMC UMC UMC  = 'MU' = 0x4D55
;                     RiseRiseRise = 'iR' = 0x6952
;            loword = bits 12..15 for identification extensions
;                     bits 11...0 for family/model/stepping per CPUID
;                                 or 0x0300 for 386, 0x0400 for 486
;
; 0123456789|ABCDEFGHIJKLMNOPQRSTUVWXYZ|abcdefghijklmnopqrstuvwxyz|
; 3        3|4              5          |6              7          |
; 0123456789|123456789ABCDEF0123456789A|123456789ABCDEF0123456789A|

%ifndef __OMF__
  %ifdef OS2
    %define __OMF__
  %endif
%endif

%ifdef __OMF__   ; Watcom+OS/2 or Borland+Win32
[SECTION DATA CLASS=DATA USE32 PUBLIC ALIGN=16]
[SECTION TEXT CLASS=CODE USE32 PUBLIC ALIGN=16]
%define __DATASECT__ [SECTION DATA]
%define __CODESECT__ [SECTION TEXT]
%else
%define __DATASECT__ [SECTION .data]
%define __CODESECT__ [SECTION .text]
%endif

global          x86ident,_x86ident
global          x86ident_haveioperm, _x86ident_haveioperm

__DATASECT__
__savident      dd 0              
_x86ident_haveioperm:             ; do we have permission to do in/out?
x86ident_haveioperm dd 0          ; we do on win9x (not NT), win16, dos

__CODESECT__
_x86ident:
x86ident:       mov     eax,[__savident]
                or      eax, eax
                jz      _ge386
                ret

_ge386:         ; an interrupt could change the AC bit, so loop 'n'
                ; times, incrementing a state counter for each state
                pushfd                  ; save flags from damage
                xor     edx,edx         ; our counter and state
                pushfd                  ; copy EFLAGS
                pop     ecx             ; ... into ecx
_386next:       mov     eax, ecx        ; copy original EFLAGS
                xor     eax, 40000h     ; toggle AC bit (1<<18) in eflags
                push    eax             ; copy modified eflags
                popfd                   ; ... to EFLAGS
                pushfd                  ; copy (possible modified) EFLAGS
                pop     eax             ; ... back to eax
                cmp     eax, ecx        ; will be 386 if no change
                setz    al              ; set to one if 386, else zero
                setnz   ah              ; set to zero if 386, else one
                add     dx,ax           ; add to totals
                mov     al,dl           ; copy our 'is386' count
                add     al,dh           ; add the 'not386' count
                cmp     al,31           ; done 31 checks?
                jb      _386next        ; continue looping if not
                popfd                   ; restore saved flags
                cmp     dl,dh           ; 'is386' count less than 'not386'?
                jb      _ge486          ; its a 486+ if so

                ;we're either a 386 or a  nextgen 386 class CPU.

                mov     ax, 5555h       ; cx+nexgen check (only if <PII)
                xor     dx, dx          ; clear high word
                cmp     ax, ax          ; set ZF=1
                mov     cx, 2           ; odd by even division
                div     cx              ; i386 clears ZF, Nx586 leaves it set
                mov     eax, 65470300h  ; 'eG' as in "GenuineIntel + i386
                jnz     near _end       ; Intel chip if flags didn't change
                ;note that the div odd/even test is only valid on <=P5
                ;on PPro and higher, the flags do change
                ;if 386sx/386dx difference is relevant, see if the ET bit in
                ;the MSW can be toggled. It can on an SX, is always 1 elsewhere.
                
                ;Nx chips don't have EFLAGS.id, but newer ones support the 
                ;cpuid opcode. (I've seen the following code in some NexGen 
                ;utility: mov si,B00Eh;mov ax,4c03h; db 0Fh,55h <-- ???)
                ;In any event, we don't need the feature flags: all zero 
                ;anyway, except for the Nx587 [75+/90Mhz] which has an fpu,
                ;the other NexGen model, the Nx586 [60/66Mhz] doesn't.

                mov     eax, 654E050Fh  ; family=5,model=0,stepping=15
                jmp     _end            ; (thats a real cpuid 1 result)

_ge486:         mov     edx, 200000h    ; ID bit
                pushfd                  ; save flags from damage
                pushfd                  ; copy EFLAGS
                pop     ecx             ; ... into ecx
                mov     eax, ecx        ; copy original flags
                xor     eax, edx        ; try to toggle ID bit
                push    eax             ; copy modified eflags
                popfd                   ; ... to EFLAGS
                pushfd                  ; push back flags
                pop     eax             ; ... into eax
                popfd                   ; restore saved flags
                and     eax, edx        ; isolate ID bit in result
                and     ecx, edx        ; isolate ID bit in original
                cmp     eax, ecx        ; must be 486 if no change
                jnz     near _ge586     ; otherwise it supports cpuid

                ; -----------------------------------------------------

                fninit                  ; try to init coprocessor
                mov     ax, 5a5ah       ; rigged status word
                fnstsw  ax              ; try to store
                or      ax, ax          ; have an fpu?
                jnz     _in486sx        ; Intel 486sx if not 

                ; The f2xm1(pi) test:
                ; amd/intel appear to have the result as 3E46FEACh ((u32)pi)
                ; (pi as dword) but I'd rather not trust it since Intel
                ; docs specifically state that 2^pi-1 is undefined.
                ; Cyrix itself uses the magic 3FC9xxxxh in its detection code.

                ;Note that here we return *AMD* for Intel/AMD 486DX cpus
                ;because 400h is a real cpuid ident for "Intel 486DX25 or 33"
                ;That way the caller can differentiate between a 'guess'
                ;and a real cpuid (for instance calling it "Intel/AMD 486")

                push    eax             ; make buffer space
                finit                   ; initialize fpu
                fldpi                   ; loads pi
                f2xm1                   ; 2^pi - 1 => "undefined" number
                fstp    dword [esp]     ; store in slot
                pop     ecx             ; get the result
                rol     ecx, 16         ; switch hi/lo words
                cmp     cx,  3FC9h      ; cyrix=>3FC9xxxxh
                mov     eax, 75410400h  ; return *AMD* 486 CPU
                jnz     near _end       ; its an intel/AMD CPU if not

                ; we know we have a Cyrix co-processor, but that 
                ; could be a cyrix NPU for a non-cyrix CPU.
                ; find out whether the main processor is a cyrix too.
                ; If it isn't, then it must be an Intel 486sx.
                ;
                ; Actually, the following 5/2 test is not valid for the M2
                ; where the flags behave just as their Intel/AMD pendants do.
                ; But the M2 has a "normal" cpuid, so it won't get here.
                ;
                ; Another test for Cyrix chips is 'UMOV reg,[mem]'
                ; (only available on a 386/486, not on a P5+, and assuming 
                ; this isn't a circuit emulator), Intel/AMD processors
                ; do a normal MOV, Cx/TI/IBM chips treat it as a NOP.

                xor     ah, ah          ; for clearing flags
                sahf                    ; scrub lower 8 bits (exc. bit 1)
                lahf                    ; reload flags
                mov     ch, ah          ; save copy
                mov     ax, 5           ; 5/2 constant
                mov     cl, 2           ; divisor
                div     cl              ; won't change flags on cyrix
                lahf                    ; load flags
                cmp     ah, ch          ; no changes?
                jz      near _cx486     ; must be a Cyrix chip if so

_in486sx:       mov     eax, 65470420h  ; Intel 486SX (no fpu)
                jmp     _end            ; go exit

                ; -----------------------------------------------------

_ge586:         push    ebx             ; cpuid trashes ebx
                xor     eax, eax        ; cpuid function zero
                cpuid                   ; => eax=maxlevels,ebx:ecx:edx=vendor
                push    eax             ; save maxlevels
                and     al, 0f0h        ; clear stepping
                cmp     eax, 00000500h  ; 50xh cpuid maxlevel? I think not!
                pop     eax             ; restore saved maxlevels
                jnz     _ne586a         ; not a P5/Step-A screwball
                mov     ebx, 756E6547h  ; P5 step A screwball (featuref=1BFh)
                mov     edx, 49656E69h  ; Genu[ineI]ntel
                mov     ecx, 6C65746Eh  ; GenuineI[ntel]
                mov     eax, 1          ; maxlevel is one
_ne586a:        mov     edx, ebx        ; this is now our vendor id (in DX)
                pop     ebx             ; restore ebx

; 0123456789|ABCDEFGHIJKLMNOPQRSTUVWXYZ|abcdefghijklmnopqrstuvwxyz|
; 3        3|4              5          |6              7          |
; 0123456789|123456789ABCDEF0123456789A|123456789ABCDEF0123456789A|

                cmp     ecx,3638784Dh   ; '68xM' as in GenuineT[Mx86]
                jnz     _neTM           ; not a Transmeta chip
                mov     edx,38784D54h   ; '8xMT' as in Genuine[TMx8]6

_neTM:          xchg    edx,eax         ; make edx=maxlevel,eax=vendor id
                shl     eax,16          ; move it up
                mov     ax,400h         ; assume 486
                or      edx,edx         ; max cpuid level is zero?
                jz      near _end       ; exit if so

                push    edx             ; save max level
                push    eax             ; save maker code
                mov     eax, 1          ; family/model/stepping/features
                push    ebx             ; cpuid trashes ebx
                xor     ebx,ebx         ; assume no brand ID
                cpuid                   ; => ax=type/family/model/stepping
                mov     edx,ebx         ; copy brand ID
                pop     ebx             ; restore ebx
                and     ax,0fffh        ; drop the type flags
                mov     cx,ax           ; save family/model/stepping bits
                pop     eax             ; restore maker code
                mov     ax,cx           ; add family/model/stepping bits
                pop     ecx             ; restore max level 
                                        ; edx has feature bits

                ;On a PII and above we take extra steps to differentiate
                ;between a Celeron/Covington, PII, Celeron-A/Mendocino, Xeon.
                ;we need to do this since cache size may be important
                ;for some cores. On a P4, we have the 'Brand' bits
                cmp     ecx,2           ; max cpuid level >= 2?
                jb      _end            ; can't be a PII+ if not
                mov     ecx,eax         ; copy our combined id
                shr     ecx,16          ; get vendor id in cx
                cmp     cx, 6547h       ; 'Ge'nuineIntel?
                jnz     _end            ; exit it not
                mov     cx, ax          ; get family/model/stepping bits
                and     cx,0ff0h        ; mask only family/model
                cmp     cx,620h         ; less than PII?
                jb      _end            ; neither brand nor cache bits needed
                or      dl,dl           ; have brand bits?
                jnz     _brand          ; continue if so
                cmp     cx,650h         ; Celeron/PII/Xeon (Covington)?
                jb      _end            ; proceed if so
                cmp     cx,660h         ; Celeron-A/PII/Xeon (Mendocino/Dixon)?
                ja      _end            ; exit if not 
                push    eax             ; save our old result
                push    ebx             ; save ebx from damage
                mov     eax,2           ; do cpuid level 2
                cpuid                   ; => dl=cache info
                pop     ebx             ; restore ebx
                pop     eax             ; restore our old result
                mov     cl,dl           ; 0x40=no cache,1=128,2=256,...
                cmp     cl,40h          ; ...,3=512,4=1024,5=2048
                mov     dl,0x01         ; assume its a plain Celeron
                jz      _brand          ; go brand if no L2
                cmp     cl,41h          ; has 128K L2?
                jz      _brand          ; its a Celeron-A if so
                mov     dl,0x04         ; assume its a Xeon
                cmp     cl,44h          ; has 1MB L2?
                jz      _brand          ; its a Xeon if so
                cmp     cl,45h          ; has 2MB L2?
                jnz     _end            ; go exit if not
_brand:         shl     dl,4            ; 'Brand' lsn->brand msn
                or      ah,dl           ; 0/fam/mod/step -> brand/fam/mod/step
                or      dl,dl           ; did we have brand bits?
                jnz     _end            ; exit if so

_end:           mov     [__savident],eax; save it for next time
                ret

;----------------------------------------------------------------------

_cx486:         ; -----------------------------------------------------
                ; If we got here we are definitely on a Cyrix CPU
                ; so (attempt to) use Cyrix 'Device ID registers' to identify.
                ;
                ; Cyrix documentation states that the in/outs here don't
                ; generate external signals on the bus (ie, the wouldn't 
                ; generate an exception), but that is apparently incorrect. 
                ; It does work for Win9x/DOS/Win16 (no protection), and
                ; NetWare (which runs at IOPL0) though.

                xor     ecx,ecx        ; faked DIR values for Cx486
                cmp     ecx,[x86ident_haveioperm] ; have IO permission?
                jz      near _cxeval   ; hop over CCR/DIR stuff if not
                
                ; #define getCx86(reg)  ({ outb((reg), 0x22); inb(0x23); })
                ; #define setCx86(reg, data) do { \ outb((reg), 0x22); \ outb((data), 0x23); \ } while (0)

                %define IODELAY db 0xEB,0x00,0xEB,0x00 ; (jmp short $+2) twice
                %macro getCCR 1
                mov     al, %1         ; the # of the CCR we want to read
                mov     dx, 22h        ; command port at 0x22
                IODELAY                ; jmp short $+2, jmp short $+2
                out     dx, al         ; access command
                inc     dx             ; read/write from/to 0x23
                in      al, dx         ; read CCRx value
                %endmacro
                %macro setCCR 1
                mov     ah, al         ; the value we want to write
                mov     al, %1         ; the # of the CCR we want to write
                mov     dx, 22h        ; command port at 0x22
                IODELAY                ; jmp short $+2, jmp short $+2
                out     dx, al         ; access command
                inc     dx             ; read/write from/to 0x23
                IODELAY                ; jmp short $+2, jmp short $+2
                mov     al, ah         ; the value we want to write
                out     dx, al         ; write CCRx value
                %endmacro

                ;Cyrix CPU configuration register indexes
                CX86_CCR0  equ 0c0h
                CX86_CCR1  equ 0c1h
                CX86_CCR2  equ 0c2h
                CX86_CCR3  equ 0c3h
                CX86_CCR4  equ 0e8h
                CX86_CCR5  equ 0e9h
                CX86_CCR6  equ 0eah
                CX86_DIR0  equ 0feh
                CX86_DIR1  equ 0ffh
                CX86_ARR_BASE  equ 0c4h
                CX86_RCR_BASE  equ 0dch

                getCCR  CX86_CCR3      ; get the CCR3 value 
                mov     cl, al         ; save ccr3
                xor     al, 80h        ; modify for change test (CCR3_MAPEN3)
                setCCR  CX86_CCR3      ; set the new CCR3 value 
                getCCR  CX86_CCR3      ; get the CCR3 value again
                xchg    al, cl         ; switch old/new values
                cmp     cl, al         ; has it changed?
                jz      _cx486s        ; no DIR registers if not

                ;all cx chips except Cx486SLC, Cx486DLC and Cx486S/A-Step
                setCCR  CX86_CCR3      ; restore the old CCR3 value 
                getCCR  CX86_DIR0      ; read DIR0
                mov     cl, al         ; save it
                xor     ch, ch         ; assume don't need DIR1
                and     al, 0f0h       ; isolate high nibble
                cmp     al, 40h        ; Media GX/GXm?
                jnz     _cxeval        ; don't need DIR1 if not
                getCCR  CX86_DIR1      ; read DIR1
                mov     ch, al         ; save it
                jmp     short _cxeval  ; go evaluate

_cx486s:        getCCR  CX86_CCR2      ; get the CCR2 value 
                mov     cl, al         ; save ccr2
                xor     al, 40h        ; modify for change test (CCR2_LOCK_NW)
                setCCR  CX86_CCR2      ; set the new CCR2 value 
                getCCR  CX86_CCR2      ; get the CCR2 value again
                xchg    al, cl         ; switch old/new values
                cmp     cl, al         ; has it changed?
                mov     cx,0000h       ; DIR0 = 0x00 = Cx486SLC
                jz      _cxeval        ; jmp if not changed, ie Cx486SLC/DLC
                setCCR  CX86_CCR2      ; restore the old CCR2 value 
                mov     cx,0010h       ; DIR0 = 0x10 = Cx486S A step

_cxeval:        ;DIR0 = cl, DIR1 = ch
                ; cpuid & 0xff0:
                ;   0x440=Media GX and GXm, 0x490=5x86, 0x520=6x86(M1), 
                ;   0x540=GXm, 0x600=6x86mx(M2)

                ;DIR0 values:
                ;  00h=80486SLC       GX: 41h(3.0x), 44h(4.0x), 45h(GX/S,3.0x)
                ;  01h=80486DLC           46h(4.0x), 47h(GX/P,3.0x), 
                ;  02h=80486SLC2,     GXm:40h(4.0x), 41h(6.0x), 42h(4.0x), 
                ;  03h=80486DLC2          43h(6.0x), 44h(7.0x), 46h(7.0x), 
                ;  04h=80486SRx,          47h(5.0x)
                ;  05h=80486DRx       (note that there are two 41s/44s/46s/47s)
                ;  06h=80486SRx2,     
                ;  07h=80486DRx2      
                ;  08h=80486SRu, 
                ;  09h=80486DRu       M1: 30h/32h(1.0x), 31h/33h(2.0x), 
                ;  0Ah=80486SRu2,         35h/37h(3.0x), 34h/36h(4.0x)
                ;  0Bh=80486DRu2      M2: 50h/58h(1.0x), 51h/59h(2.0x),
                ;  10h=80486S,            52h/5Ah(2.5x), 53h/5Bh(3.0x),
                ;  11h=80486S2            54h/5Ch(3.5x), 55h/5Dh(4.0x),
                ;  12h=80486Se,           56h/5Eh(4.5x), 57h/5Fh(5.0x)
                ;  13h=80486S2e       
                ;  1Ah=80486DX,  FFh=very likely not a Cyrix/IBM processor 
                ;  1Bh=80486DX2,      
                ;  1Fh=80486DX4       
                ;  28h/2Ah: 5x86 (1.0x mode) 
                ;  29h/2Bh: 5x86 (2.0x mode)
                ;  2Dh/2Fh: 5x86 (3.0x mode) 
                ;  2Dh/2Fh: 5x86 (3.0x mode)
                ;  2Ch/2Eh: 5x86 (4.0x mode)

                ;DIR1 values: 
                ;  bits 7..4 STEP processor stepping
                ;       3..0 REV processor revision 

                mov     eax, 79430000h ; cyrix something.
                mov     al, cl         ; set emulated model/stepping = DIR0

                mov     dl, cl         ; copy DIR0
                and     dl, 0f0h       ; dir0_msn get "family"

                cmp     dl, 20h        ; < 5x86?
                jb      _cx4x86        ; proceed if so
                jz      _cx5x86        ; Cx5x86
                cmp     dl, 30h        ; Cx6x86(M1)?
                jz      _cx6x86        ; proceed if so
                cmp     dl, 40h        ; Media GX/GXm?
                jz      _cx4xgx        ; proceed if so
                cmp     dl, 0f0h       ; TI, Overdrive and other.
                jz      _ti4x86        ; proceed if so
                cmp     dl, 50h        ; Cx6x86MX(M2)?
                jnz     _cx4x86        ; no? something wrong. assume 486
                ;DIR0 = 50h - 5fh
                mov     ah, 6          ; 686 class CPU
                and     al, 0fh        ; 065x -> 060x
                jmp     _end           ; model 0=6x86MX
_cx6x86:        ;DIR0 = 30h - 3fh
                mov     ah, 5          ; 586 class CPU
                and     al, 2fh        ; 053x -> 052x
                jmp     _end           ; model 2=6x86
_cx5x86:        ;DIR0 = 20h - 2fh      ; freebsd says (but I don't believe)
                mov     dh, cl         ; ... if the dir0 >= 20h and dir0 < 28h
                and     dh, 0fh        ; ... then this is an 6x86(M1). Although
                cmp     dh, 8          ; ... thats implausible, we do it anyway
                jb      _cx6x86        ; ... since Cyrix doesn't doc 20h-27h
                mov     ah, 4          ; 486 class CPU
                add     al, 70h        ; 042x -> 049x
                jmp     _end           ; model 9=5x86
_cx4xgx:        ;DIR0 = 40h - 4fh; DIR1=set
                mov     dh, ch         ; freebsd says ...
                and     dh, 0f0h       ; ... if the dir1_msn (stepping)
                cmp     dh, 030h       ; ... == 3, then this is a GXm (MMX!)
                jnz     _cx4x86        ; this is a MediaGX if not
                mov     ah, 5          ; 586 class CPU if GXm
                jmp     _end           ; GXm = class 5, model 4
_ti4x86:        ;DIR0=0xFE=TI486SXL,0xFD=Overdrive,
                and     al,0fh         ; convert to Cx486
_cx4x86:        or      ah, 4          ; 486 class CPU
                jmp     _end           ; model 0=Cx486SLC/DLC/SRx/DRx,
                                       ; model 1=Cx486S/DX/DX2/DX4, 4=MediaGX

