/*
 * CPU detection routines for rc5 distributed.net effort
 * Written in a dark and stormy night by Cyrus Patel <cyp@fb14.uni-mainz.de>
 *
 * See x86ident.asm for documentation.
 *
*/

.data
_id:	        .ascii "@(#)$Id: x86ident.S,v 1.1.2.1 2001/01/21 15:09:27 cyp Exp $"
		.byte 0
		.align 4
_savident:	.long 0              // avoid serialization caused by CPUID

#if defined(__bsdi__) || defined(__NeXT__)
#define cpuid .byte 0x0f, 0xa2
#endif

.globl  _x86ident
.globl	x86ident

.text
_x86ident:
x86ident:
		movl     _savident,%eax
                orl      %eax, %eax
                jz       _Getx86FamID
                ret

_Getx86FamID:   pushl    %ebx           // save ebx from damage
                pushf                   // save flags from damage

                pushf                   // copy EFLAGS
                popl     %ecx           // ... into ecx

                // an interrupt would change the AC bit, so loop 'n'
                // times, incrementing a state counter for each state
                xorl    %ebx, %ebx      // bl=is386 count, bh=not386 count
                movb    $31, %dl        // the number of times we check
_386next:       movl    %ecx, %eax      // copy original EFLAGS
                xorl    $(1 << 18),%eax // toggle AC bit in eflags
                pushl   %eax            // copy modified eflags
                popf                    // ... to EFLAGS
                pushf                   // copy (possible modified) EFLAGS
                popl    %eax            // ... back to eax
                subl    %ecx, %eax      // will be 386 if no change
                cmpl    $1, %eax        // set carry if 386
                sbbl    %eax, %eax      // make 0xffffffff if carry set
                decb    %dl             // decrement loop count
                jz      _386eval        // break if done all
                incb    %bl             // increment the is386 count
                orb     %al,%al         // clear zero if is386
                jnz     _386next        // go loop if hit
                decb    %bl             // undo last change
                incb    %bh             // increment the not386 count
                jmp     _386next        // go loop for other hit
_386eval:       cmpb    %bh, %bl        // is386 count > not386 count
                movl    $(('n << 24) + ('I << 16) + 0x30), %eax // assume i386
                ja      _nocpuidxit     // exit if so
		
                pushf                   // added
                popl    %ecx            // added
                movl    %ecx, %eax      // copy original flags
                xorl    $0x200000,%eax  // try to toggle ID bit
                pushl   %eax            // copy modified eflags
                popf                    // ... to EFLAGS
                pushf                   // push back flags
                popl    %eax            // ... into eax
                xor     %ecx,%eax       // must be 486 if no change
                jnz     _cpuidGI        // proceed if change

                xorb    %ah, %ah        // for clearing flags
                sahf                    // scrub lower 8 bits (exc. bit 1)
                lahf                    // reload flags
                movb    %ah, %ch        // save copy
                movw    $5, %ax         // 5/2 constant
                movb    $2, %cl         // divisor
                divb    %cl             // won't change flags on cyrix
                lahf                    // load flags
                subb    %ch, %ah        // no changes?
                movl    $(('y << 24) + ('C << 16) + 0x40), %eax // assume cyrix 486 CPU
                jz      _nocpuidxit     // exit if cyrix cpu
                // could use Cyrix 'Device ID registers' to get
                // exact name and model
                // 6x86 MAY return 486 here...

                fninit                  // try to init coprocessor
                movw    $0x5a5a, %ax    // rigged status word
                fnstsw  %ax             // try to store
                orw     %ax, %ax        // have an fpu?
                movl    $(('n << 24) + ('I << 16) + 0x40), %eax // no copro so must be intel
                jnz     _nocpuidxit     // if no fpu, go exit

                pushl   %eax            // make buffer space
                finit                   // initialize fpu
                fldpi                   // loads pi
                f2xm1                   // 2^pi - 1 => "undefined" number
                fstps   (%esp)	        // store in slot
                popl    %ecx            // get the result
                cmpl    $0x3FC9, %ecx   // cyrix=>3FC9h, amd=>???
                movl    $(('y << 24) + ('C << 16) + 0x40), %eax // assume cx486 CPU
                jz      _nocpuidxit     // exit if cyrix
                cmpl    $0x3E46FEAC, %ecx // pi as a dword?
                movl    $(('n << 24) + ('I << 16) + 0x40), %eax // assume intel CPU
                jz      _nocpuidxit     // exit if intel (>=P5 it's (u32)PI)
                movl    $(('u << 24) + ('A << 16) + 0x40), %eax // assume amd CPU
_nocpuidxit:    jmp     _end            // go exit

_cpuidGI:       xorl    %eax, %eax      // cpuid function zero
                cpuid                   // do the honky tonk

                movw    %bx, %cx        // move maker id
                xchgl   %eax, %ecx      // save eax, set mfg id
                shll    $16, %eax       // move it up
                movb    $0x40, %al      // assume 486
                orl     %ecx, %ecx      // only supports CPUID fxn 0?
                jz      _end            // exit if so

                pushw   %bx             // save maker code
                movl    $1, %eax        // family/model/stepping/features
                cpuid                   // do it
                movl    %eax, %ecx      // copy family/model/stepping bits

                xorl    %eax, %eax      // zero the register
                popw    %ax             // restore maker code
                shll    $16, %eax       // shift it to the left 16 bits
                andl    $0xffff, %ecx   // mask out high bits, just in case
		orl     %ecx, %eax      // add to eax's low 16 bits
		jmp     _end            // done, return
                // C calling routine interprets family/proc id

_end:           movl    %eax, _savident // save it for next time
                popf
                popl    %ebx
                ret

