	.text
	.file	"dispatch_module"
	.p2align	4, 0x90         # -- Begin function __get_system_isa
	.type	__get_system_isa,@function
__get_system_isa:                       # @__get_system_isa
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	movl	$1, %eax
	#APP
	cpuid
	#NO_APP
	movl	%edx, %r8d
	movl	%ecx, %edi
	xorl	%ecx, %ecx
	movl	$7, %eax
	#APP
	xchgl	%ebx, %esi
	cpuid
	xchgl	%ebx, %esi

	#NO_APP
	testl	$134217728, %edi        # imm = 0x8000000
	je	.LBB0_9
# %bb.1:                                # %land.lhs.true
	movl	%esi, %eax
	andl	$65568, %eax            # imm = 0x10020
	cmpl	$65568, %eax            # imm = 0x10020
	jne	.LBB0_5
# %bb.2:                                # %land.lhs.true9
	xorl	%ecx, %ecx
	#APP
	.byte	15
	.byte	1
	.byte	208
	#NO_APP
	andl	$230, %eax
	cmpl	$230, %eax
	jne	.LBB0_5
# %bb.3:                                # %if.then
	movl	%esi, %ecx
	andl	$-805175296, %ecx       # imm = 0xD0020000
	movl	$6, %eax
	cmpl	$-805175296, %ecx       # imm = 0xD0020000
	je	.LBB0_11
# %bb.4:                                # %if.else
	movl	%esi, %ecx
	andl	$469762048, %ecx        # imm = 0x1C000000
	movl	$5, %eax
	cmpl	$469762048, %ecx        # imm = 0x1C000000
	je	.LBB0_11
.LBB0_5:                                # %if.end39
	movl	%edi, %eax
	andl	$402653184, %eax        # imm = 0x18000000
	cmpl	$402653184, %eax        # imm = 0x18000000
	jne	.LBB0_9
# %bb.6:                                # %land.lhs.true47
	xorl	%ecx, %ecx
	#APP
	.byte	15
	.byte	1
	.byte	208
	#NO_APP
	andl	$6, %eax
	cmpl	$6, %eax
	jne	.LBB0_9
# %bb.7:                                # %if.then50
	andl	$1610612736, %edi       # imm = 0x60000000
	movl	$2, %eax
	cmpl	$1610612736, %edi       # imm = 0x60000000
	jne	.LBB0_11
# %bb.8:                                # %if.then58
	shrl	$5, %esi
	andl	$1, %esi
	addl	$3, %esi
	movl	%esi, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB0_9:                                # %if.else65
	.cfi_def_cfa_offset 16
	testl	$524288, %edi           # imm = 0x80000
	movl	$1, %eax
	jne	.LBB0_11
# %bb.10:                               # %if.else70
	xorl	%eax, %eax
	testl	$67108864, %r8d         # imm = 0x4000000
	je	.LBB0_12
.LBB0_11:                               # %return
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB0_12:                               # %if.else75
	.cfi_def_cfa_offset 16
	callq	abort@PLT
.Lfunc_end0:
	.size	__get_system_isa, .Lfunc_end0-__get_system_isa
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function __set_system_isa
	.type	__set_system_isa,@function
__set_system_isa:                       # @__set_system_isa
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rax
	.cfi_def_cfa_offset 16
	cmpl	$-1, __system_best_isa(%rip)
	je	.LBB1_1
# %bb.2:                                # %done
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.LBB1_1:                                # %set_system_isa
	.cfi_def_cfa_offset 16
	callq	__get_system_isa
	movl	%eax, __system_best_isa(%rip)
	popq	%rax
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	__set_system_isa, .Lfunc_end1-__set_system_isa
	.cfi_endproc
                                        # -- End function
	.globl	rc5_16pipe              # -- Begin function rc5_16pipe
	.p2align	4, 0x90
	.type	rc5_16pipe,@function
rc5_16pipe:                             # @rc5_16pipe
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	pushq	%rax
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	movl	%r9d, %r14d
	movl	%r8d, %r15d
	movl	%ecx, %r12d
	movl	%edx, %r13d
	movl	%esi, %ebx
	movl	%edi, %ebp
	callq	__set_system_isa
	movl	__system_best_isa(%rip), %eax
	cmpl	$6, %eax
	jl	.LBB2_2
# %bb.1:                                # %do_call
	movl	%ebp, %edi
	movl	%ebx, %esi
	movl	%r13d, %edx
	movl	%r12d, %ecx
	movl	%r15d, %r8d
	movl	%r14d, %r9d
	movq	72(%rsp), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	movq	72(%rsp), %rax
	pushq	%rax
	.cfi_adjust_cfa_offset 8
	callq	rc5_16pipe_avx512skx@PLT
	jmp	.LBB2_4
.LBB2_2:                                # %next_try
	.cfi_def_cfa_offset 64
	movq	72(%rsp), %r11
	movq	64(%rsp), %r10
	cmpl	$5, %eax
	jl	.LBB2_6
# %bb.3:                                # %do_call2
	movl	%ebp, %edi
	movl	%ebx, %esi
	movl	%r13d, %edx
	movl	%r12d, %ecx
	movl	%r15d, %r8d
	movl	%r14d, %r9d
	pushq	%r11
	.cfi_adjust_cfa_offset 8
	pushq	%r10
	.cfi_adjust_cfa_offset 8
	callq	rc5_16pipe_avx512knl@PLT
.LBB2_4:                                # %do_call2
	addq	$24, %rsp
	.cfi_adjust_cfa_offset -24
	jmp	.LBB2_5
.LBB2_6:                                # %next_try3
	.cfi_def_cfa_offset 64
	callq	abort@PLT
	addq	$8, %rsp
	.cfi_def_cfa_offset 56
.LBB2_5:                                # %do_call2
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end2:
	.size	rc5_16pipe, .Lfunc_end2-rc5_16pipe
	.cfi_endproc
                                        # -- End function
	.type	__system_best_isa,@object # @__system_best_isa
	.data
	.p2align	2
__system_best_isa:
	.long	4294967295              # 0xffffffff
	.size	__system_best_isa, 4


	.section	".note.GNU-stack","",@progbits
