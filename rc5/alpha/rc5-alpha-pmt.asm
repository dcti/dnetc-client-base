//	RC5-64 core decrypt for Alpha-AXP CPU's
//	Copyright (c) Pedro Miguel Teixeira, 1998
//	All Rights Reserved.

	.globl rc5_unit_func

	.data


S1:
	.quad 0,0,0,0,0,0,0,0,0,0,0,0,0


	.text
rc5_unit_func:

	ldl		$2,16($16);
	ldl		$3,20($16);

	mov		0xB7E15163,$0;


	zapnot	$0,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$5;
	and		$1,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+4;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+8;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+12;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+16;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+20;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+24;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+28;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+32;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+36;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+40;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+44;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+48;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+52;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+56;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+60;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+64;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+68;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+72;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+76;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+80;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+84;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+88;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+92;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+96;

	addl	$0,0x9E3779B9,$0;
	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+100;

	ldl		$0,S1;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1;
	ldl		$0,S1+4;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+4;

	ldl		$0,S1+8;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+8;
	ldl		$0,S1+12;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+12;

	ldl		$0,S1+16;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+16;
	ldl		$0,S1+20;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+20;

	ldl		$0,S1+24;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+24;
	ldl		$0,S1+28;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+28;

	ldl		$0,S1+32;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+32;
	ldl		$0,S1+36;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+36;

	ldl		$0,S1+40;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+40;
	ldl		$0,S1+44;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+44;

	ldl		$0,S1+48;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+48;
	ldl		$0,S1+52;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+52;

	ldl		$0,S1+56;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+56;
	ldl		$0,S1+60;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+60;

	ldl		$0,S1+64;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+64;
	ldl		$0,S1+68;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+68;

	ldl		$0,S1+72;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+72;
	ldl		$0,S1+76;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+76;

	ldl		$0,S1+80;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+80;
	ldl		$0,S1+84;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+84;

	ldl		$0,S1+88;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+88;
	ldl		$0,S1+92;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+92;

	ldl		$0,S1+96;

	addl	$0,$2,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$2,$4;
	addl	$4,$3,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$3;

	stl		$1,S1+96;
	ldl		$0,S1+100;

	addl	$0,$3,$4;
	addl	$4,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$4;
	or		$1,$4,$1;
	addl	$1,$3,$4;
	addl	$4,$2,$5;
	and		$4,31,$4;
	zapnot	$5,0xF,$5;
	sll		$5,$4,$5;
	extll	$5,4,$4;
	or		$5,$4,$2;

	stl		$1,S1+100;

	ldl		$4,4($16);
	ldl		$5,($16);

	ldl		$0,S1;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+4;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+8;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+12;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;
	ldl		$0,S1+16;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+20;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+24;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+28;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+32;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+36;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+40;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+44;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+48;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+52;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+56;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+60;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+64;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+68;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+72;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+76;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+80;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+84;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+88;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+92;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$0,S1+96;

	addl	$0,$2,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$4;
	and		$5,31,$6;
	zapnot	$4,0xF,$4;
	sll		$4,$6,$4;
	extll	$4,4,$6;
	or		$4,$6,$4;
	addl	$4,$1,$4;
	addl	$1,$2,$6;
	addl	$6,$3,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$3;

	ldl		$0,S1+100;

	addl	$0,$3,$6;
	addl	$6,$1,$1;
	zapnot	$1,0xF,$1;
	sll		$1,3,$1;
	extll	$1,4,$6;
	or		$1,$6,$1;
	xor		$4,$5,$5;
	and		$4,31,$6;
	zapnot	$5,0xF,$5;
	sll		$5,$6,$5;
	extll	$5,4,$6;
	or		$5,$6,$5;
	addl	$5,$1,$5;
	addl	$1,$3,$6;
	addl	$6,$2,$7;
	and		$6,31,$6;
	zapnot	$7,0xF,$7;
	sll		$7,$6,$7;
	extll	$7,4,$6;
	or		$7,$6,$2;

	ldl		$2,12($16);
	xor		$2,$4,$2
	zapnot	$2,0xF,$2
	bgt		$2,wrongkey

	ldl		$3,8($16);
	xor		$3,$5,$3
	zapnot	$3,0xF,$3
	bgt		$3,wrongkey

	mov		1,$0
	ret
wrongkey:
	mov		$31,$0

	ret
	.end
