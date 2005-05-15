/*
 * Copyright distributed.net 2004-2005 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * $Id: Support.h,v 1.1.2.2 2005/05/15 11:29:07 piru Exp $
 *
 * Created by Ilkka Lehtoranta <ilkleht@isoveli.org>
 *
 * ----------------------------------------------------------------------
 * MUI GUI module for MorphOS client - MUI application class code
 * ----------------------------------------------------------------------
*/

#ifndef	__SUPPORT_H__
#define	__SUPPORT_H__

static inline BYTE strncmp(CONST_STRPTR a, CONST_STRPTR b, ULONG length)
{
	BYTE	c	= 0;

	while (length)
	{
		c	= *a++ - *b++;
		length--;

		if (c != 0)
			break;
	}

	return c;
}

static inline ULONG strlen(const char *str)
{
	ULONG	len	= 0;

	for (;;)
	{
		if (*str++ == '\0')
			break;

		len++;
	}

	return len;
}

static inline void strcpy(char *dst, const char *src)
{
	UBYTE	c;

	do
	{
		c	= *src++;
		*dst++	= c;
	}
	while (c);
}

static inline void strncpy(char *dst, const char *src, ULONG max)
{
	UBYTE	c;

	for (;;)
	{
		max--;

		if ((max == 0) || ((c = *src++) == 0))
			break;

		*dst++	= c;
	}

	*dst	= '\0';
}

#endif	/* __SUPPORT_H__ */
