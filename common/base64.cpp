/* 
 * Copyright distributed.net 1997-1999 - All Rights Reserved
 * For use in distributed.net projects only.
 * Any other distribution or use of this source violates copyright.
 *
 * On success, both functions return the number of bytes in the outbuf (not 
 * counting the terminating '\0'). On error, both return < 0.
 * note: The outbuffer is terminated only if there is sufficient space 
 * remaining after conversion.
 *
*/ 
const char *base64_cpp(void) {
return "@(#)$Id: base64.cpp,v 1.5 1999/11/08 02:02:33 cyp Exp $"; }

static unsigned char base64table[64] = 
{
  'A','B','C','D','E','F','G','H','I','J','K','L','M',
  'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
  'a','b','c','d','e','f','g','h','i','j','k','l','m',
  'n','o','p','q','r','s','t','u','v','w','x','y','z',
  '0','1','2','3','4','5','6','7','8','9','+','/' 
};

int base64_encode(char *outbuf, const char *inbuf, 
                  unsigned int outbuflen, unsigned int inbuflen )
{ /* outbuff must be at least (strlen(inbuf))*4/3) bytes */

  unsigned int outlen = 0, errors = 0;
    
  if (outbuflen < (((inbuflen)*4)/3))
    errors++;
  else if (inbuflen)
  {
    #define B64_ENC(Ch) (char) (base64table[((unsigned char)(Ch)) & 63])
    
    for (; inbuflen > 2; inbuflen -= 3, inbuf += 3)
    {
      *outbuf++ = B64_ENC(inbuf[0] >> 2);
      *outbuf++ = B64_ENC(((inbuf[0] << 4) & 060) | ((inbuf[1] >> 4) & 017));
      *outbuf++ = B64_ENC(((inbuf[1] << 2) & 074) | ((inbuf[2] >> 6) & 03));
      *outbuf++ = B64_ENC(inbuf[2] & 077);
      outlen += 4;
    }
    if (inbuflen == 1)
    {
      *outbuf++ = B64_ENC(inbuf[0] >> 2);
      *outbuf++ = B64_ENC((inbuf[0] << 4) & 060);
      *outbuf++ = '=';
      *outbuf++ = '=';
      outlen += 4;
    }
    else if (inbuflen == 2)
    {
      *outbuf++ = B64_ENC(inbuf[0] >> 2);
      *outbuf++ = B64_ENC(((inbuf[0] << 4) & 060) | ((inbuf[1] >> 4) & 017));
      *outbuf++ = B64_ENC((inbuf[1] << 2) & 074);
      *outbuf++ = '=';
      outlen += 4;
    }
  }
  if (outlen < outbuflen)
    *outbuf = 0;

  return (errors ? -1 : (int)outlen);
}


int base64_decode(char *outbuf, const char *inbuf,
                  unsigned int outbuflen, unsigned int inbuflen )
{ /* outbuf can be same as inbuf */

  unsigned char decoder[256]; unsigned int outlen = 0, read_pos = 0;
  unsigned long bits = 0; int c, char_count = 0, errors = 0;

  for (c = 0; c < ((int)sizeof(decoder)); c++)
    decoder[c] = '\0';
  for (c = (sizeof(base64table)-1); c >= 0 ; c--) 
    decoder[base64table[c]] = ((unsigned char)(c+1));

  while (!errors)
  {
    if (read_pos == inbuflen)
    {
      if (char_count)
        errors++; /* at least ((4 - char_count) * 6)) bits truncated */
      break;
    }
    else if ((c = inbuf[read_pos++]) == '=')
    {
      if (char_count == 1)
        errors++; /* at least 2 bits missing */
      else if (char_count == 2)
      {
        if ((outbuflen - outlen) < 1)
          errors++; /* no space in outbuffer */
        else
          outbuf[outlen++] = (char)(((bits >> 10) & 0xff));
      }
      else if (char_count == 3)
      {
        if ((outbuflen - outlen) < 2)
          errors++; /* no space in outbuffer */
        else
        {
          outbuf[outlen++] = (char)(((bits >> 16) & 0xff));
          outbuf[outlen++] = (char)(((bits >> 8) & 0xff));
        }
      }
      break;
    }
    else if (c < ((int)sizeof(decoder)))
    {
      if (decoder[c])
      {
        bits += (decoder[c]-1);
        if ((++char_count) < 4)
          bits <<= 6;
        else if ((outbuflen - outlen) < 3)
          errors++;
        else
        {
          outbuf[outlen++] = (char)((bits >> 16));
          outbuf[outlen++] = (char)(((bits >> 8) & 0xff));
          outbuf[outlen++] = (char)((bits & 0xff));
          bits = 0;
          char_count = 0;
        }
      }
    }
  }

  if (outlen < outbuflen)
    outbuf[outlen] = '\0';

  return (errors ? -1 : (int)outlen);
}  
