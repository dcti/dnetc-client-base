// Copyright distributed.net 1997-1999 - All Rights Reserved
// For use in distributed.net projects only.
// Any other distribution or use of this source violates copyright.
//

#if (!defined(lint) && defined(__showids__))
const char *base64_cpp(void) {
return "@(#)$Id: base64.cpp,v 1.1.2.1 1999/04/04 09:44:33 jlawson Exp $"; }
#endif

static unsigned char base64table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef"
                                     "ghijklmnopqrstuvwxyz0123456789+/";

int base64_encode(char *outbuf, const char *inbuf )
{ /* outbuff must be at least (strlen(inbuf))*4/3) bytes */

  unsigned int length = 0;
  while (inbuf[length]) 
    length++;           
  
  #define B64_ENC(Ch) (char) (base64table[(char)(Ch) & 63])

  for (; length > 2; length -= 3, inbuf += 3)
    {
    *outbuf++ = B64_ENC(inbuf[0] >> 2);
    *outbuf++ = B64_ENC(((inbuf[0] << 4) & 060) | ((inbuf[1] >> 4) & 017));
    *outbuf++ = B64_ENC(((inbuf[1] << 2) & 074) | ((inbuf[2] >> 6) & 03));
    *outbuf++ = B64_ENC(inbuf[2] & 077);
    }
  if (length == 1)
    {
    *outbuf++ = B64_ENC(inbuf[0] >> 2);
    *outbuf++ = B64_ENC((inbuf[0] << 4) & 060);
    *outbuf++ = '=';
    *outbuf++ = '=';
    }
  else if (length == 2)
    {
    *outbuf++ = B64_ENC(inbuf[0] >> 2);
    *outbuf++ = B64_ENC(((inbuf[0] << 4) & 060) | ((inbuf[1] >> 4) & 017));
    *outbuf++ = B64_ENC((inbuf[1] << 2) & 074);
    *outbuf++ = '=';
    }
  *outbuf = 0;

  return 0;
}


int base64_decode(char *outbuf, const char *inbuf )
{
  static char inalphabet[256], decoder[256];
  int i, bits, c, char_count, errors = 0;

  for (i = (64/*sizeof(base64table)*/-1); i >= 0 ; i--) 
    {
    inalphabet[base64table[i]] = 1;
    decoder[base64table[i]] = ((unsigned char)(i));
    }
  char_count = 0;
  bits = 0;
  while ((c = *inbuf++) != 0) 
    {
    if (c == '=')
      {
      switch (char_count) 
        {
        case 1:
          //base64 encoding incomplete: at least 2 bits missing
          errors++;
          break;
        case 2:
          *outbuf++ = (char)((bits >> 10));
          break;
        case 3:
          *outbuf++ = (char)((bits >> 16));
          *outbuf++ = (char)(((bits >> 8) & 0xff));
          break;
        }
      break;
      }
    if (c > 255 || ! inalphabet[c])
      continue;
    bits += decoder[c];
    char_count++;
    if (char_count == 4) 
      {
      *outbuf++ = (char)((bits >> 16));
      *outbuf++ = (char)(((bits >> 8) & 0xff));
      *outbuf++ = (char)((bits & 0xff));
      bits = 0;
      char_count = 0;
      }
    else 
      {
      bits <<= 6;
      }
    }
  if (c == 0 && char_count) 
    {
    //base64 encoding incomplete: at least ((4 - char_count) * 6)) bits truncated
    errors++;
    }
  return ((errors) ? (-1) : (0));
}

