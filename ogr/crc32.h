extern unsigned crc32tab[];

#define CRC32(crc,c) (crc32tab[((unsigned char)(crc) ^ (c)) & 0xff] ^ (((crc) >> 8) & 0x00FFFFFFl))
