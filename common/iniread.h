/*
 * .ini (configuration file ala windows) file read/write routines
 * written by Cyrus Patel <cyp@fb14.uni-mainz.de>, no copyright.
 *
*/
#ifndef __INIREAD_H__
#define __INIREAD_H__ "@(#)$Id: iniread.h,v 1.36 2000/06/02 06:24:56 jlawson Exp $"

unsigned long GetPrivateProfileStringB( const char *sect, const char *key, 
                                    const char *defval, char *buffer, 
                                    unsigned long buffsize, 
                                    const char *filename );
int WritePrivateProfileStringB( const char *sect, const char *key, 
                                    const char *value, const char *filename );
unsigned int GetPrivateProfileIntB( const char *sect, const char *key, 
                                    int defvalue, const char *filename );
int WritePrivateProfileIntB( const char *sect, const char *key, 
                                     int value, const char *filename );

#endif /* ifndef __INIREAD_H__ */
