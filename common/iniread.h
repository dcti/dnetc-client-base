// Hey, Emacs, this a -*-C++-*- file !
//
// $Log: iniread.h,v $
// Revision 1.17  1999/01/26 20:17:34  cyp
// new ini stuff from proxy
//
//
// 

#ifndef __INIREAD_H__
#define __INIREAD_H__

unsigned long GetPrivateProfileStringA( const char *sect, const char *key, 
                                    const char *defval, char *buffer, 
                                    unsigned long buffsize, 
                                    const char *filename );

int WritePrivateProfileStringA( const char *sect, const char *key, 
                                    const char *value, const char *filename );

unsigned int GetPrivateProfileIntA( const char *sect, const char *key, 
                                    int defvalue, const char *filename );

int WritePrivateProfileIntA( const char *sect, const char *key, 
                                     int value, const char *filename );

#endif


