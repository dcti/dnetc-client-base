/*
 * stubsplit.h
 *
 * This function takes an OGR stub of length N and splits it into
 * multiple stubs of length N+1. For each new stub, the callback
 * function is called. If the callback function returns 0, enumeration
 * is stopped immediately.
 */

#include "ogr.h"

typedef int (*stub_callback)(void *userdata, struct Stub *stub);

int stub_split(struct Stub *stub, stub_callback callback, void *userdata);
