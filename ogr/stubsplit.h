/*
 * stubsplit.h
 *
 * This function takes an OGR stub of length N and splits it into
 * multiple stubs of length N+1. For each new stub, the callback
 * function is called. If the callback function returns 0, enumeration
 * is stopped immediately.
 */

#include "stub.h"

typedef int (*stub_callback)(void *userdata, struct stub *stub);

int stub_split(struct stub *stub, stub_callback callback, void *userdata);
