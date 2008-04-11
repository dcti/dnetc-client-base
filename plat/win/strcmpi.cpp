#define WIN32_LEAN_AND_MEAN
#include <string.h>
#include <windows.h>

//! Case-insensitive string comparison
/*!
 * \param s1 First input string to compare.
 * \param s2 Second input string to compare.
 * \return Returns -1 if s1 is lexographically less than s2, 1
 *      if s1 is lexographycally greater than s2, or 0 if they 
 *      are equal.
 */
int strcmpi(const char *s1, const char *s2)
{
  return lstrcmpi(s1, s2);
}
