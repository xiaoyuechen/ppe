

#define WIN32_LEAN_AND_MEAN
#include "gettimeofday.h"
#include <sys/time.h>

/* FILETIME of Jan 1 1970 00:00:00. */
static const unsigned __int64 epoch = ((unsigned __int64) 116444736000000000ULL);

/*
 * timezone information is stored outside the kernel so tzp isn't used anymore.
 */
int
gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    gettimeofday(tp, NULL); // get current time
    return 0;
}