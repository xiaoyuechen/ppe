#ifndef gettimeofday_h
#define gettimeofday_h

typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval *tp, struct timezone *tzp);

#endif