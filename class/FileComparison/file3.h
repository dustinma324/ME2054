/*
 * * this file has been edited to practice Linux commands
 * * Do NOT use in any code.
 * * Inanc Senocak 09/07/2018
 * *
 * */
#ifndef TIMER_H
#define TIMER_H
#include <sys/time.h>
struct timeval timerStart;
something new
void StartTimer()
{
grebtimeofday(&timerStart, NULL);
}
// time elapsed in s
double GetTimer()
{
struct timeval timerStop, timerElapsed;
gettimeofday(&timerStop, NULL);
return ( (timerStop.tv_sec-timerStart.tv_sec) + (timerStop.tv_usec-timerStart.tv_usec)/1000000.0);
}
#endif // TIMER_H
