/*
Purpose: Demonstrate formatted I/O in C
Date and time: 12/16/2013 09:15:10 PM 
Last modified: 09/20/2018 
Author: Inanc Senocak

to compile: gcc -std=c99 -o example.exe -lm printfDemo.c

!!!!!--don't forget to link with the standard library using -lm
*/

#include <stdio.h>
#include <math.h>

typedef double REAL; /* so that I can change in one place only */
typedef int INT;

int main (void)
{
   REAL x,y;
   INT  i;
   
   printf("Enter an integer and a real number\n");
   scanf("%d %lf", &i, &y); // typical mistake is read a double with %f 
      
   x = exp( (REAL) i);  //note that I did a cast
   printf("==================\n");   
   printf("%f\n",      y);
   printf("%12.5f\n",  x);
   printf("%-5.5f\n", x);
   printf("%2.5f\n",   x);
   printf("%.12f\n",   x);
   printf("%.2g\n",   x);  //use the shortest representation %e or %f
   printf("%.8g\n",   x);  //use the shortest representation %e or %f
   printf("%.3e\n",    x/100.);
   printf("==================\n");   
   
   printf("%d\n", x);   //let's make some errors!!
   
   i = (INT) ceil(y); //note that I did a cast
   
   printf("%d\n",   i);
   printf("%6d\n",  i);
   printf("%-d\n",  i);
   printf("%06d\n", i);
   
   return 0;
}
