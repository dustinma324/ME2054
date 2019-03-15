/* printf example */
#include <stdio.h>
#include <stdlib.h>

int main()
{
   printf ("Characters: %c %c \n", 'a', 65);

   printf ("Decimals: %d %ld \n", 1977, 650000L);

   printf ("Preceding with blanks: %10d \n", 1977);

   printf ("Preceding with zeros: %010d \n", 1977);

   printf ("Some different radices: %d %x %o %#x %#o \n", 100, 100, 100, 100, 100);

   printf ("floats: %4.2f %+.0e %E \n", 3.1416, 3.1416, 3.1416);

   printf ("Width trick:  %*d \n", 35, 10);  // 35 is the width argument for %*d

   printf ("%*s \n", 15, "A string"); //note the %*s

printf("========================\n");
int *ptr;
int  i = 5;
ptr = &i;
printf("i = %d, prt = %d,\n",*ptr,ptr);


   return EXIT_SUCCESS;
}
