/* File:     arg_passReference.c
 * Purpose:  Illustrate argument passing in C
 *
 * Input:    Three ints
 * Output:   Values of variables at various points in the program.
 *
 * Compile:  Using Linux and gcc
 *           gcc -g -Wall -o arg_passReference arg_passReference.c
 * Usage:    arg_passing2
 */

#include <stdio.h>

void passByReference(int* x_p, int* y_p, int* z_p);

/*------------------------------------------------------------*/
int main(void) {
   int x, y, z;
   printf("Demonstrating pass by reference\n");
   printf("Enter three ints\n");
   scanf("%d%d%d", &x, &y, &z);
   printf("In main before calling the function, x = %d, y = %d, z = %d\n\n", x, y, z);

   passByReference(&x, &y, &z);

   printf("In main after calling the function, x = %d, y = %d, z = %d\n", x, y, z);

   return 0;
}  /* main */


/*------------------------------------------------------------*/
void passByReference(int * x_p, int* y_p, int* z_p) {

   printf("Inside the function before the operations, x = %d, y = %d, z = %d\n", *x_p, *y_p, *z_p);

   *x_p = 2 * (*x_p);
   *y_p = *y_p - 3;
   *z_p = 7 + *z_p;

   printf("Inside the function after the operations, x = %d, y = %d, z = %d\n\n", *x_p, *y_p, *z_p);

}  /* Pass_by_reference */

