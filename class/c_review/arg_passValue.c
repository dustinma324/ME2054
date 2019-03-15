/* File:    arg_passValue.c
* Purpose:  Illustrate argument passing in C
* Author:   Inanc Senocak
* Date:     02/24/2014
* 
* Input:    Three ints
* Output:   Values of variables at various points in the program.
*
* Compile:  Using Linux and gcc
*           gcc -g -Wall -o arg_passing1 arg_passing1.c
* Usage:    arg_passing1
*/

#include <stdio.h>

void passByValue(int x, int y, int z);

/*------------------------------------------------------------*/
int main(void) {

   int x, y, z;

   printf("Demonstrating pass by value\n");
   printf("Enter three ints\n");
   scanf("%d%d%d", &x, &y, &z);
   printf("In main before calling the function, x = %d, y = %d, z = %d\n\n", x, y, z);

   passByValue(x, y, z);

   printf("In main after calling the function, x = %d, y = %d, z = %d\n", x, y, z);

   return 0;
}  /* main */


/*------------------------------------------------------------*/
void passByValue(int xx, int yy, int zz) {

   printf("Inside the function before operations, x = %d, y = %d, z = %d\n", xx, yy, zz);

   xx = 2 + xx;
   yy = yy - 3;
   zz = 7 + zz;

   printf("Inside the function after operations, x = %d, y = %d, z = %d\n\n", xx, yy, zz);

}  /* Pass_by_value */

