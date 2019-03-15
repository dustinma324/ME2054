//gcc -std=c99 -o array_pointer.exe -lm pointer.c
//

#include <stdio.h>

int main(){
/*
int i ,*ptr;
ptr = &i;
i = 13;
printf("*ptr = %d\n",*ptr);
*ptr = 4;
printf("i = %d\n",i);
*/

/*
int a[]={2,3,7};
int *ptr,*q_p;
ptr = &a[0];
*ptr=6;
printf("a[0]= %d\n",a[0]);
*/

/*
int a[] = {2,3,7,4,2,1};

int *p = &a[0], sum = 0;

while( p < &a[6] ){
sum += *p++;
}
printf("sum = %d\n",sum);
*/

/*
int a[] = {2,5,7,-3};
int *p = a;
p[0] = 3;

printf("a[0] = %d\n",a[0]);
*(p+2) = 10;

printf("*(p+2) = a[2] = %d\n",a[2]);
*/

int a[5][6], *p;
p = a; /* or p = &a[0][0] */
a[0][1] = 4;
a[1][0] = 5;
printf("p+1 = %d\n",*(p+1));
printf("p+6 = %d\n",*(p+6));

return 0;
}
