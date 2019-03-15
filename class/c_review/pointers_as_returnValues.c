
#include<stdio.h>
#include<stdlib.h>
int *fun();
int main()
{
    int *ptr;
    ptr=fun();
    printf("%d\n",*ptr);
    return 0;
}	

int *fun()
{
    //int *point;
    int *point = malloc(1*(sizeof *point));
    
    if (point == NULL)
       printf("Memory allocation failed\n");

    *point=12;  
    return point;
}
