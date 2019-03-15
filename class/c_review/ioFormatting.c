/* this program demonstrates multiple formats for 
 * integer, string, and float I/O
 * using the printf function
 */

#include <stdio.h>

int main()
{
	int n;
	float f;
	double dbl;
	char s[100];
        char r;

	printf("enter an integer\n");
	scanf("%d", &n);

	printf("print an integer, no formatting\n");
	printf("%d\n", n);

        printf("Continue the exercise (y/n)?\n");
        scanf(" %c", &r);

        while (r == 'y'){

        printf("printing an integer, padded on left with spaces to total 6 chars\n");
        printf("%6d\n", n);

	printf("printing an integer, padded on right with spaces to total 6 chars\n");
	printf("%-6d\n", n);

	printf("printing an integer, padded on left with zeroes to total 6 chars\n");
	printf("%.6d\n", n);

	printf("enter a string (whitespace delineated)\n"); 
	scanf("%s", s);

	printf("print a string, no formattingi\n"); 
	printf("%s\n", s);

	printf("%* print a string, padded with spaces on left to 20 chars\n"); 
	printf("%20s\n", s);
	
        printf("print a string, padded with spaces on right to 20 chars\n"); 
	printf("%-20s\n", s);

        printf("print a string, truncated after 3 chars\n");
	printf("%3s\n", s);

	printf("enter a single precision floating point number\n"); 
	scanf("%f", &f);

	printf("print a float, default precision is 6 places\n"); 
	printf("%f\n", f);

	printf("enter a double precision floating point number\n");
	scanf("%lf", &dbl);

	printf("print a double, default precision is 6 places\n");
	printf("%f\n", dbl);

	printf("print a double, 2 places of precision \n");
	printf("%.2f\n", dbl);

	printf("print a double, 2 places of precision, padded with space to 10\n");
	printf("%10.2f\n", dbl);

	printf("print a double, use exponential notation if more than 3 digits\n");
	printf("%.3g\n", dbl);
        }
 
        return 0;
}


