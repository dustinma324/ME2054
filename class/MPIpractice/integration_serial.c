/* Numerical integration with the trapezoidal rule
 *
 * Author: Inanc Senocak
 *
 * Date: 04/15/2016
 * Last edited:
 *
 * gcc -O3 -lm -std=c99 -o integrate.exe integration_serial.c
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double f(const double x) { return x * x; } // function to integrate

double integrate(const double a, const double b, const int n) {

  double h = (b - a) / (double) n;

  double integral = (f(b) + f(a)) * 0.5f;

  for (int i = 1; i < n; i++) {

    double x = a + i * h;
    integral += f(x);
  }
  return integral * h;
}

int main(void) {

  // define the lower and upper limits of the definite integral
  double lower_limit, upper_limit;
  // define the number of intervals per processor
  int nInterval;

  printf("we are going to integrate x^2 on the interval [l,u]\n");
  printf("Enter the lower and upper limits of the definite integral:\n");
  scanf("%lf", &lower_limit); // note that scanf is reading a DOUBLE hence %lf
  scanf("%lf", &upper_limit);
  printf("Enter the number of intervals\n");
  scanf("%d", &nInterval);

  double integral = integrate(lower_limit, upper_limit, nInterval);

  double exactIntegral = pow(upper_limit, 3) / 3.f - pow(lower_limit, 3) / 3.f;
  printf("lower limit     = %f\n", lower_limit);
  printf("upper limit     = %f\n", upper_limit);
  printf("Exact     = %f\n", exactIntegral);
  printf("Numerical = %f\n", integral);

  return EXIT_SUCCESS;
}
