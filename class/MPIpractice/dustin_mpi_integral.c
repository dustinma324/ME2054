#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> /* need it for MPI functions */

const int MAX_STRING = 100;

#typedef REAL float
#define N 10

float trapezoid(REAL const a, REAL const b, REAL const n){
	
	REAL h = (b-a)/N;

	approx = (f(a)+f(b)/2.0f);

	for (int i = 1; i < = N-1; i++){
		x_i = a + i*h;
		approx += f(x_i);
	}
	return approx = h*approx;
}

float f(REAL const x){
REAL y;
return y = x * x;
}


int main(void) {

  int nProcs; /* number of processes */
  int myRank; /* process rank */
  
  MPI_Init(NULL, NULL);                 /* initialize MPI. if main(void) */
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs); /* get the number of processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank); /* get the rank of a process */

  int input = atoi(argv[1]);

  Get a,b,n;
  h = (b-a)/n;
  local_a = a + my_rank*locacl_n*h;
  local_b = local_a + local_n*h;
  local_integral = trapezoid(local_a, local_b, local_n, h);

  if (myRank != 0) {

    MPI_Send(greeting, strlen(greeting) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    /* strlen(greeting) + 1 --> +1 for the \0 character that terminates a C string */

  } else {

    /* print message for process 0 */
    printf("Hello from process %d of %d ! The number is: %d\n", myRank, nProcs, input);

    /* receive messages from other processes and print them */
    for (i = 1; i < nProcs; i++) {

      MPI_Recv(greeting, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      printf("%s\n", greeting);
    }
  }
  printf("length: %d\n",strlen(greeting));

  MPI_Finalize(); /* don't forget to finalize MPI */

  return EXIT_SUCCESS;
}
