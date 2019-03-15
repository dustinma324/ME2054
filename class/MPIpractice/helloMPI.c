#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h> /* need it for MPI functions */

const int MAX_STRING = 100;

int main(int argc, char* argv[]) {

  if (argc < 2) {
       perror("Command-line usage: executableName < integer number >");
       exit(1);
  }

  char greeting[MAX_STRING];
  int nProcs; /* number of processes */
  int myRank; /* process rank */
  int i;
  
  //MPI_Init(NULL, NULL);                 /* initialize MPI. if main(void) */
  MPI_Init(&argc, &argv);                 /* initialize MPI */
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs); /* get the number of processes */
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank); /* get the rank of a process */

  int input = atoi(argv[1]);

  if (myRank != 0) {
    /* print a message to a string */
    sprintf(greeting, "Hello from process %d of %d ! The number is: %d", myRank, nProcs, input);

    /* send the string to process 0 */
    MPI_Send(greeting, strlen(greeting) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    /* strlen(greeting) + 1 --> +1 for the \0 character that terminates a C string */

  } else {

    /* print message for process 0 */
    printf("Hello from process %d of %d ! The number is: %d\n", myRank, nProcs, input);

    /* receive messages from other processes and print them */
    for (i = 1; i < nProcs; i++) {

      MPI_Recv(greeting, MAX_STRING, MPI_CHAR, i, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      printf("%s\n", greeting);
    }
  }
  printf("length: %d\n",strlen(greeting));

  MPI_Finalize(); /* don't forget to finalize MPI */

  return EXIT_SUCCESS;
}
