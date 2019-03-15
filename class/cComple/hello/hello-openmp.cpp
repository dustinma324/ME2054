#include <iostream>

#include <omp.h>

int main( int argc, char* argv[] )
{


  int tid;     // thread id
  int nthread; // number of threads

  // begin parallel region
  #pragma omp parallel private(tid)
  {

    tid = omp_get_thread_num();
    nthread = omp_get_num_threads();
    
    #pragma omp critical
    {
      std::cout << "Hello from: " << tid << " of " << nthread << std::endl;
    }

  } // end parallel region

  return 0;

}
