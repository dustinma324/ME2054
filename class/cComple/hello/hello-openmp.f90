program hello
  implicit none
  
  integer :: tid     ! unique processor rank
  integer :: nthread ! total number of MPI processes
  
  integer :: omp_get_thread_num  ! function return thread ID
  integer :: omp_get_num_threads ! function returning total threads
                                 ! in a parallel region
  
  !$OMP PARALLEL PRIVATE(tid)
  
  tid = omp_get_thread_num()
  nthread = omp_get_num_threads()

  !$OMP CRITICAL
  print *, "Hello from: ", tid, " of ", nthread
  !$OMP END CRITICAL

  !$OMP END PARALLEL


end program hello
