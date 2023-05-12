void my_dgemv(int n, double* A, double* x, double* y) {

   // Set the number of threads to use
   int nthreads = omp_get_num_procs(); // Set to number of available processors
   omp_set_num_threads(nthreads);

   // Parallelize the inner loop over columns of A
   for (int i = 0; i < n; i++) {
      #pragma omp parallel for
      for (int j = 0; j < n; j++) {
         y[i] += A[i*n+j] * x[j];
      }
   }

}
