#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

const char* dgemv_desc = "OpenMP dgemv nested loop.";

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */

void my_dgemv(int n, double* A, double* x, double* y) {

   // Set the number of threads to use
   int nthreads = omp_get_num_procs(); // Set to number of available processors
   omp_set_num_threads(nthreads);

   // Parallelize the outer loop over rows of A
   #pragma omp parallel for
   for (int i = 0; i < n; i++) {
      // Parallelize the inner loop over columns of A
      #pragma omp parallel for
      for (int j = 0; j < n; j++) {
         y[i] += A[i*n+j] * x[j];
      }
   }

}
