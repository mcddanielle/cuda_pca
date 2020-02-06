// C/C++ example for the CUBLAS (NVIDIA)
// implementation of NIPALS-PCA algorithm
//
// M. Andrecut (c) 2008
//
//to compile
//nvcc -O3 nipals_pca.c -lgsl -lgslcblas -lm -lcublas
//
// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// includes, cuda
#include <cublas_v2.h>  //DM deprecated

// matrix indexing convention
#define id(m, n, ld) (((n) * (ld) + (m)))

// declarations
int nipals_cublas(int, int, int,
		  double *, double *, double *);

int print_results(int, int, int,
		  double *, double *, double *, double *);

// main

int main(int argc, char** argv) {

  // PCA model: X = T * P’ + R
  
  // input: X, MxN matrix (data)
  // input: M = number of rows in X
  // input: N = number of columns in X
  // input: K = number of components (K<=N)
  
  // output: T, MxK scores matrix
  // output: P, NxN loads matrix
  // output: R, MxN residual matrix

  int M = 1000, m;
  int N = M/2, n;
  int K = 25;
  
  printf("\nProblem dimensions: MxN=%dx%d, K=%d\n", M, N, K);

  // initialize srand and clock
  srand(time(NULL));
  clock_t start=clock();
  double dtime;

  // initialize cublas
  //cublasStatus status;  //dm replace deprecated
  cublasStatus_t status;
  status = cublasInit();
  
  if(status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }
  
  // initiallize some random test data X
  double *X;
  X = (double*)malloc(M*N * sizeof(X[0]));

  if(X == 0) {
    fprintf(stderr, "! host memory allocation error: X\n");
    return EXIT_FAILURE;
  }
  
  for(m = 0; m < M; m++) {
    for(n = 0; n < N; n++) {
      X[id(m, n, M)] = rand() / (double)RAND_MAX;
    }
  }

  // allocate host memory for T, P, R

  double *T;
  T = (double*)malloc(M*K * sizeof(T[0]));;
  
  if(T == 0) {
    fprintf(stderr, "! host memory allocation error: T\n");
    return EXIT_FAILURE;
  }
  
  double *P;
  P = (double*)malloc(N*K * sizeof(P[0]));;
  
  if(P == 0) { fprintf(stderr, "! host memory allocation error: P\n");
    return EXIT_FAILURE;
  }
  
  double *R;
  R = (double*)malloc(M*N * sizeof(R[0]));;

  if(R == 0) {
    fprintf(stderr, "! host memory allocation error: R\n");
    return EXIT_FAILURE;
  }
  
  dtime = ((double)clock() - start)/CLOCKS_PER_SEC;
  printf("\nTime for data allocation: %f\n", dtime);

  // call nipals_cublas()
  start=clock();
  memcpy(R, X, M*N * sizeof(X[0]));
  nipals_cublas(M, N, K, T, P, R);
  dtime = ((double)clock() - start)/CLOCKS_PER_SEC;

  printf("\nTime for NIPALS-PCA computation on device: %f\n", dtime);
  print_results(M, N, K, X, T, P, R);

  // memory clean up
  free(R);
  free(P);
  free(T);
  free(X);

  // shutdown
  status = cublasShutdown();

  if(status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "! cublas shutdown error\n"); return EXIT_FAILURE;
  }
  
  if(argc <= 1 || strcmp(argv[1], "-noprompt")) {
    printf("\nPress ENTER to exit...\n"); getchar();
  }
  return EXIT_SUCCESS;
}

int nipals_cublas(int M, int N, int K, double *T, double *P, double *R)
{
  // PCA model: X = T * P’ + R
  
  // input: X, MxN matrix (data)
  // input: M = number of rows in X
  // input: N = number of columns in X (N<=M)
  // input: K = number of components (K<N)

  // output: T, MxK scores matrix
  // output: P, NxK loads matrix
  // output: R, MxN residual matrix
  
  // CUBLAS error handling
  //cublasStatus status;   //dm replace deprecated
  cublasStatus_t status;

  // maximum number of iterations
  int J = 10000;

  // max error
  double er = 1.0e-7;

  int k, n, j;

  // transfer the host matrix X to device matrix dR
  double *dR = 0;

  status = cublasAlloc(M*N, sizeof(dR[0]), (void**)&dR);
  
  if(status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "! device memory allocation error (dR)\n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetMatrix(M, N, sizeof(R[0]), R, M, dR, M);
  
  if(status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "! device access error (write dR)\n");
    return EXIT_FAILURE; }
    
  // allocate device memory for T, P
  double *dT = 0;
  
  status = cublasAlloc(M*K, sizeof(dT[0]), (void**)&dT);

  if(status != CUBLAS_STATUS_SUCCESS){
    fprintf(stderr, "! device memory allocation error (dT)\n");
    return EXIT_FAILURE;
  }
  
  double *dP = 0;
  status = cublasAlloc(N*K, sizeof(dP[0]), (void**)&dP);
  
  if(status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "! device memory allocation error (dP)\n");
    return EXIT_FAILURE;
  }
  
  // mean center the data

  double *dU = 0;

  status = cublasAlloc(M, sizeof(dU[0]), (void**)&dU);
  
  if(status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "! device memory allocation error (dU)\n");
    return EXIT_FAILURE;
  }
  cublasDcopy(M, &dR[0], 1, dU, 1);
  
  for(n=1; n<N; n++) {
    cublasDaxpy(M, 1.0, &dR[n*M], 1, dU, 1);
  }
  
  for(n=0; n<N; n++) {
    cublasDaxpy(M, -1.0/N, dU, 1, &dR[n*M], 1);
  }

  double a, b;
  for(k=0; k<K; k++) {
    
    cublasDcopy(M, &dR[k*M], 1, &dT[k*M], 1);
    
    a = 0.0;

    for(j=0; j<J; j++) {
      
      cublasDgemv('t', M, N, 1.0, dR, M, &dT[k*M], 1, 0.0, &dP[k*N], 1);
      
      cublasDscal(N, 1.0/cublasDnrm2(N, &dP[k*N], 1), &dP[k*N], 1);
      
      cublasDgemv('n', M, N, 1.0, dR, M, &dP[k*N], 1, 0.0, &dT[k*M], 1);

      b = cublasDnrm2(M, &dT[k*M], 1);
      
      if(fabs(a - b) < er*b) break;
      
      a = b;
    }
    
    cublasDger(M, N, -1.0, &dT[k*M], 1, &dP[k*N], 1, dR, M);
    
  }
  
  // transfer device dT to host T
  cublasGetMatrix(M, K, sizeof(dT[0]), dT, M, T, M);

  // transfer device dP to host P
  cublasGetMatrix(N, K, sizeof(dP[0]), dP, N, P, N);

  // transfer device dR to host R
  cublasGetMatrix(M, N, sizeof(dR[0]), dR, M, R, M);

  // clean up memory
  status = cublasFree(dP);
  status = cublasFree(dT);
  status = cublasFree(dR);
  
  return EXIT_SUCCESS;
}

int print_results(int M, int N, int K,
		  double *X, double *T, double *P, double *R)
{
  int m, n, k;

  // If M < 13 print the results on screen

  if(M > 12)
    return EXIT_SUCCESS;

  printf("\nX\n");
  
  for(m=0; m<M; m++) {
    for(n=0; n<N; n++) {
      printf("%+f ", X[id( m, n,M)]);
    }
    printf("\n");
  }
  printf("\nT\n");

  
  for(m=0; m<M; m++) {
    for(n=0; n<K; n++) {
      printf("%+f ", T[id(m, n, M)]);
    }
    printf("\n");
  }
  
  double a;

  printf("\nT’ * T\n");
  
  for(m = 0; m<K; m++) {
    for(n=0; n<K; n++) {
      a=0;

      for(k=0; k<M; k++) {
	a = a + T[id(k, m, M)] * T[id(k, n, M)];
      }
      printf("%+f ", a);
    }
    printf("\n");
  }
  printf("\nP\n");

  for(m=0; m<N; m++) {
    for(n=0; n<K; n++) {
      printf("%+f ", P[id(m, n, N)]);
    }
    printf("\n");
  }
  printf("\nP’ * P\n");
  
  for(m = 0; m<K; m++) {
    for(n=0; n<K; n++) {
      a=0;
      
      for(k=0; k<N; k++) {
	a = a + P[id(k, m, N)] * P[id(k, n, N)];
      }
      
      printf("%+f ", a);
    }
    printf("\n");
  }
  printf("\nR\n");

  for(m=0; m<M; m++) {
    for(n=0; n<N; n++) {
      printf("%+f ", R[id( m, n,M)]);
    }
    printf("\n");
  }
  return EXIT_SUCCESS;
}
