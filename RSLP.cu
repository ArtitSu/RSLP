#include "mex.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <chrono>
#include <cuda_profiler_api.h>

//mexcuda RSLP.cu -lcublas -lcusolver -lcudart
//nvprof --profile-from-start off "C:\Program Files\MATLAB\R2018b\bin\win64\MATLAB.exe" -r RSLPtest

#define VERBOSE 1
#define LOG_ERROR 1

FILE *logfile;

void LogCudaError(const char* title, cudaError_t ret){
    if(!LOG_ERROR) 
        return;
    fprintf(logfile, "%s = %s\n", title, cudaGetErrorName(ret));
}
void LogCudaSolverError(const char* title, cusolverStatus_t ret){
    const char *errorname;
    if(!LOG_ERROR) 
        return;
    switch(ret){
        case CUSOLVER_STATUS_SUCCESS:
            errorname = "CUSOLVER_STATUS_SUCCESS";
            break;
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            errorname = "CUSOLVER_STATUS_NOT_INITIALIZED";
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            errorname = "CUSOLVER_STATUS_ALLOC_FAILED";
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            errorname = "CUSOLVER_STATUS_INVALID_VALUE";
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            errorname = "CUSOLVER_STATUS_ARCH_MISMATCH";
            break;
        case CUSOLVER_STATUS_MAPPING_ERROR:
            errorname = "CUSOLVER_STATUS_MAPPING_ERROR";
            break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            errorname = "CUSOLVER_STATUS_EXECUTION_FAILED";
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            errorname = "CUSOLVER_STATUS_INTERNAL_ERROR";
            break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            errorname = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
            break;
        case CUSOLVER_STATUS_NOT_SUPPORTED:
            errorname = "CUSOLVER_STATUS_NOT_SUPPORTED";
            break;
        case CUSOLVER_STATUS_ZERO_PIVOT:
            errorname = "CUSOLVER_STATUS_ZERO_PIVOT";
            break;
        case CUSOLVER_STATUS_INVALID_LICENSE:
            errorname = "CUSOLVER_STATUS_INVALID_LICENSE";
            break;
        default:
            errorname = "Unknow error";
    }
    fprintf(logfile, "%s = %s\n", title, errorname);
}

__global__ void initZeroVector(double *deviceVector, int M) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < M) {
          deviceVector[i] = 0;
    }
}
__global__ void initdeviceA(double *deviceA, double * deviceAineq, double * deviceAeq, int Mineq, int Meq, int N){
    int row = blockDim.x*blockIdx.x + threadIdx.x;
    if ( row < Mineq + Meq){
        for ( int col = 0 ; col < N ; col++){
            if (row < Mineq){
                deviceA[ col * (Mineq + Meq) + row ] = deviceAineq[ col * Mineq + row ] ;
            }
            else {
                deviceA[ col * (Mineq + Meq) + row ] = deviceAeq[ col * Meq + (row - Mineq) ] ;
            }
        }
    }
}

__global__ void initIdentityGPU(double *deviceI, int Mineq, int Meq) {
    int i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < Mineq + Meq) {
        for (int a = 0 ; a < Mineq ; a++){
          if(i == a)
              deviceI[a * (Mineq + Meq) + i ] = 1.0;
          else
              deviceI[a * (Mineq + Meq) + i ] = 0.0;
        }
    }
}

__global__ void CopyCol(int i, double* deviceVector, double* deviceA, double* deviceI, int basic_number, int N){
    int a = blockDim.x*blockIdx.x + threadIdx.x;
    if( a < basic_number){
        if(i < N){
            deviceVector[a] = deviceA[basic_number*i+a]; 
        }
        else{
            deviceVector[a] = deviceI[basic_number*(i-N) + a];
        }
    }
}

__global__ void initart(double *art, int M){
    int a = blockDim.x*blockIdx.x + threadIdx.x;
    if(a < M){
        art[a] = -1.0;
    }
}

void *LUfactorization( double *deviceA, int M, int *deviceIpiv, double * d_work, int *devInfo, 
    cusolverDnHandle_t cusolverH, const int ldA ){

//LU factorization
cusolverDnDgetrf(cusolverH, M, M, deviceA, ldA, d_work, deviceIpiv, devInfo);
cudaDeviceSynchronize();

return 0;

}

double *SolveLinear(double *deviceA, double *deviceB, int M, int *deviceIpiv, double *d_work, int *devInfo, 
 cusolverDnHandle_t cusolverH, const int ldA, const int ldB, double *output) {


//solve
cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, M, 1, deviceA, ldA, deviceIpiv, deviceB, ldB, devInfo);
cudaDeviceSynchronize();

cudaMemcpy(output, deviceB, sizeof(double) * M, cudaMemcpyDeviceToHost);

return 0; 
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ){
    double *deviceC, *deviceA, *deviceMatrix, *deviceVector, *deviceVectorN, *deviceI, *deviceAeq, *deviceAineq;
    int *deviceIpiv;
    double *Aineq, *bineq, *c, *x, *Aeq, *beq;
    int N, Mineq, Meq;
    int it = 0 ;
    double* one = new double [1] ;
    one[0] = 1.0;
    double* zero = new double [1];
    zero[0] = 0;
    double value = 0 ;
    double *d_work = NULL ;
    int *devInfo = NULL;
    int lwork;
    
    char const * const errId = "parallel:gpu:CuSOLVER:InvalidInput";
    char const * const errMsg = "Invalid Input to MEX file.";


    if( (nrhs != 5) || !mxIsDouble(prhs[0]) || mxGetM(prhs[1]) != mxGetM(prhs[2]) || 
        mxGetM(prhs[3]) != mxGetM(prhs[4])){

        mexErrMsgIdAndTxt(errId, errMsg);
    
    }

    logfile = fopen("rslplog.txt","w");
    if(VERBOSE)
        fprintf(logfile, "== RSLP start ==\n");
    
    cudaProfilerStart();

    //receive data from inputs
    c =(double *)mxGetData(prhs[0]);
    Aineq =(double *)mxGetData(prhs[1]);
    bineq =(double *)mxGetData(prhs[2]);
    Aeq = (double *)mxGetData(prhs[3]);
    beq = (double *)mxGetData(prhs[4]);

    //sizes
    N = (int)mxGetN(prhs[0]); //column of A (number of variables)
    Mineq = (int)mxGetM(prhs[1]); //row of A (number of unequalities)
    Meq = (int)mxGetM(prhs[3]); //row of Aeq (number of equalities)
    int basic_number = Mineq + Meq;
    int non_basic_number = N - Meq;
    fprintf(logfile, "basic number = %d + %d = %d\n", Mineq, Meq, basic_number);
    fprintf(logfile, "non basic number = %d - %d = %d\n", N, Meq, non_basic_number);
    const int ldA = basic_number;
    const int ldB = basic_number;

    int threadsPerBlock = 512 ; 
    int blocksPerGrid = (basic_number + threadsPerBlock -1)/ threadsPerBlock;
    
    double alpha = 1.0;
    double beta = 0;

    //Malloc 
    cudaMalloc(&deviceMatrix, sizeof(double) * (basic_number) * (basic_number));
    cudaMalloc(&deviceVector, sizeof(double) * basic_number );
    cudaMalloc(&deviceVectorN, sizeof(double) * N );
    cudaMalloc(&deviceC, sizeof(double) * (N+Mineq));
    cudaMalloc(&deviceI, sizeof(double) * basic_number * Mineq);
    cudaMalloc(&deviceAineq, sizeof(double) * Mineq * N);
    cudaMalloc(&deviceAeq, sizeof(double) * Meq * N);
    cudaMalloc(&deviceA, sizeof(double) * basic_number * N);

    //Memcpy A
    //fprintf(logfile, "== Aineq ==\n");
    //for(int row = 0 ; row < Mineq ; row++){
    //    for(int col = 0 ; col < N ; col++){
    //        fprintf(logfile, "%9.9f\t", Aineq[col*Mineq + row]);
    //    }
    //    fprintf(logfile, "\n");
    //}
    //fprintf(logfile, "== Aeq ==\n");
    //for(int row = 0 ; row < Meq ; row++){
    //    for(int col = 0 ; col < N ; col++){
    //        fprintf(logfile, "%9.9f\t", Aeq[col*Meq + row]);
    //    }
    //    fprintf(logfile, "\n");
    //}
    cudaMemcpy(deviceAineq, Aineq, sizeof(double) * Mineq * N, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceAeq, Aeq, sizeof(double) * Meq * N, cudaMemcpyHostToDevice);
    initdeviceA<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceAineq, deviceAeq, Mineq, Meq, N);
    //double *_A = new double[basic_number * N];
    //cudaMemcpy(_A, deviceA, sizeof(double) * basic_number * N, cudaMemcpyDeviceToHost);
    //fprintf(logfile, "== A ==\n");
    //for(int row = 0 ; row < basic_number ; row++){
    //    for(int col = 0 ; col < N ; col++){
    //        fprintf(logfile, "%9.9f\t", _A[col*basic_number + row]);
    //    }
    //    fprintf(logfile, "\n");
    //}

    //Memcpy C
    initZeroVector<<<blocksPerGrid, threadsPerBlock>>>(deviceC, (Mineq+N));
    cudaMemcpy(deviceC, c, sizeof(double) * N, cudaMemcpyHostToDevice);

    //init Identity
    initIdentityGPU<<<blocksPerGrid, threadsPerBlock>>>(deviceI, Mineq, Meq);
    //double *I = new double[basic_number * Mineq];
    //cudaMemcpy(I, deviceI, sizeof(double) * basic_number * Mineq, cudaMemcpyDeviceToHost);
    
    //fprintf(logfile, "== I ==\n");
    //for(int row = 0 ; row < basic_number ; row++){
    //    for(int col = 0 ; col < Mineq ; col++){
    //        fprintf(logfile, "%9.9f\t", I[col*basic_number + row]);
    //    }
    //    fprintf(logfile, "\n");
    //}

    //create outputs
    plhs[0] = mxCreateDoubleScalar(value);
    plhs[1] = mxCreateNumericMatrix(N, 1, mxDOUBLE_CLASS, mxREAL);
    x = ( double *)mxGetData(plhs[1]);

    //create DeviceInformation and Permution Identity Vector
    cudaMalloc(&devInfo, sizeof(double));
    cudaMalloc(&deviceIpiv, sizeof(int) * basic_number);
    
    //setting Handle variable
    cublasHandle_t handle;
    cudaStream_t stream = NULL;
    cusolverDnHandle_t cusolverH = NULL ;
    cusolverDnCreate(&cusolverH);
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusolverDnSetStream(cusolverH, stream);
    cublasCreate(&handle);
    
    //get local work size
    cusolverDnDgetrf_bufferSize(cusolverH, basic_number, basic_number, deviceMatrix, ldA, &lwork);
    cudaMalloc((void**)&d_work, sizeof(double)*lwork);

    //tic
    auto start = std::chrono::high_resolution_clock::now();

    //create all variables for RSLP

    
    //define basic_index
    int *basic_index = new int[basic_number];
    for(int i = 0 ; i < basic_number ; i++){
        basic_index[i] = N - Meq + i;
    }
    //fprintf(logfile, "=== basic_index ===\n");
    //for(int col = 0 ; col < basic_number ; col++){
    //    fprintf(logfile, "%d\t", basic_index[col]);
    //}
    //fprintf(logfile, "\n");
    
    //define non_basic_index
    int *non_basic_index = new int[non_basic_number + 1];
    for(int i = 0; i < non_basic_number + 1 ; i++){
        non_basic_index[i] = i; 
    }
    //fprintf(logfile, "=== non_basic_index ===\n");
    //for(int col = 0 ; col < non_basic_number ; col++){
    //    fprintf(logfile, "%d\t", non_basic_index[col]);
    //}
    //fprintf(logfile, "\n");
    
    //define B
    double *B = new double[basic_number * basic_number];
    
    //define c_basic
    double* c_basic;
    cudaMalloc(&c_basic, sizeof(double) * basic_number );

    for (int i = 0 ; i < basic_number ; i++){
        CopyCol<<<blocksPerGrid, threadsPerBlock>>>( basic_index[i], deviceVector, deviceA, deviceI, basic_number, N);
        cudaMemcpy(B + i * basic_number , deviceVector, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
        cudaMemcpy(c_basic + i , deviceC + basic_index[i], sizeof(double), cudaMemcpyDeviceToDevice);
    }

    //fprintf(logfile, "== B ==\n");
    //for(int row = 0 ; row < basic_number ; row++){
    //    for(int col = 0 ; col < basic_number ; col++){
    //        fprintf(logfile, "%9.9f\t", B[col*basic_number + row]);
    //    }
    //    fprintf(logfile, "\n");
    //}

    //define x_B
    double *x_B = new double[basic_number];

    //define P and min_P
    double *P = new double[basic_number];
    double *min_P = new double[basic_number];

    //define c_d
    double c_d = 0 ;
    double min_c_d ;
    

    //define unbound and infeasible condition
    bool unbound = false;
    bool infeasible = false;

    int leave_index, enter_index;

    //check initial basic solution
    bool phaseI = false ; 

    cudaMemcpy(deviceMatrix, B, sizeof(double) * basic_number * basic_number, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector, bineq, sizeof(double) * Mineq , cudaMemcpyHostToDevice);
    cudaMemcpy(deviceVector + Mineq, beq, sizeof(double) * Meq , cudaMemcpyHostToDevice);
    LUfactorization(deviceMatrix, basic_number, deviceIpiv, d_work, devInfo, 
                    cusolverH, ldA);   
    //calculate x_B = B/b
    SolveLinear(deviceMatrix, deviceVector, basic_number, deviceIpiv, d_work, devInfo, cusolverH, ldA, ldB, x_B); 
    
    double min_x_B = 999999999999999999999999999999999.0;
    for ( int i = 0 ; i < basic_number ; i++){
        if ( x_B[i] < min_x_B ){
            min_x_B = x_B[i];
            leave_index = i;
            //fprintf(logfile, "min ratio_%d = %9.9f\n", i, ratio);
        }
    }

    //fprintf(logfile, "== x_B ==\n");
    //for(int row = 0 ; row < basic_number ; row++){
    //    fprintf(logfile, "%9.9f\n", x_B[row]);
    //}
    //fprintf(logfile, "\n");

    //artificial column
    
    double* art ;
    if( min_x_B < 0){
        phaseI = true;

        fprintf(logfile, "== phase I ==\n");

        int a = basic_index[leave_index];
        //fprintf(logfile, "a = %d\n", a);
        basic_index[leave_index] = int(basic_number + non_basic_number + 1);
        for (int i = 0 ; i < basic_number ; i++ ){
            if (basic_index[i] == basic_number + non_basic_number + 1){
                cudaMemcpy(c_basic + i, one, sizeof(double), cudaMemcpyHostToDevice);
            }
            else{
                cudaMemcpy(c_basic + i, zero, sizeof(double), cudaMemcpyHostToDevice);
            }
        }
        non_basic_index[non_basic_number] = a;
        
        cudaMalloc(&art, sizeof(double) * basic_number);

        initart<<<blocksPerGrid, threadsPerBlock>>>(art, basic_number);
        cudaMemcpy(deviceMatrix, B, sizeof(double) * basic_number * basic_number, cudaMemcpyHostToDevice);
        
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
            basic_number, 1, basic_number, 
            &alpha, deviceMatrix, basic_number,
            art, basic_number, &beta, 
            deviceVector, basic_number);
        
        cudaMemcpy(art , deviceVector, sizeof(double) *basic_number, cudaMemcpyDeviceToDevice);
        
        //double* host_art = new double[basic_number];
        //cudaMemcpy(host_art , art , sizeof(double) *basic_number, cudaMemcpyDeviceToHost);
        //
        //fprintf(logfile, "== host_art ==\n");
        //for (int i = 0; i < basic_number ;i++){
        //    fprintf(logfile, "%9.9f\n", host_art[i]);
        //}
        //fprintf(logfile, "\n");

        cudaMemcpy(B+basic_number*leave_index, art, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
        
        //fprintf(logfile, "== basic variable ==\n");
        //for (int i = 0; i < basic_number ;i++){
        //    fprintf(logfile, "%d\t", basic_index[i]);
        //}
        //fprintf(logfile, "\n");

        //fprintf(logfile, "== non-basic variable ==\n");
        //for (int i = 0; i <non_basic_number + 1 ;i++){
        //    fprintf(logfile, "%d\t", non_basic_index[i]);
        //}
        //fprintf(logfile, "\n");

        //double* host_c_basic = new double[basic_number];
        //cudaMemcpy(host_c_basic, c_basic, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
        //fprintf(logfile, "== c_basic ==\n");
        //for(int row = 0 ; row < basic_number ; row++){
        //    fprintf(logfile, "%9.9f\t", host_c_basic[row]);
        //}
        //fprintf(logfile, "\n");
    } 

    while (phaseI == true){
        //fprintf(logfile, "== iteration %d ==\n",it);
        cudaMemcpy(deviceMatrix, B, sizeof(double) * basic_number * basic_number, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceVector, bineq, sizeof(double) * Mineq , cudaMemcpyHostToDevice);
        cudaMemcpy(deviceVector + Mineq, beq, sizeof(double) * Meq , cudaMemcpyHostToDevice);
        LUfactorization(deviceMatrix, basic_number, deviceIpiv, d_work, devInfo, 
            cusolverH, ldA);
        SolveLinear(deviceMatrix, deviceVector, basic_number, deviceIpiv, d_work, devInfo, cusolverH, ldA, ldB, x_B); 
        //fprintf(logfile, "== x_B ==\n");
        //for(int row = 0 ; row < basic_index ; row++){
        //    for(int col = 0 ; col < 1 ; col++){
        //        fprintf(logfile, "%9.9f\t", x_B[col*basic_index + row]);
        //    }
        //    fprintf(logfile, "\n");
        //}
        min_c_d = 9999999999999999999999999999999999999999999999.0 ;
        //cudaMemcpy(host_c_basic, c_basic, sizeof(double) * M, cudaMemcpyDeviceToHost);
            
        //fprintf(logfile, "== c_basic ==\n");
        //for(int row = 0 ; row < M ; row++){
        //    for(int col = 0 ; col < 1 ; col++){
        //        fprintf(logfile, "%9.9f\t", host_c_basic[col*M + row]);
        //    }
        //    fprintf(logfile, "\n");
        //}
        //find entering index
        for(int i = 0 ; i < non_basic_number+1 ; i++){
            
            // === calculate P ===
            if ( non_basic_index[i] == non_basic_number+basic_number+1){
                cudaMemcpy(deviceVector, art, sizeof(double) * basic_number, cudaMemcpyDeviceToDevice);
                SolveLinear(deviceMatrix, deviceVector, basic_number, deviceIpiv, d_work, devInfo, cusolverH, ldA, ldB, P);
                //calculate c_d(i)
                cublasDdot (handle, basic_number, deviceVector, 1, c_basic, 1, &value);
                c_d = 1.0 - value ;     
                //fprintf(logfile, "c_d_%d =  %4.4f - %4.4f = %4.4f \n", i, 1.0, value, c_d);
            }
            else{
                CopyCol<<<blocksPerGrid, threadsPerBlock>>>(non_basic_index[i], deviceVector, deviceA, deviceI, basic_number, N);
                SolveLinear(deviceMatrix, deviceVector, basic_number, deviceIpiv, d_work, devInfo, cusolverH, ldA, ldB, P);
                //calculate c_d(i)
                cublasDdot (handle, basic_number, deviceVector, 1, c_basic, 1, &value);
                //fprintf(logfile, "c_basic*P =  %4.4f \n", value);
                c_d = 0.0 - value ; 
                //fprintf(logfile, "c_d_%d =  %4.4f - %4.4f = %4.4f \n", i, 0.0, value, c_d);
            }
            //fprintf(logfile, "== P ==\n");
            //for(int row = 0 ; row < M ; row++){
            //    for(int col = 0 ; col < 1 ; col++){
            //        fprintf(logfile, "%9.9f\t", P[col*M + row]);
            //    }
            //    fprintf(logfile, "\n");
            //}
            

            if (c_d <= min_c_d){
                min_c_d = c_d;
                cudaMemcpy(min_P, P, sizeof(double) * basic_number, cudaMemcpyHostToHost);
                enter_index = i;
            }
        }
        //fprintf(logfile, "min_c_d = %9.9f\n", min_c_d);
        //fprintf(logfile, "== min_P ==\n");
        //for(int row = 0 ; row < basic_index ; row++){
        //    for(int col = 0 ; col < 1 ; col++){
        //        fprintf(logfile, "%9.9f\t", min_P[col*basic_index + row]);
        //    }
        //    fprintf(logfile, "\n");
        //}
        if ( min_c_d >= 0 ){
            

            //fprintf(logfile, "=== basic variable ===\n");
            //for(int col = 0 ; col < basic_number ; col++){
            //    fprintf(logfile, "%d\t", basic_index[col]);
            //}
            //fprintf(logfile, "\n");
            //fprintf(logfile, "=== non-basic variable ===\n");
            //for(int col = 0 ; col < non_basic_number+1 ; col++){
            //    fprintf(logfile, "%d\t", non_basic_index[col]);
            //}
            //fprintf(logfile, "\n");
            
            break;
        }
        else {
            unbound = true;
            for(int i = 0; i < basic_number ; i++){
                if (min_P[i] > 0){
                    unbound = false;
                    //fprintf(logfile, "min_P[%d] > 0\n", i);
                    break;
                }
            }
            if(unbound == true){
                break;
            }
            double m = 999999999999999999999999999999999.0;

            leave_index = -1;

            for ( int i = 0 ; i < basic_number ; i++){
                if(min_P[i] > 0){
                    double ratio = x_B[i] / min_P[i];
                    if ( ratio < m && ratio > 0){
                        m = ratio;
                        leave_index = i;

                        fprintf(logfile, "min ratio_%d = %9.9f\n", i, ratio);
                    }
                    //else if (ratio == m){
                    //    fprintf(logfile, "ratio_%d = %9.9f, m = %9.9f\n", i, ratio, m);

                    //}
                }
            }
            if (leave_index == -1){
                unbound = true;
                //fprintf(logfile, "leave_index = -1\n");
            }
            
            if (unbound == true){
                break;
            }
            fprintf(logfile, "enter_index = %d\n", non_basic_index[enter_index]);
            fprintf(logfile, "leave_index = %d\n", basic_index[leave_index]);

            //update basic feasible solution
            int a = basic_index[leave_index];
            basic_index[leave_index] = non_basic_index[enter_index];

            if (non_basic_index[enter_index] == non_basic_number+basic_number+1){
                cudaMemcpy(c_basic+leave_index, one, sizeof(double) * 1, cudaMemcpyHostToDevice);
            }
            else{
                cudaMemcpy(c_basic+leave_index, zero, sizeof(double) * 1, cudaMemcpyHostToDevice);
            }

            non_basic_index[enter_index] = a;

            if (basic_index[leave_index] == non_basic_number+basic_number+1){
                cudaMemcpy(B+basic_number*leave_index, art, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
            }

            else{
                CopyCol<<<blocksPerGrid, threadsPerBlock>>>(basic_index[leave_index], deviceVector, deviceA, deviceI, basic_number, N);
                cudaMemcpy(B+basic_number*leave_index, deviceVector, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
            }

            //fprintf(logfile, "== new B ==\n");
            //for(int row = 0 ; row < M ; row++){
            //    for(int col = 0 ; col < M ; col++){
            //        fprintf(logfile, "%9.9f\t", B[col*M + row]);
            //    }
            //    fprintf(logfile, "\n");
            //}
            //fprintf(logfile, "=== basic variable ===\n");
            //for(int col = 0 ; col < M ; col++){
            //    fprintf(logfile, "%d\t", basic_index[col]);
            //}
            //fprintf(logfile, "\n");
            //fprintf(logfile, "=== non-basic variable ===\n");
            //for(int col = 0 ; col < N+1 ; col++){
            //    fprintf(logfile, "%d\t", non_basic_index[col]);
            //}
            //fprintf(logfile, "\n");
        }
        
        it++;
    }
    
    min_x_B = 999999999999999999999999999999999.0;
    for ( int i = 0 ; i < basic_number ; i++){
        if ( x_B[i] < min_x_B ){
            min_x_B = x_B[i];
            //fprintf(logfile, "min ratio_%d = %9.9f\n", i, ratio);
        }
        if(basic_index[i] == basic_number + non_basic_number + 1){
            infeasible = true;
        }
    }
    
    if( min_x_B < 0 && phaseI == true){
        infeasible = true;
    }
    if (infeasible == true){
        fprintf(logfile, "=== infeasible solution ===\n");
    }
    else{
        infeasible = false;
        fprintf(logfile, "=== phase II ===\n");
        if (phaseI == true){
            
            for (int i = 0; i < basic_number ; i++){
                cudaMemcpy(c_basic + i, deviceC + basic_index[i], sizeof(double), cudaMemcpyDeviceToDevice);
            }

            for(int i = 0; i < non_basic_number+1; i++){
                if ( non_basic_index[i] == non_basic_number+basic_number+1 ){
                    for (int j = i+1 ; j < non_basic_number+1 ; j++){
                        non_basic_index[j-1] = non_basic_index[j];
                    }
                    break;
                }
            }
        }
    }
    //fprintf(logfile, "== new B ==\n");
    //        for(int row = 0 ; row < basic_number ; row++){
    //            for(int col = 0 ; col < basic_number ; col++){
    //                fprintf(logfile, "%9.9f\t", B[col*basic_number + row]);
    //           }
    //            fprintf(logfile, "\n");
    //        }
    //fprintf(logfile, "== basic_index ==\n");
    //for(int col = 0 ; col < basic_number ; col++){
    //    fprintf(logfile, "%d\t", basic_index[col]);
    //}
    //fprintf(logfile, "\n");
    //
    //fprintf(logfile, "== non_basic_index ==\n");
    //for(int col = 0 ; col < non_basic_number+1 ; col++){
    //    fprintf(logfile, "%d\t", non_basic_index[col]);
    //}
    //fprintf(logfile, "\n");

    while(infeasible == false){
        fprintf(logfile, "== iteration %d ==\n",it);
        cudaMemcpy(deviceMatrix, B, sizeof(double) * basic_number * basic_number, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceVector, bineq, sizeof(double) * Mineq , cudaMemcpyHostToDevice);
        cudaMemcpy(deviceVector + Mineq, beq, sizeof(double) * Meq , cudaMemcpyHostToDevice);
        LUfactorization(deviceMatrix, basic_number, deviceIpiv, d_work, devInfo, 
            cusolverH, ldA);
        SolveLinear(deviceMatrix, deviceVector, basic_number, deviceIpiv, d_work, devInfo, cusolverH, ldA, ldB, x_B); 
        //fprintf(logfile, "== x_B ==\n");
        //for(int row = 0 ; row < basic_number ; row++){
        //    fprintf(logfile, "%9.9f\t", x_B[row]);
        //}
        min_c_d = 9999999999999999999999999999999999999999999999.0 ;
        
        //double *host_c_basic = new double[basic_number];

        //cudaMemcpy(host_c_basic, c_basic, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
          
        //fprintf(logfile, "== c_basic ==\n");
        //for(int row = 0 ; row < basic_number ; row++){
        //    fprintf(logfile, "%9.9f\n", host_c_basic[row]);
        //}
        //CopyCol<<<blocksPerGrid, threadsPerBlock>>>(non_basic_index[0], deviceVector, deviceA, deviceI, basic_number, N);
        //cudaMemcpy(host_c_basic, deviceVector, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
        //  
        //fprintf(logfile, "== cpyCol ==\n");
        //for(int row = 0 ; row < basic_number ; row++){
        //    fprintf(logfile, "%9.9f\n", host_c_basic[row]);
        //}
        //find entering index
        for(int i = 0 ; i < non_basic_number ; i++){
            
            // === calculate P ===
            CopyCol<<<blocksPerGrid, threadsPerBlock>>>(non_basic_index[i], deviceVector, deviceA, deviceI, basic_number, N);
            
            //fprintf(logfile, "== A_I[%d] ==\n", non_basic_index[i]);
            //for(int row = 0 ; row < basic_number ; row++){
            //    fprintf(logfile, "%9.9f\n", P[row]);
            //}

            SolveLinear(deviceMatrix, deviceVector, basic_number, deviceIpiv, d_work, devInfo, cusolverH, ldA, ldB, P);
            
            //fprintf(logfile, "== P ==\n");
            //for(int row = 0 ; row < basic_number ; row++){
            //    fprintf(logfile, "%9.9f\n", P[row]);
            //}


            //calculate c_d(i)
            cublasDdot (handle, basic_number, deviceVector, 1, c_basic, 1, &value);
            if (non_basic_index[i] < N)
            c_d = c[non_basic_index[i]] - value ;
            else c_d = 0 - value ;
            //fprintf(logfile, "c_d_%d =  %4.4f - %4.4f = %4.4f \n", i, c[non_basic_index[i]], value, c_d);
            if (c_d <= min_c_d){
                min_c_d = c_d;
                cudaMemcpy(min_P, P, sizeof(double) * basic_number, cudaMemcpyHostToHost);
                enter_index = i;
                //fprintf(logfile, "c_d_%d =  %4.4f - %4.4f = %4.4f \n", i, c[non_basic_index[i]], value, c_d);
            }
        }
        fprintf(logfile, "min_c_d = %9.9f\n", min_c_d);
        //fprintf(logfile, "== min_P ==\n");
        //for(int row = 0 ; row < basic_number ; row++){
        //    for(int col = 0 ; col < 1 ; col++){
        //        fprintf(logfile, "%9.9f\t", min_P[col*basic_number + row]);
        //    }
        //    fprintf(logfile, "\n");
        //}
        if ( min_c_d >= 0 ){
            

            //fprintf(logfile, "=== basic variable ===\n");
            //for(int col = 0 ; col < basic_number ; col++){
            //    fprintf(logfile, "%d\t", basic_index[col]);
            //}
            //fprintf(logfile, "\n");
            //fprintf(logfile, "=== non-basic variable ===\n");
            //for(int col = 0 ; col < non_basic_number ; col++){
            //    fprintf(logfile, "%d\t", non_basic_index[col]);
            //}
            //fprintf(logfile, "\n");
            
            break;
        }
        else {
            unbound = true;
            for(int i = 0; i < basic_number ; i++){
                if (min_P[i] > 0){
                    unbound = false;
                    //fprintf(logfile, "min_P[%d] > 0\n", i);
                    break;
                }
            }
            if(unbound == true){
                break;
            }
            double m = 999999999999999999999999999999999.0;

            leave_index = -1;
            for ( int i = 0 ; i < basic_number ; i++){
                if(min_P[i] > 0){
                    double ratio = x_B[i] / min_P[i];
                    if ( ratio < m && ratio > 0){
                        m = ratio;
                        leave_index = i;

                        //fprintf(logfile, "min ratio_%d = %9.9f\n", i, x_B[i], min_P[i], ratio);
                    }
                    //else if (ratio == m){
                    //    fprintf(logfile, "ratio_%d = %9.9f, m = %9.9f\n", i, ratio, m);
                    //    repeat++;
                    //}
                }
            }
            if (leave_index == -1){
                unbound = true;
                //fprintf(logfile, "leave_index = -1\n");
            }

            if (unbound == true){
                break;
            }
            fprintf(logfile, "min ratio = %9.9f\n",m);
            fprintf(logfile, "enter_index = %d\n", non_basic_index[enter_index]);
            fprintf(logfile, "leave_index = %d\n", basic_index[leave_index]);

            //update basic feasible solution
            int a = basic_index[leave_index];
            basic_index[leave_index] = non_basic_index[enter_index];
            cudaMemcpy(c_basic+leave_index, deviceC + non_basic_index[enter_index], sizeof(double) * 1, cudaMemcpyHostToDevice);
            non_basic_index[enter_index] = a;
            CopyCol<<<blocksPerGrid, threadsPerBlock>>>(basic_index[leave_index], deviceVector, deviceA, deviceI, basic_number, N);
            cudaMemcpy(B+basic_number*leave_index, deviceVector, sizeof(double) * basic_number, cudaMemcpyDeviceToHost);
            //fprintf(logfile, "== new B ==\n");
            //for(int row = 0 ; row < basic_number ; row++){
            //    for(int col = 0 ; col < basic_number ; col++){
            //        fprintf(logfile, "%9.9f\t", B[col*basic_number + row]);
            //    }
            //    fprintf(logfile, "\n");
            //}
            //fprintf(logfile, "=== basic variable ===\n");
            //for(int col = 0 ; col < basic_number ; col++){
            //    fprintf(logfile, "%d\t", basic_index[col]);
            //}
            //fprintf(logfile, "\n");
            //fprintf(logfile, "=== non-basic variable ===\n");
            //for(int col = 0 ; col < non_basic_number ; col++){
            //    fprintf(logfile, "%d\t", non_basic_index[col]);
            //}
            //fprintf(logfile, "\n");
        }
        
        it++;
    }
    if(infeasible == true){
        
        value = 0.0;
        //plhs[0] = mxCreateDoubleScalar(value);
    }
    else{
        if (unbound == false){
            fprintf(logfile, "=== optimal solution ===\n");
            for(int i = 0; i < basic_number ; i++) {
                if(basic_index[i] < N){
                    x[basic_index[i]] = x_B[i];
                }
            }
            cudaMemcpy(deviceVectorN, x, sizeof(double) * N, cudaMemcpyHostToDevice);
            cublasDdot (handle, N, deviceVectorN, 1, deviceC, 1, &value);

            plhs[0] = mxCreateDoubleScalar(value);   
        }
        else{
            fprintf(logfile, "=== unbounded solution ===\n");
            value = 0.0;
            //plhs[0] = mxCreateDoubleScalar(value);
        }
    }

    //toc
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    fprintf(logfile, "Runtime = %lf \n", elapsed.count());
    
    fprintf(logfile, "minimum value = %9.9f\n", value);
    fprintf(logfile, "iteration = %d\n", it);

    
    delete[] one;
    delete[] zero;
    delete[] basic_index;
    delete[] non_basic_index;
    delete[] B;
    delete[] x_B;
    delete[] min_P;
    delete[] P;

    cublasDestroy(handle);
    cusolverDnDestroy(cusolverH);
    cudaStreamDestroy(stream);
    if(phaseI == true)cudaFree(art);
    cudaFree(devInfo);
    cudaFree(d_work);
    cudaFree(deviceIpiv);
    cudaFree(deviceMatrix);
    cudaFree(deviceVector);
    cudaFree(deviceVectorN);
    cudaFree(deviceC);
    cudaFree(deviceI);
    cudaFree(deviceAineq);
    cudaFree(deviceAeq);
    cudaFree(deviceA);
    cudaFree(c_basic);
    
    cudaDeviceReset();
    cudaProfilerStop();

    if(VERBOSE)
        fprintf(logfile, "== RSLP finished ==\n");
    fclose(logfile);
}