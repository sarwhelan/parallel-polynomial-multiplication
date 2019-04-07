/*
    CS 4402 Distributed and Parallel Systems
    Assignment 2 Question 2: N^2/t thread blocks with t threads each, where t ∈ {64, 128, 256, 512}
    Sarah Whelan 250778849

    TO RUN: nvcc q2_swhela2.cu -o q2_swhela2
            ./q2_swhela2
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int modBy = 103; // common prime num used for modding coefficient values during generation, multiplication, and addition

void genPolynomials(int *polyA, int *polyB, int size);
void multPolynomialsSerial(int *polyA, int *polyB, int polySize, int *product, int productSize);
__global__ void multPolynomialsParallel(int *polyA, int *polyB, int *product, int polySize, int modBy, int numBlocks);
__global__ void sumProductsParallel(int prodSize, int threadsPerBlock, int *summedProduct, int *products, int numBlocks, int modBy);
void checkCUDAError(const char* msg);

int main() {
    srand(time(NULL));
    int numTerms;

    // get user desired input on length of polynomials
    printf("Specify the number of terms in the polynomial by specifying the exponent on base 2, UP TO 10, e.g. enter '3' if you want 2^3 terms (AKA 8 terms) per polynomial: ");
    scanf("%d", &numTerms);

    printf("\nYou entered '%d'.\n", numTerms);
    if (numTerms > 10) {
        printf("Invalid entry. The maximum number of terms is 2^10. Please enter a term less than or equal to 10 next time.");
        return 1;
    }
    
    // then bitshift by input value to determine actual value of numTerms
    numTerms = 1 << numTerms;
    printf("Number of terms per polynomial = %d, hence each polynomial will have degree = %d.\n\n", numTerms, numTerms-1);

    int threadsPerBlock;
    printf("Specify the number of threads per thread block as one of {64, 128, 256, 512}: ");
    scanf("%d", &threadsPerBlock);
    // if (!(threadsPerBlock == 64 || threadsPerBlock == 128 || threadsPerBlock == 256 || threadsPerBlock == 512)) {
    //     printf("Invalid entry. Number of threads must be one of {64, 128, 256, 512}.");
    //     return 1;
    // }

    // calculate number of blocks: n^2 / t
    int blocks = (numTerms * numTerms) / threadsPerBlock;

    // instantiate and allocate host memory blocks to store each polynomial of size numTerms
    int *host_polyA, *host_polyB;
    host_polyA = (int *) malloc(numTerms * sizeof(int));
    host_polyB = (int *) malloc(numTerms * sizeof(int));

    // generate random polynomials of size numTerms
    printf("Generating polynomials...\n\n");
    genPolynomials(host_polyA, host_polyB, numTerms);

    printf("polyA:\n");
    for (int i = 0; i < numTerms; i++) {
        printf("%dx^%d ", host_polyA[i], i);
        if (i != numTerms-1) {
            printf("+ ");
        }
    }

    printf("\n\npolyB:\n");
    for (int i = 0; i < numTerms; i++) {
        printf("%dx^%d ", host_polyB[i], i);
        if (i != numTerms-1) {
            printf("+ ");
        }
    }

    printf("\n\n");

    // determine degree of product
    int degreeOfProduct = (numTerms - 1) * 2; // e.g. degree(polyA, polyB) = 3 then x^3 * x^3 = x^6 and degree = numTerms - 1

    // allocate blocks of memory on the host for storing the product with size degreeOfProduct + 1 (serial)
    // and numTerms*numTerms for the intermediary parallel product, as well asthe final parallel product
    // two different allocations in order to verify results at the end!
    int *host_product_serial, *host_product_parallel, *host_final_product;
    host_product_serial = (int *) malloc((degreeOfProduct+1) * sizeof(int)); // sum of products is intrinsic
    host_product_parallel = (int *) malloc(numTerms * numTerms * sizeof(int)); // because of n threads in each n thread blocks
    host_final_product = (int *) malloc((degreeOfProduct+1) * sizeof(int)); // final product from parallel version once summed

    // ensure all vals in host_product_parallel are 0 (this is done within the serial version so we don't need to worry about that one)
    for (int i = 0; i < numTerms*numTerms; i++) {
        host_product_parallel[i] = 0;
    }
    // ensure all vals in host_final_product are 0
    for (int i = 0; i < degreeOfProduct+1; i++) {
        host_final_product[i] = 0;
    }

    // initialize and allocate memory on the devices for storing dev_polyA, dev_polyB, and dev_product
    int *dev_polyA, *dev_polyB, *dev_product;
    cudaMalloc( (void **) &dev_polyA, numTerms * sizeof(int));
    cudaMalloc( (void **) &dev_polyB, numTerms * sizeof(int));
    cudaMalloc( (void **) &dev_product, numTerms * numTerms * sizeof(int));

    // copy polynomials: host -> device (dest, src, size, direction)
    cudaMemcpy(dev_polyA, host_polyA, numTerms * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_polyB, host_polyB, numTerms * sizeof(int), cudaMemcpyHostToDevice); 
    cudaMemcpy(dev_product, host_product_parallel, numTerms * numTerms * sizeof(int), cudaMemcpyHostToDevice);

    // setup kernel params & launch
    dim3 dimGrid(blocks);
    dim3 dimBlock(threadsPerBlock);
    multPolynomialsParallel<<<dimGrid, dimBlock>>>(dev_polyA, dev_polyB, dev_product, numTerms, modBy, blocks);

    cudaThreadSynchronize(); // wait for ALL threads from all blocks to complete
    checkCUDAError("kernel invocation");

    // copy dev_product back into host_product_parallel (dest, src, size, direction)
    cudaMemcpy(host_product_parallel, dev_product, numTerms * numTerms * sizeof(int), cudaMemcpyDeviceToHost);
    
    /* ~~~ now we need to deal with the summation of intermediary products ~~~ */

    // allocate device mem for final product
    int *dev_final;
    cudaMalloc( (void **) &dev_final, (degreeOfProduct+1) * sizeof(int));

    // copy zero'd host_final_product to dev_final and host_product_parallel to dev_product
    // (dest, src, size, direction)
    cudaMemcpy(dev_final, host_final_product, (degreeOfProduct+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_product, host_product_parallel, numTerms * numTerms * sizeof(int), cudaMemcpyHostToDevice);

    // parameters are (int prodSize, int threadsPerBlock, int *summedProduct, int *products, int numBlocks, int modBy)
    sumProductsParallel<<<dimGrid, dimBlock>>>(degreeOfProduct+1, threadsPerBlock, dev_final, dev_product, blocks, modBy);

    cudaThreadSynchronize(); // wait for ALL threads from all blocks to complete
    checkCUDAError("kernel invocation");

    // copy summation of products back to host (dest, src, size, direction)
    cudaMemcpy(host_final_product, dev_final, (degreeOfProduct+1) * sizeof(int), cudaMemcpyDeviceToHost);

    // multiply polynomials in serial and write to host_product_serial for verification
    multPolynomialsSerial(host_polyA, host_polyB, numTerms, host_product_serial, degreeOfProduct+1);

    printf("Serial result:\n");
    for (int i = 0; i < degreeOfProduct+1; i++) {
        printf("%dx^%d ", host_product_serial[i], i);
        if (i != degreeOfProduct) {
            printf("+ ");
        }
    }
    printf("\n\nParallel result:\n");
    for (int i = 0; i < degreeOfProduct+1; i++) {
        printf("%dx^%d ", host_final_product[i], i);
        if (i != degreeOfProduct) {
            printf("+ ");
        }
    }
    printf("\n\n");
    bool allRight = 1;
    for (int i = 0; i < degreeOfProduct+1; i++) {
        if (host_product_serial[i] == host_final_product[i]) {
            continue;
        } else {
            printf("Coefficients at degree %d are not equivalent: serial!=parallel (%d!=%d)\n", i, host_product_serial[i], host_final_product[i]);
            allRight = 0;
        }
    }
    if (allRight) {
        printf("Verification successful. The serial and parallel polynomial multiplications produced the same result!\n\n");
    } else {
        printf("Looks like there were some discrepancies. Verification failed.\n\n");
    }

    // free host and device memory
    free(host_polyA);
    free(host_polyB);
    free(host_product_serial);
    free(host_product_parallel);
    free(host_final_product);

    cudaFree(dev_polyA);
    cudaFree(dev_polyB);
    cudaFree(dev_product);
    cudaFree(dev_final);

    return 0;
}

// genPolynomials takes two polynomials and their size (number of terms per polynomial),
// and generates random coefficients for each term mod p
void genPolynomials(int *polyA, int *polyB, int size) {

    // coefficient generation using rand mod p where p = 103
    for (int i = 0; i < size; i++) {
        polyA[i] = rand() % modBy;
        if (polyA[i] == 0) { // we don't want any zeros!!!
            polyA[i] = 1;
        }

        polyB[i] = rand() % modBy;
        if (polyB[i] == 0) {
            polyB[i] = 1;
        }
    }
}

// multPolynomialsSerial takes two polynomials and their size, in addition to a memory block to place 
// the sum of products into, as well as the size of the product polynomial
void multPolynomialsSerial(int *polyA, int *polyB, int polySize, int *product, int productSize) {
    int degreeOfTerms;

    // ensure all coefficients of product are 0
    for (int i = 0; i < productSize; i++) {
        product[i] = 0;
    }

    // calculate sum of products
    for (int a = 0; a < polySize; a++) { // iterate through terms in A
        for (int b = 0; b < polySize; b++) { // for each term in A, iterate through all terms in B
            // add degrees (indices) to determine which index this product belongs to in the product array block
            degreeOfTerms = a + b;

            // add product of terms to previous sum and mod by 103
            product[degreeOfTerms] = (product[degreeOfTerms] + polyA[a] * polyB[b]) % modBy;
        }
    }
}

// multPolynomialsParallel determines the intermediary products of the polynomial multiplication problem
__global__ void multPolynomialsParallel(int *polyA, int *polyB, int *product, int polySize, int modBy, int numBlocks) {
    int a, b, blocksPerA, blockPos;

    blocksPerA = numBlocks / polySize; // e.g. if numBlocks = 2048 and polySize = 512, 4 thread blocks will be assigned to one coefficient in A
    blockPos = blockIdx.x % blocksPerA; // i.e. is my thread block the first one assigned to A (blockPos = 0) or the 2nd (=1), 3rd (=2)?

    a = blockIdx.x / blocksPerA; // e.g. if blockId is 5, we need to access A[2]
    b = threadIdx.x + blockPos * blockDim.x;  

    printf("I am thread %d in block %d. blocksPerA is %d and my blockPos is %d. a = %d, b = %d\n", threadIdx.x, blockIdx.x, blocksPerA, blockPos, a, b);


    int myIndex = blockDim.x * blockIdx.x + threadIdx.x; // where to write this thread's product
    product[myIndex] = (polyA[a] * polyB[b]) % modBy;
}

// sumProductsParallel uses prodSize threads, each thread corresponding to a degree, to sum common terms and determine the actual product of polyA and polyB
__global__ void sumProductsParallel(int prodSize, int threadsPerBlock, int *summedProduct, int *products, int numBlocks, int modBy) {
    int responsibleFor = blockIdx.x * blockDim.x + threadIdx.x; // used to check which threads are going to be active during this step

    // I am fully aware this is gross and not efficient at all but it does the job
    if (responsibleFor < prodSize) { // e.g. 1 < 7 so thread 1 is going to be in charge of summing the x^1 terms, threads >= prodSize will be inactive for remainder
        for (int blockNum = 0; blockNum < numBlocks; blockNum++) { // loop through blocks
            for (int indexInBlock = 0; indexInBlock < threadsPerBlock; indexInBlock++) { // loop through each index per block
                int degreeOfElement = blockNum + indexInBlock; // the degree related to the coefficient stored at each products[] index is equal to the block number + the relative index in the block
                if (degreeOfElement == responsibleFor) { // if this thread is responsible for the degree we just calculated
                    int spotInProducts = blockNum * blockDim.x + indexInBlock; // get its actual index in products[]
                    summedProduct[responsibleFor] = (summedProduct[responsibleFor] + products[spotInProducts]) % modBy; // and write that value into the final summedProduct[our degree]
                }
            }
        }
    }
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

