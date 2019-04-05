#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int modBy = 103; // common prime num used for modding coefficient values during generation, multiplication, and addition

void genPolynomials(int *polyA, int *polyB, int size);
void multPolynomialsSerial(int *polyA, int *polyB, int polySize, int *product, int productSize);
__global__ void multPolynomialsParallel(int *polyA, int *polyB, int *product, int polySize, int modBy);
void checkCUDAError(const char* msg);

int main() {
    srand(time(NULL));
    int numTerms;

    // get user desired input on length of polynomials
    printf("Specify the number of terms in the polynomial by specifying the exponent on base 2 UP TO 10, e.g. type 3 if you want 2^3 terms per polynomial: ");
    scanf("%d", &numTerms);

    printf("Value entered is %d\n", numTerms);
    if (numTerms > 10) {
        printf("Invalid entry. The maximum number of terms is 2^10. Please enter a term less than or equal to 10 next time.");
        return 0;
    }
    
    // then bitshift by input value to determine actual value of numTerms
    numTerms = 1 << numTerms;
    printf("Number of terms per polynomial is %d, hence each polynomial has degree %d\n\n", numTerms, numTerms-1);

    // use numTerms as the number of blocks per thread and the number of blocks
    int threadsPerBlock = numTerms;
    int blocks = numTerms;

    // instantiate and allocate host memory blocks to store each polynomial of size numTerms
    int *host_polyA, *host_polyB;
    host_polyA = (int *) malloc(numTerms * sizeof(int));
    host_polyB = (int *) malloc(numTerms * sizeof(int));

    // generate random polynomials of size numTerms
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
    multPolynomialsParallel<<<dimGrid, dimBlock>>>(dev_polyA, dev_polyB, dev_product, numTerms, modBy);

    cudaThreadSynchronize(); // wait for ALL threads from all blocks to complete
    checkCUDAError("kernel invocation");

    // now copy dev_product back into host_product_parallel (dest, src, size, direction)
    cudaMemcpy(host_product_parallel, dev_product, numTerms * numTerms * sizeof(int), cudaMemcpyDeviceToHost);
    
    // multiply polynomials in serial and write to host_product_serial for verification later
    multPolynomialsSerial(host_polyA, host_polyB, numTerms, host_product_serial, degreeOfProduct + 1);

    printf("serial produced:\n");
    for (int i = 0; i < degreeOfProduct+1; i++) {
        printf("%dx^%d ", host_product_serial[i], i);
        if (i != degreeOfProduct) {
            printf("+ ");
        }
    }
    printf("\n\nintermediary parallel step:\n");
    for (int i = 0; i < numTerms*numTerms; i++) {
        printf("%dx^%d ", host_product_parallel[i], i);
        if (i != degreeOfProduct) {
            printf("+ ");
        }
        if (i%threadsPerBlock == 0) {
            printf("| ");
        }
    }
    // printf("\n\nequal??? ");
    // for (int i = 0; i < degreeOfProduct+1; i++) {
    //     if (host_product_serial[i] == host_final_product[i]) {
    //         printf("Y ");
    //     } else {
    //         printf("N ");
    //     }
    // }

    // free host and device memory
    free(host_polyA);
    free(host_polyB);
    free(host_product_serial);
    free(host_final_product);

    cudaFree(dev_polyA);
    cudaFree(dev_polyB);
    cudaFree(dev_product);

    return 0;
}

// genPolynomials takes two polynomials and their size (number of terms per polynomial),
// and generates random coefficients for each term mod p
void genPolynomials(int *polyA, int *polyB, int size) {

    // coefficient generation using rand mod p where p = 103
    for (int i = 0; i < size; i++) {
        polyA[i] = rand() % modBy;
        if (polyA[i] == 0) {
            polyA[i] = 1;
        }

        polyB[i] = rand() % modBy;
        if (polyB[i] == 0) {
            polyB[i] = 1;
        }
    }
}

// multPolynomialsSerial takes two polynomials and their size, in addition to a 
// product memory block to place the sum of products into, as well as the size of the product polynomial
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

__global__ void multPolynomialsParallel(int *polyA, int *polyB, int *product, int polySize, int modBy) {
    // int a = blockIdx.x; // n blocks means each block has a corresponding term in A
    // int b = (blockIdx.x + threadIdx.x) % polySize; // each thread assigned to term in B
    // int degreeOfTerms = a + b;
    // product[degreeOfTerms] = (product[degreeOfTerms] + polyA[a] * polyB[b]) % modBy; 

    int a = blockIdx.x;
    int b = threadIdx.x;
    int myIndex = blockDim.x * blockIdx.x + threadIdx.x;
    product[myIndex] = polyA[a] * polyB[b];
}

void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "CUDA error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

