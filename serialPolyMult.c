#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int modBy = 103; // common prime num used for modding coefficient values during generation, multiplication, and addition

// function prototypes
void genPolynomials(int *polyA, int *polyB, int size);
void multPolynomialsSerial(int *polyA, int *polyB, int polySize, int *product, int degreeOfProduct);

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

    // initialize memory blocks to store each polynomial of size numTerms
    int *polyA, *polyB;
    polyA = (int *) malloc(numTerms * sizeof(int));
    polyB = (int *) malloc(numTerms * sizeof(int));

    // generate polynomials by filling elements of memory blocks
    genPolynomials(polyA, polyB, numTerms);

    printf("polyA:\n");
    for (int i = 0; i < numTerms; i++) {
        printf("%dx^%d ", polyA[i], i);
        if (i != numTerms-1) {
            printf("+ ");
        }
    }

    printf("\n\npolyB:\n");
    for (int i = 0; i < numTerms; i++) {
        printf("%dx^%d ", polyB[i], i);
        if (i != numTerms-1) {
            printf("+ ");
        }
    }

    printf("\n\n");

    // determine degree of product
    int degreeOfProduct = (numTerms - 1) * 2; // e.g. degree(polyA, polyB) = 3 then x^3 * x^3 = x^6 and degree = numTerms - 1

    // allocate new array for storing the product with size degreeOfProduct + 1
    int *product;
    product = (int *) malloc((degreeOfProduct+1) * sizeof(int));

    // multiply polynomials
    multPolynomialsSerial(polyA, polyB, numTerms, product, degreeOfProduct + 1);

    // print result
    printf("resulting product of polynomials after applying mod is:\n");
    for (int i = 0; i < degreeOfProduct+1; i++) {
        printf("%dx^%d ", product[i], i);
        if (i != degreeOfProduct) {
            printf("+ ");
        }
    }

    // free storage!
    free(polyA);
    free(polyB);
    free(product);

    return 1;
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
        if (polyB == 0) {
            polyB[i] = 1;
        }
    }
}

// multPolynomialsSerial takes two polynomials and their size, in addition to a 
// product memory block to place the sum of products into, as well as the size of the product polynomial
void multPolynomialsSerial(int *polyA, int *polyB, int polySize, int *product, int productSize) {
    int degreeOfTerms;

    // set all coefficients to 0
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

