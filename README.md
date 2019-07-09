# parallel-polynomial-multiplication
Program to compute the product of two randomly generated univariate polynomials of the same degree, with maximum degree being 1023 (2^10).
Degree of polynomials is based on user input.

You can think of polynomial multiplication the same as normal long-multiplication; this will help you understand the steps of the computation which are easily parallelized. First you compute the intermediary products, and then you sum those intermediary products to determine the final product. [Here is a refresher](https://www.mathsisfun.com/algebra/polynomials-multiplication-long.html).



Built with CUDA C++ and executed on a NVIDIA GPU.
