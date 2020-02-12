#include "cuda_alg.hpp"

#include <iostream>

void cuda_kernel();

int cuda_alg_test()
{
    std::cout << "Running CUDA alg test" << std::endl;

    cuda_kernel();

    return 1;
}