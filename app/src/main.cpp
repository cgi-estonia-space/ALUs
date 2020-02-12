#include <iostream>
#include <string>

#include "inverter.hpp"
#include "cuda_alg.hpp"

int main(int argc, char* argv[])
{
    auto const arg1 = std::string(argv[1]);
    if (argc == 2 && arg1 == "inverter")
    {
        inverterTimeTest();
    }
    else if (argc == 2 && arg1 == "cuda")
    {
        cuda_alg_test();
    }
    else
    {
        std::cout << "Usage: ./me {app}" << std::endl
                  << "App options that run as a simple example currently are: inverter, cuda" << std::endl;
    }

    return 0;
}
