#include <iostream>
#include <string>

#include "inverter.hpp"
#include "cuda_alg.hpp"

void printHelp()
{
    std::cout << "Usage: ./me {alg} [file]" << std::endl
              << "Algorithm options that run as a simple example currently are: inverter, cuda" << std::endl
              << "Optional second argument is for an input file." << std::endl;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printHelp();
        return 1;
    }

    auto const arg1 = std::string(argv[1]);

    if (argc == 3 && arg1 == "inverter")
    {
        inverterTimeTest(argv[2]);
    }
    else if (argc == 2 && arg1 == "cuda")
    {
        cuda_alg_test();
    }
    else if (argc == 3 && arg1 == "inverterP")
    {
        inverterTimeTest(argv[2]);
    }
    else
    {
        printHelp();
        return 2;
    }

    return 0;
}
