
#include "command_line_options.h"

#include <array>
#include <string>

#include "gmock/gmock.h"

namespace {

using namespace alus::app;
using namespace ::testing;

TEST(CommandLineOptions, PrintHelpSucceeds) {
    CommandLineOptions options{};
    std::cout << options.GetHelp() << std::endl;
}

TEST(CommandLineOptions, DoRequireHelpIsTrue) {
    {
        constexpr int argc{4};
        const std::array<std::string, argc> test_args{"alus", "--help", "--alg_name", "coherence"};
        const char* args[test_args.size()] ;

        for (int i = 0; i < argc; i++) {
            args[i] = test_args.at(i).c_str();
        }

        const CommandLineOptions opt(argc, args);
        ASSERT_THAT(opt.DoRequireHelp(), IsTrue());
    }

    {
        constexpr int argc{1};
        const std::array<std::string, argc> test_args{"alus"};
        const char* args[test_args.size()] ;

        for (int i = 0; i < argc; i++) {
            args[i] = test_args.at(i).c_str();
        }

        const CommandLineOptions opt(argc, args);
        ASSERT_THAT(opt.DoRequireHelp(), IsTrue());
    }
}

TEST(CommandLineOptions, DoRequireHelpIsFalse) {
    constexpr int argc{3};
    const std::array<std::string, argc> test_args{"alus", "--alg_name", "coherence"};
    const char* args[test_args.size()] ;

    for (int i = 0; i < argc; i++) {
        args[i] = test_args.at(i).c_str();
    }

    const CommandLineOptions opt(argc, args);
    ASSERT_THAT(opt.DoRequireHelp(), IsFalse());
}

}