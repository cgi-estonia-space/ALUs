/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include "command_line_options.h"

#include <array>
#include <string>

#include "gmock/gmock.h"

namespace {

using ::testing::IsFalse;
using ::testing::IsTrue;

using alus::app::CommandLineOptions;

TEST(CommandLineOptions, PrintHelpSucceeds) {
    CommandLineOptions options{};
    std::cout << options.GetHelp() << std::endl;
}

TEST(CommandLineOptions, DoRequireHelpIsTrue) {
    {
        constexpr int argc{4};  // NOLINT
        const std::array<std::string, argc> test_args{"alus", "--help", "--alg_name", "coherence"};
        const char* args[test_args.size()];

        for (int i = 0; i < argc; i++) {
            args[i] = test_args.at(i).c_str();
        }

        const CommandLineOptions opt(argc, args);
        ASSERT_THAT(opt.DoRequireHelp(), IsTrue());
    }

    {
        constexpr int argc{1};  // NOLINT
        const std::array<std::string, argc> test_args{"alus"};
        const char* args[test_args.size()];

        for (int i = 0; i < argc; i++) {
            args[i] = test_args.at(i).c_str();
        }

        const CommandLineOptions opt(argc, args);
        ASSERT_THAT(opt.DoRequireHelp(), IsTrue());
    }
}

TEST(CommandLineOptions, DoRequireHelpIsFalse) {
    constexpr int argc{3};  // NOLINT
    const std::array<std::string, argc> test_args{"alus", "--alg_name", "coherence"};
    const char* args[test_args.size()];

    for (int i = 0; i < argc; i++) {
        args[i] = test_args.at(i).c_str();
    }

    const CommandLineOptions opt(argc, args);
    ASSERT_THAT(opt.DoRequireHelp(), IsFalse());
}

}  // namespace
