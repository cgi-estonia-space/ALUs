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

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

#include "../../VERSION"
#include "alus_log.h"
#include "command_line_options.h"
#include "execute.h"

namespace {
void PrintHelp(const std::string& options_help) {
    std::cout << "ALUs - Gabor feature extraction" << std::endl;
    std::cout << "Version " << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
    std::cout << std::endl;
    std::cout << options_help << std::endl;
    std::cout << std::endl;
    std::cout << "https://bitbucket.org/cgi-ee-space/alus" << std::endl;
}
}  // namespace

int main(int argc, char* argv[]) {
    std::string help_string{};
    try {
        alus::common::log::Initialize();

#ifdef NDEBUG
        alus::common::log::SetLevel(alus::common::log::Level::INFO);
#endif
        alus::featurextractiongabor::Arguments args;
        help_string = args.GetHelp();
        std::vector<char*> args_raw(argc);
        for (auto i{0}; i < argc; i++) {
            args_raw.at(i) = argv[i];
        }
        args.ParseArgs(args_raw);

        if (args.IsHelpRequested()) {
            PrintHelp(help_string);
            return 0;
        }

        alus::featurextractiongabor::Execute exe(args.GetOrientationCount(), args.GetFrequencyCount(),
                                                 args.GetPatchSize(), args.GetInput());
        exe.GenerateInputs();
        if (args.IsConvolutionInputsRequested()) {
            exe.SaveGaborInputsTo(args.GetConvolutionInputsStorePath());
        }
        exe.CalculateGabor();
        exe.SaveResultsTo(args.GetOutputPath());

    } catch (const boost::program_options::error& e) {
        LOGE << e.what();
        PrintHelp(help_string);
        return 1;
    } catch (const std::exception& e) {
        LOGE << e.what();
        LOGE << "Exiting because of an error." << std::endl;
        return 2;
    } catch (...) {
        LOGE << "Caught an unknown exception." << std::endl;
        return 3;
    }

    return 0;
}