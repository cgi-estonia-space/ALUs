#include "algorithm_parameters.h"

#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include <boost/algorithm/string.hpp>

namespace {
constexpr char ALGORITHM_NAME_END_MARK{':'};
constexpr char PARAMETERS_SEPARATOR{','};
constexpr char PARAMETER_VALUE_SEPARATOR{'='};
constexpr char ALGORITHM_SEPARATORS{';'};
}  // namespace

namespace alus::app {

AlgorithmParameters::AlgorithmParameters(std::string_view parameter_list) {
    alg_name_ = ParseAlgName(parameter_list);
    params_ = ParseParameters(
        alg_name_.empty() ? parameter_list : parameter_list.substr(alg_name_.length() + 1, parameter_list.length()));
}

std::string AlgorithmParameters::ParseAlgName(std::string_view parameter_list) {
    const auto found = parameter_list.find(ALGORITHM_NAME_END_MARK);
    if (found == std::string_view::npos) {
        return {};
    }

    return std::string(parameter_list.substr(0, found));
}

AlgorithmParameters::Table AlgorithmParameters::ParseParameters(std::string_view parameters_only_list) {
    std::vector<std::string> parameter_pairs{};
    boost::split(parameter_pairs, parameters_only_list, std::bind1st(std::equal_to<char>(), PARAMETERS_SEPARATOR));

    Table params_and_values{};
    for (auto&& pp : parameter_pairs) {
        const auto param_val_sep_pos = pp.find(PARAMETER_VALUE_SEPARATOR);
        if (param_val_sep_pos == std::string_view::npos || param_val_sep_pos == pp.length() - 1) {
            throw std::runtime_error("Invalid algorithm parameter value pair - '" + std::string(pp) + "'");
        }

        const auto result = params_and_values.try_emplace(pp.substr(0, param_val_sep_pos),
                                                          pp.substr(param_val_sep_pos + 1, pp.length()));
        if (!result.second) {
            throw std::runtime_error("Redefining algorithm parameter value - '" + std::string(pp) + "'");
        }
    }

    return params_and_values;
}

AlgorithmParameters::AlgParamTables AlgorithmParameters::TryCreateFrom(std::string_view algs_param_values) {
    std::vector<std::string> param_value_split_by_alg;
    boost::split(
        param_value_split_by_alg, algs_param_values, std::bind1st(std::equal_to<char>(), ALGORITHM_SEPARATORS));

    AlgParamTables alg_params;
    for (auto&& pva : param_value_split_by_alg) {
        const AlgorithmParameters alg_param_parser{pva};
        alg_params.insert({std::string(alg_param_parser.GetAlgorithmName()), alg_param_parser.GetParameters()});
    }

    return alg_params;
}

AlgorithmParameters::AlgParamTables AlgorithmParameters::TryCreateFromFile(std::string_view file_path) {
    std::ifstream conf_file;
    conf_file.exceptions(std::ofstream::badbit);
    conf_file.open(file_path.data(), std::ios_base::in);
    std::string alg_params(std::istreambuf_iterator<char>(conf_file), {});
    alg_params.erase(std::remove(alg_params.begin(), alg_params.end(), '\n'), alg_params.end());

    return AlgorithmParameters::TryCreateFrom(alg_params);
}

AlgorithmParameters::AlgParamTables AlgorithmParameters::MergeAndWarn(const AlgParamTables& file_parameters,
                                                                      const AlgParamTables& command_line_parameters,
                                                                      std::string& warnings) {
    std::stringstream warnings_stream;
    // Command line overwrites file ones, so this is basis, because merge()
    // does not insert file conf parameters if they are already existing in
    // merged_args.
    AlgParamTables merged_args = command_line_parameters;
    for (auto&& alg : file_parameters) {
        const auto& alg_name = alg.first;
        Table file_conf_params = alg.second;  // no const because of merge().

        // If command line configuration miss this algorithm insert
        // everything from file conf.
        if (!merged_args.count(alg_name)) {
            merged_args.insert({alg_name, file_conf_params});
            continue;
        }

        // Merge file ones.
        merged_args.at(alg_name).merge(file_conf_params);

        // Create warning description about overwriting file configuration ones.
        const auto& command_line_table = command_line_parameters.at(alg_name);
        for (auto&& param : file_conf_params) {
            if (command_line_table.count(param.first)) {
                warnings_stream << alg_name << " parameter '" << param.first
                                << "' redeclared on command line, overruling file configuration value to '"
                                << command_line_table.at(param.first) << "'." << std::endl;
            }
        }
    }

    warnings = warnings_stream.str();

    return merged_args;
}

}  // namespace alus::app