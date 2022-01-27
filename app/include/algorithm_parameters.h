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
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace alus::app {

class AlgorithmParameters {
public:
    using Table = std::unordered_map<std::string, std::string>;
    using AlgParamTables = std::unordered_map<std::string, Table>;

    AlgorithmParameters() = delete;

    /**
     * Construct a SINGLE algorithm parameters.
     *
     * @param parameter_list A string in a following format: "alg_name:param1=value,param2=value" or
     *                       "param1=value,param2=value"
     */
    explicit AlgorithmParameters(std::string_view parameter_list);

    [[nodiscard]] std::string_view GetAlgorithmName() const { return alg_name_; }
    [[nodiscard]] const Table& GetParameters() const { return params_; }

    /**
     * Construct a SINGLE or many algorithm parameters.
     *
     * Syntax follows same structure as for single algorithm, but multiple algorithms' parameters are spearated by ";"
     *
     * @param parameter_list A string following a format of @AlgorithmParameters() or for
     *                       multiple algorithms "alg_name1:<params>;alg_name2:<params>;..."
     */
    static AlgParamTables TryCreateFrom(std::string_view algs_param_values);
    /**
     * Construct a SINGLE or many algorithm parameters.
     *
     * @param file_path Path to a file that consists of compatible syntax of algorithm parameters as for
     *                  @TryCreateFrom()
     */
    static AlgParamTables TryCreateFromFile(std::string_view file_path);

    static AlgParamTables MergeAndWarn(const AlgParamTables& file_parameters,
                                       const AlgParamTables& command_line_parameters, std::string& warnings);

private:
    static std::string ParseAlgName(std::string_view parameter_list);
    static Table ParseParameters(std::string_view parameters_only_list);

    std::string alg_name_{};
    Table params_{};
};

}  // namespace alus::app
