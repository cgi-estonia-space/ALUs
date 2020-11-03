#pragma once

#include <cstddef>
#include <unordered_map>
#include <string>
#include <string_view>
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
                                       const AlgParamTables& command_line_parameters,
                                       std::string& warnings);

   private:

    static std::string ParseAlgName(std::string_view parameter_list);
    static Table ParseParameters(std::string_view parameters_only_list);

    std::string alg_name_{};
    Table params_{};
};

}
