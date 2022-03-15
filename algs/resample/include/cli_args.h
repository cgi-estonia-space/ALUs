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

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include <boost/program_options.hpp>

#include "alus_log.h"
#include "app_utils.h"
#include "raster_properties.h"
#include "resample_method.h"

namespace alus::resample {

class Arguments final {
public:
    Arguments();
    explicit Arguments(const std::vector<char*>& args);

    void ParseArgs(const std::vector<char*>& args);
    void Check();

    [[nodiscard]] bool IsHelpRequested() const;
    [[nodiscard]] std::string GetHelp() const;
    [[nodiscard]] const std::vector<std::string>& GetInputs() const { return input_datasets_; }
    [[nodiscard]] std::optional<size_t> GetResamplingDimensionOfBand() const;
    [[nodiscard]] RasterDimension GetResamplingDimension() const { return resample_image_dim_; }
    [[nodiscard]] RasterDimension GetTileDimension() const { return tile_dim_; }
    [[nodiscard]] size_t GetPixelOverlap() const { return pixel_overlap_; }
    [[nodiscard]] Method GetResampleMethod() const { return resample_method_; }
    [[nodiscard]] std::string_view GetOutputPath() const { return results_output_path_; }
    [[nodiscard]] std::optional<std::string> GetOutputFormat() const;
    [[nodiscard]] std::optional<std::string> GetOutputCrs() const;
    [[nodiscard]] const std::vector<size_t>& GetExcludedBands() const { return exclude_list_; }
    [[nodiscard]] size_t GetGpuMemoryPercentage() const { return alus_args_.GetGpuMemoryPercentage(); }
    [[nodiscard]] common::log::Level GetLogLevel() const { return alus_args_.GetLogLevel(); }

    ~Arguments() = default;

private:
    void ConstructCliArgs();
    void ConstructResampleMethodList();
    [[nodiscard]] static RasterDimension ParseDimensionsFromString(std::string_view arg);

    boost::program_options::variables_map vm_;
    boost::program_options::options_description app_args_{""};
    app::Arguments alus_args_;
    boost::program_options::options_description combined_args_{"Arguments"};

    std::vector<std::string> input_datasets_{};
    std::string results_output_path_{};
    std::string resample_image_dim_arg_{};
    size_t resample_dim_of_band_{};
    RasterDimension resample_image_dim_{};
    std::string tile_dim_arg_{};
    RasterDimension tile_dim_{};
    std::string output_format_{};
    std::string output_crs_{};
    size_t pixel_overlap_{};
    std::string resample_method_arg_{};
    std::map<std::string, Method> resample_arg_to_method_map_{};
    Method resample_method_{};
    std::vector<size_t> exclude_list_{};
};
}  // namespace alus::resample