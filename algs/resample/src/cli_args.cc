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

#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>

#include "cli_args.h"

namespace alus::resample {

namespace po = boost::program_options;

Arguments::Arguments() {
    ConstructResampleMethodList();
    ConstructCliArgs();
}

Arguments::Arguments(const std::vector<char*>& args) : Arguments() { ParseArgs(args); }

void Arguments::ConstructCliArgs() {
    std::string methods_list_help("Resampling method, one of the following -\n");
    for (const auto& [arg, type] : resample_arg_to_method_map_) {
        methods_list_help.append(arg).append(", ");
    }
    methods_list_help.erase(methods_list_help.length() - 2);

    // clang-format off
    app_args_.add_options()("help, h", "Print help")
        ("input,i", po::value<std::vector<std::string>>(&input_datasets_)->required(), "Input dataset(s)")
        ("destination,d", po::value<std::string>(&results_output_path_)->required(), "Results output path")
        ("dim_band", po::value<size_t>(&resample_dim_of_band_), "Dimension of the resampled image(s) "
         "taken from the specified band's dimensions of the (first) input. Alternative would be to manually specify "
         "using '--dim' or '--width' and '-height' arguments.")
        ("dim", po::value<std::string>(&resample_image_dim_arg_), "Dimension of the resampled image(s) '{width}x{height}' "
         "in pixels. For example - '1000x1500'. "
         "Alternatively can use '--width' and '--height' arguments or '--dim_band'.")
        ("--width", po::value<int>(&resample_image_dim_.columnsX), "Width of the resampled image(s). "
         "Must be specified together with '--height' argument. "
         "Alternatively can use '--dim' or '--dim_band' arguments.")
        ("--height", po::value<int>(&resample_image_dim_.rowsY), "Width of the resampled image(s). "
         "Must be specified together with '--width' argument. "
         "Alternatively can use '--dim' or '--dim_band' arguments.")
        ("tile_dim", po::value<std::string>(&tile_dim_arg_),
         "Output tiles' dimension '{width}x{height}' in pixels. For example - '100x50'. Alternatively can use"
         " '--tile_width' and '--tile_height' arguments.")
        ("tile_width", po::value<int>(&tile_dim_.columnsX), "Output tiles' width. Alternatively can use '--tile_dim'.")
        ("tile_height", po::value<int>(&tile_dim_.rowsY), "Output tiles' height. Alternatively can use '--tile_dim'.")
        ("overlap", po::value<size_t>(&pixel_overlap_)->default_value(0), "Tiles overlap in pixels")
        ("method,m", po::value<std::string>(&resample_method_arg_)->required(), methods_list_help.c_str())
        ("format,f", po::value<std::string>(&output_format_), "One of the following - GeoTIFF, NetCDF, Zarr. "
         "Or leave empty to use input format(must match one from listed ones)")
        ("crs,p", po::value<std::string>(&output_crs_), "Coordinate reference system/projection "
         "for output tiles. Leave empty to use input CRS. Consult GDAL/PROJ for supported ones. In general this "
         "value is supplied to SetWellKnownGeoCS() GDAL API. Some valid examples - 'WGS84', 'EPSG:4326'.")
        ("exclude", po::value<std::vector<size_t>>(&exclude_list_), "Bands to be excluded from resampling. "
         "When multiple inputs are specified which differ in band count a warning is given if band number specified "
         "is not present. Starting from 1. Argument can be specified multiple times.");
    // clang-format on

    combined_args_.add(app_args_).add(alus_args_.Get());
}

void Arguments::ParseArgs(const std::vector<char*>& args) {
    po::store(po::parse_command_line(static_cast<int>(args.size()), args.data(), combined_args_), vm_);
}

void Arguments::Check() {
    po::notify(vm_);

    if (vm_.count("dim_band") == 0U && vm_.count("dim") == 0U &&
        (vm_.count("width") == 0U || vm_.count("height") == 0U)) {
        throw std::invalid_argument(
            "Incorrectly supplied resampling dimensions. Either use '--dim_band' or '--dim' or "
            "define both '--width' and '--height'");
    }

    if (vm_.count("dim") != 0U) {
        resample_image_dim_ = ParseDimensionsFromString(resample_image_dim_arg_);
    }

    if (vm_.count("tile_dim") == 0U && (vm_.count("tile_width") == 0U || vm_.count("tile_height") == 0U)) {
        throw std::invalid_argument(
            "Incorrectly supplied output tile dimensions. Either use '--tile_dim' or define both '--tile_width' and "
            "'--tile_height'");
    }

    if (vm_.count("tile_dim") != 0U) {
        tile_dim_ = ParseDimensionsFromString(tile_dim_arg_);
    }

    if (!resample_arg_to_method_map_.count(resample_method_arg_)) {
        throw std::invalid_argument(resample_method_arg_ + " is not a supported resampling method");
    }
    resample_method_ = resample_arg_to_method_map_.at(resample_method_arg_);

    alus_args_.Check();
}

bool Arguments::IsHelpRequested() const { return vm_.count("help") != 0; }

std::string Arguments::GetHelp() const {
    std::stringstream help;
    help << combined_args_;
    return help.str();
}

std::optional<size_t> Arguments::GetResamplingDimensionOfBand() const {
    if (vm_.count("dim_band") == 0U) {
        return std::nullopt;
    }

    return std::make_optional(resample_dim_of_band_);
}

std::optional<std::string> Arguments::GetOutputFormat() const {
    if (output_format_.empty()) {
        return std::nullopt;
    }

    return std::make_optional(output_format_);
}

std::optional<std::string> Arguments::GetOutputCrs() const {
    if (output_crs_.empty()) {
        return std::nullopt;
    }

    return std::make_optional(output_crs_);
}

void Arguments::ConstructResampleMethodList() {
    resample_arg_to_method_map_.try_emplace("nearest-neighbour", Method::NEAREST_NEIGHBOUR);
    resample_arg_to_method_map_.try_emplace("linear", Method::LINEAR);
    resample_arg_to_method_map_.try_emplace("cubic", Method::CUBIC);
    resample_arg_to_method_map_.try_emplace("cubic2p-bspline", Method::CUBIC2P_BSPLINE);
    resample_arg_to_method_map_.try_emplace("cubic2p-catmullrom", Method::CUBIC2P_CATMULLROM);
    resample_arg_to_method_map_.try_emplace("cubic2p-c05c03", Method::CUBIC2P_C05C03);
    resample_arg_to_method_map_.try_emplace("super", Method::SUPER);
    resample_arg_to_method_map_.try_emplace("lanczos", Method::LANCZOS);
    resample_arg_to_method_map_.try_emplace("lanczos3", Method::LANCZOS3);
    resample_arg_to_method_map_.try_emplace("smooth-edge", Method::SMOOTH_EDGE);
}

RasterDimension Arguments::ParseDimensionsFromString(std::string_view arg) {
    std::vector<std::string> sizes;
    boost::split(sizes, arg, boost::is_any_of("x"));
    if (sizes.size() != 2) {
        throw std::invalid_argument("Supplied dimension argument(" + std::string(arg) + ") is in a wrong format.");
    }

    return {static_cast<int>(std::stoul(sizes.front())),
            static_cast<int>(std::stoul(sizes.back()))};
}

}  // namespace alus::resample