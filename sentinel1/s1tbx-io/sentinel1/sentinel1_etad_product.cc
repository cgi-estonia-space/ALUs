/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#include "sentinel1_etad_product.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iterator>
#include <limits>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cpl_port.h>
#include <gdal.h>
#include <gdal_priv.h>
#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/filesystem/path.hpp>
#include <pugixml.hpp>

#include "ceres-core/core/i_virtual_dir.h"
#include "etad_computation.cuh"
#include "gdal_util.h"

namespace {
constexpr double INVALID = std::numeric_limits<double>::quiet_NaN();

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("ETAD: " + message);
    }
}

std::string Text(const pugi::xml_node& node, const char* field) {
    const std::string value = boost::algorithm::trim_copy(std::string(node.child_value(field)));
    Require(!value.empty(), "missing " + std::string(node.name()) + "/" + field);
    return value;
}

double Number(const std::string& text) {
    size_t consumed{};
    const double value = std::stod(text, &consumed);
    Require(consumed == text.size() && std::isfinite(value), "invalid number '" + text + "'");
    return value;
}

double Number(const pugi::xml_node& node, const char* field) { return Number(Text(node, field)); }

int PositiveInteger(const std::string& text) {
    size_t consumed{};
    const int value = std::stoi(text, &consumed);
    Require(consumed == text.size() && value > 0, "invalid positive integer '" + text + "'");
    return value;
}

int Index(const pugi::xml_node& node, const char* field) {
    Require(node.attribute(field), "missing " + std::string(node.name()) + "@" + field);
    return PositiveInteger(node.attribute(field).value());
}

void CheckCount(const pugi::xml_node& node, size_t count) {
    Require(static_cast<size_t>(Index(node, "count")) == count, "count mismatch in " + std::string(node.name()));
}

double Time(const pugi::xml_node& node, const char* field) {
    constexpr size_t DATE_TIME_SEPARATOR = 10;
    auto text = Text(node, field);
    static const std::regex UTC_PATTERN(R"(\d{4}-\d{2}-\d{2}T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d{1,6})?Z?)");
    Require(std::regex_match(text, UTC_PATTERN), "invalid UTC timestamp '" + text + "'");
    if (text.back() == 'Z') {
        text.pop_back();
    }
    text[DATE_TIME_SEPARATOR] = ' ';
    const auto time = boost::posix_time::time_from_string(text);
    const boost::posix_time::ptime epoch(boost::gregorian::date(2000, 1, 1));
    const auto elapsed = time - epoch;
    Require(!elapsed.is_special(), "invalid UTC timestamp '" + text + "'");
    return std::chrono::duration<double>(std::chrono::microseconds(elapsed.total_microseconds())).count();
}

alus::s1tbx::etad::Coverage ReadCoverage(const pugi::xml_node& node) {
    alus::s1tbx::etad::Coverage coverage{Time(node, "azimuthTimeMin"), Time(node, "azimuthTimeMax"),
                                         Number(node, "rangeTimeMin"), Number(node, "rangeTimeMax")};
    Require(coverage.azimuth_min < coverage.azimuth_max && coverage.range_min < coverage.range_max,
            "invalid temporal coverage");
    return coverage;
}

size_t GridSize(size_t rows, size_t columns) {
    Require(rows >= 2 && columns >= 2 && rows <= std::numeric_limits<size_t>::max() / columns,
            "grid must have at least two rows and columns and a representable size");
    return rows * columns;
}

void CheckGrid(const alus::s1tbx::etad::Grid& grid) {
    Require(grid.values.size() == GridSize(grid.rows, grid.columns), "grid buffer size mismatch");
}

void CheckSameGrid(const alus::s1tbx::etad::Grid& left, const alus::s1tbx::etad::Grid& right) {
    CheckGrid(left);
    CheckGrid(right);
    Require(left.rows == right.rows && left.columns == right.columns, "layer dimensions differ");
}

void CheckFrequency(double frequency_hz) {
    Require(std::isfinite(frequency_hz) && frequency_hz > 0, "radar frequency must be positive and finite, in Hz");
}

std::string SingleFile(alus::ceres::IVirtualDir& directory, const std::string& folder, const std::string& extension) {
    std::vector<std::string> matches;
    for (const auto& name : directory.List(folder)) {
        if (boost::algorithm::iequals(std::filesystem::path(name).extension().string(), extension)) {
            matches.push_back(folder + "/" + name);
        }
    }
    Require(matches.size() == 1, "expected one " + extension + " file in " + folder);
    return matches.front();
}

std::vector<alus::s1tbx::etad::Calibration> ReadCalibrations(const pugi::xml_node& aux) {
    const auto direct_calibrations = aux.child("instrumentTimingCalibrationList");
    const bool has_direct_calibrations = !direct_calibrations.empty();
    auto list = direct_calibrations;
    const char* entry_name = "instrumentTimingCalibration";
    if (!has_direct_calibrations) {
        list = aux.child("instrumentTimingCalibrationOffsetList");
        entry_name = "instrumentTimingCalibrationOffset";
    }
    std::vector<alus::s1tbx::etad::Calibration> calibrations;
    for (const auto& node : list.children(entry_name)) {
        alus::s1tbx::etad::Calibration calibration;
        calibration.swath_id = Text(node, "swath");
        calibration.polarisation = Text(node, "polarisation");
        if (has_direct_calibrations) {
            calibration.range_seconds = Number(node, "rangeCalibration");
            calibration.azimuth_seconds = Number(node, "azimuthCalibration");
        } else {
            const auto reference = aux.child("instrumentTimingCalibrationReference");
            calibration.range_seconds = Number(reference, "rangeCalibration");
            calibration.azimuth_seconds = Number(reference, "azimuthCalibration");
            // Keep the product reference and channel offsets separate. Selecting/applying channel offsets
            // requires polarisation-aware processing; these are not additional atmospheric delay layers.
            calibration.range_offset_seconds = Number(node, "rangeOffset");
            calibration.azimuth_offset_seconds = Number(node, "azimuthOffset");
        }
        calibrations.push_back(std::move(calibration));
    }
    CheckCount(list, calibrations.size());
    return calibrations;
}
}  // namespace

namespace alus::s1tbx::etad {

Metadata ParseAnnotation(std::string_view xml) {
    pugi::xml_document document;
    const auto result = document.load_buffer(xml.data(), xml.size());
    Require(result, "annotation XML: " + std::string(result.description()));
    const auto root = document.child("etadProduct");
    Require(root, "annotation root must be etadProduct");
    Metadata metadata;
    const auto header = root.child("etadHeader");
    metadata.mission = Text(header, "missionId");
    metadata.mode = Text(header, "mode");
    Require(metadata.mission.rfind("S1", 0) == 0 && metadata.mode == "IW" && Text(header, "productType") == "SLC",
            "only Sentinel-1 IW SLC ETAD products are supported");
    metadata.coverage = ReadCoverage(root.child("productCoverage").child("temporalCoverage"));
    metadata.carrier_frequency_hz = Number(root.child("productInformation"), "carrierFrequency");
    CheckFrequency(metadata.carrier_frequency_hz);

    const auto components = root.child("productComponents");
    const auto products = components.child("inputProductList");
    std::set<int> product_ids;
    for (const auto& node : products.children("inputProduct")) {
        Acquisition acquisition;
        acquisition.p_index = Index(node, "pIndex");
        Require(product_ids.insert(acquisition.p_index).second, "duplicate pIndex");
        acquisition.product_id = Text(node, "productID");
        acquisition.start_time = Time(node, "startTime");
        acquisition.stop_time = Time(node, "stopTime");
        Require(acquisition.start_time <= acquisition.stop_time, "invalid acquisition time interval");
        std::set<int> swath_ids;
        std::set<std::string> swath_names;
        const auto swaths = node.child("swathList");
        for (const auto& swath_node : swaths.children("swath")) {
            Swath swath;
            swath.s_index = Index(swath_node, "sIndex");
            swath.swath_id = Text(swath_node, "swathID");
            Require(swath_ids.insert(swath.s_index).second && swath_names.insert(swath.swath_id).second,
                    "duplicate swath");
            Require(swath.swath_id == "IW1" || swath.swath_id == "IW2" || swath.swath_id == "IW3", "invalid IW swath");
            const auto indices = swath_node.child("bIndexList");
            std::set<int> unique_indices;
            for (const auto& index : indices.children("bIndex")) {
                const int b_index = PositiveInteger(boost::algorithm::trim_copy(std::string(index.child_value())));
                Require(unique_indices.insert(b_index).second, "duplicate bIndex in swath");
                swath.burst_indices.push_back(b_index);
            }
            CheckCount(indices, swath.burst_indices.size());
            acquisition.swaths.push_back(std::move(swath));
        }
        CheckCount(swaths, acquisition.swaths.size());
        metadata.acquisitions.push_back(std::move(acquisition));
    }
    CheckCount(products, metadata.acquisitions.size());
    Require(
        static_cast<size_t>(PositiveInteger(Text(components, "numberOfInputProducts"))) == metadata.acquisitions.size(),
        "numberOfInputProducts mismatch");

    const auto burst_list = root.child("etadBurstList");
    std::set<int> burst_ids;
    constexpr std::array<const char*, 4> CORNERS{"EarlyAzimuthNearRange", "EarlyAzimuthFarRange",
                                                 "LateAzimuthNearRange", "LateAzimuthFarRange"};
    for (const auto& node : burst_list.children("etadBurst")) {
        Burst burst;
        const auto data = node.child("burstData");
        burst.b_index = Index(data, "bIndex");
        burst.p_index = Index(data, "pIndex");
        burst.s_index = Index(data, "sIndex");
        burst.swath_id = Text(data, "swathID");
        Require(burst_ids.insert(burst.b_index).second, "duplicate global bIndex");
        burst.coverage = ReadCoverage(node.child("burstCoverage").child("temporalCoverage"));
        const auto spatial = node.child("burstCoverage").child("spatialCoverage");
        for (size_t i = 0; i < CORNERS.size(); ++i) {
            const auto corner = spatial.find_child_by_attribute("coordinates", "corner", CORNERS[i]);
            burst.corners[i] = {Number(corner, "latitude"), Number(corner, "longitude")};
        }
        const auto grid = node.child("gridInformation");
        burst.grid_start_azimuth = Number(grid, "gridStartAzimuthTime");
        burst.grid_start_range = Number(grid, "gridStartRangeTime");
        burst.sampling_azimuth = Number(grid.child("gridSampling"), "azimuth");
        burst.sampling_range = Number(grid.child("gridSampling"), "range");
        Require(burst.sampling_azimuth > 0 && burst.sampling_range > 0, "grid sampling must be positive");
        burst.rows = PositiveInteger(Text(grid.child("gridDimensions"), "azimuthExtent"));
        burst.columns = PositiveInteger(Text(grid.child("gridDimensions"), "rangeExtent"));
        GridSize(burst.rows, burst.columns);
        metadata.bursts.push_back(std::move(burst));
    }
    CheckCount(burst_list, metadata.bursts.size());
    Require(static_cast<size_t>(PositiveInteger(Text(components, "numberOfBursts"))) == metadata.bursts.size(),
            "numberOfBursts mismatch");
    std::set<int> referenced;
    std::set<std::string> all_swaths;
    for (const auto& acquisition : metadata.acquisitions) {
        for (const auto& swath : acquisition.swaths) {
            all_swaths.insert(swath.swath_id);
            for (const int b_index : swath.burst_indices) {
                const auto& burst = GetBurst(metadata, b_index);
                Require(burst.p_index == acquisition.p_index && burst.s_index == swath.s_index &&
                            burst.swath_id == swath.swath_id && referenced.insert(b_index).second,
                        "inconsistent burst reference " + std::to_string(b_index));
            }
        }
    }
    Require(referenced.size() == metadata.bursts.size(), "unreferenced burst metadata");
    Require(static_cast<size_t>(PositiveInteger(Text(components, "numberOfSwaths"))) == all_swaths.size(),
            "numberOfSwaths mismatch");

    metadata.calibrations =
        ReadCalibrations(root.child("processingInformation").child("auxInputData").child("auxSetap"));
    return metadata;
}

const Acquisition& MatchAcquisition(const Metadata& metadata, std::string_view slc_product_name) {
    static const std::regex SENSING_TIMES(R"(\d{8}T\d{6}_\d{8}T\d{6}(?:_\d{6})?)");
    std::smatch match;
    const std::string name(slc_product_name);
    Require(std::regex_search(name, match, SENSING_TIMES), "no sensing timestamps in SLC name '" + name + "'");
    const auto token = boost::algorithm::to_lower_copy(match.str());
    const Acquisition* found = nullptr;
    for (const auto& acquisition : metadata.acquisitions) {
        if (boost::algorithm::to_lower_copy(acquisition.product_id).find(token) != std::string::npos) {
            Require(found == nullptr, "ambiguous acquisition match for '" + name + "'");
            found = &acquisition;
        }
    }
    Require(found != nullptr, "no acquisition matches '" + name + "'");
    return *found;
}

const Acquisition* FindAcquisition(const Metadata& metadata, double azimuth_time) {
    for (const auto& acquisition : metadata.acquisitions) {
        if (acquisition.start_time <= azimuth_time && azimuth_time <= acquisition.stop_time) {
            return &acquisition;
        }
    }
    return nullptr;
}

const Burst& GetBurst(const Metadata& metadata, int b_index) {
    const auto found = std::find_if(metadata.bursts.begin(), metadata.bursts.end(),
                                    [b_index](const auto& burst) { return burst.b_index == b_index; });
    Require(found != metadata.bursts.end(), "unknown bIndex " + std::to_string(b_index));
    return *found;
}

const Burst* FindBurst(const Metadata& metadata, int p_index, std::string_view swath_id, double azimuth_time,
                       double range_time) {
    for (const auto& acquisition : metadata.acquisitions) {
        if (acquisition.p_index != p_index) {
            continue;
        }
        for (const auto& swath : acquisition.swaths) {
            if (swath.swath_id != swath_id) {
                continue;
            }
            for (const int b_index : swath.burst_indices) {
                const auto& burst = GetBurst(metadata, b_index);
                const auto& bounds = burst.coverage;
                if (azimuth_time > bounds.azimuth_min && azimuth_time < bounds.azimuth_max &&
                    range_time > bounds.range_min && range_time < bounds.range_max) {
                    return &burst;
                }
            }
        }
    }
    return nullptr;
}

double RangeCalibration(const Metadata& metadata, std::string_view swath_id) {
    for (const auto& calibration : metadata.calibrations) {
        if (calibration.swath_id == swath_id) {
            return calibration.range_seconds;
        }
    }
    return 0.0;
}

double Interpolate(const Grid& grid, const Burst& burst, double azimuth_time, double range_time) {
    CheckGrid(grid);
    Require(grid.rows == burst.rows && grid.columns == burst.columns, "grid does not match burst dimensions");
    Require(std::isfinite(burst.sampling_azimuth) && burst.sampling_azimuth > 0 &&
                std::isfinite(burst.sampling_range) && burst.sampling_range > 0,
            "invalid grid sampling");
    return SampleGrid(View(grid), Geometry(burst), azimuth_time, range_time);
}

Grid ComputePhase(const Grid& troposphere, const Grid& geodetic, const Grid& ionosphere, double frequency_hz,
                  double calibration_seconds) {
    CheckSameGrid(troposphere, geodetic);
    CheckSameGrid(troposphere, ionosphere);
    CheckFrequency(frequency_hz);
    Require(std::isfinite(calibration_seconds), "invalid range calibration");
    Grid phase{troposphere.rows, troposphere.columns, std::vector<double>(troposphere.values.size())};
    for (size_t i = 0; i < phase.values.size(); ++i) {
        phase.values[i] = RangePhase(troposphere.values[i], geodetic.values[i], ionosphere.values[i],
                                     calibration_seconds, frequency_hz);
    }
    return phase;
}

Grid ComputeTroposphericGradient(const Grid& troposphere, const Grid& height) {
    CheckSameGrid(troposphere, height);
    Grid gradient{height.rows, height.columns, std::vector<double>(height.values.size(), 0.0)};
    for (size_t y = 0; y < height.rows; ++y) {
        for (size_t x = 0; x < height.columns; ++x) {
            gradient.values[y * height.columns + x] = TroposphericGradientAt(View(troposphere), View(height), y, x);
        }
    }
    return gradient;
}

InSarLayers ComputeInSarLayers(const Grid& troposphere, const Grid& geodetic, const Grid& ionosphere,
                               const Grid& height, double frequency_hz, double calibration_seconds) {
    auto phase = ComputePhase(troposphere, geodetic, ionosphere, frequency_hz, calibration_seconds);
    auto gradient = ComputeTroposphericGradient(troposphere, height);
    for (auto& value : gradient.values) {
        value = DelayToPhase(value, frequency_hz);
    }
    return {std::move(phase), height, std::move(gradient)};
}
}  // namespace alus::s1tbx::etad

namespace alus::s1tbx {

Sentinel1EtadProduct Sentinel1EtadProduct::Open(const std::filesystem::path& path) {
    Sentinel1EtadProduct product;
    auto input = std::filesystem::absolute(path).lexically_normal();
    if (input.filename().empty()) {
        input = input.parent_path();
    }
    Require(std::filesystem::exists(input), "input does not exist: " + input.string());
    if (input.filename() == "manifest.safe") {
        input = input.parent_path();
    }
    Require(std::filesystem::is_directory(input) || boost::algorithm::iequals(input.extension().string(), ".zip"),
            "expected SAFE directory, manifest.safe, or ZIP");
    product.directory_ = ceres::IVirtualDir::Create(boost::filesystem::path(input.string()));
    Require(product.directory_ != nullptr, "cannot open " + input.string());
    auto& directory = *product.directory_;
    std::string root;
    if (!directory.Exists("manifest.safe")) {
        std::vector<std::string> roots;
        for (const auto& entry : directory.List("")) {
            if (directory.Exists(entry + "/manifest.safe")) {
                roots.push_back(entry);
            }
        }
        Require(roots.size() == 1, "expected one SAFE root containing manifest.safe");
        root = roots.front() + "/";
    }
    product.name_ = root.empty() ? input.stem().string() : std::filesystem::path(root).parent_path().stem().string();
    const auto annotation = SingleFile(directory, root + "annotation", ".xml");
    const auto measurement = SingleFile(directory, root + "measurement", ".nc");
    // GetFile uses the virtual directory's owned extraction path (unlike Zip::GetInputStream).
    const auto annotation_path = directory.GetFile(annotation);
    std::ifstream stream(annotation_path.string(), std::ios::binary);
    Require(stream.good(), "cannot read annotation " + annotation);
    const std::string xml{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    product.metadata_ = etad::ParseAnnotation(xml);
    product.measurement_path_ = directory.GetFile(measurement).string();
    Require(std::filesystem::is_regular_file(product.measurement_path_), "measurement extraction failed");
    return product;
}

etad::Grid Sentinel1EtadProduct::LoadLayer(int b_index, std::string_view layer) const {
    const auto& burst = etad::GetBurst(metadata_, b_index);
    Require(!layer.empty() && layer.find_first_of("/\\") == std::string_view::npos, "invalid layer name");
    std::ostringstream group_name;
    group_name << "Burst" << std::setfill('0') << std::setw(4) << burst.b_index;
    const std::string context = burst.swath_id + "/" + group_name.str() + "/" + std::string(layer);
    const char* const drivers[] = {gdal::constants::GDAL_NETCDF_DRIVER, nullptr};
    // Read multidimensional arrays rather than raster bands: no GDAL bottom-up/georeferencing row conversion.
    GDALAllRegister();
    const std::unique_ptr<GDALDataset, decltype(&GDALClose)> dataset(
        static_cast<GDALDataset*>(GDALOpenEx(measurement_path_.c_str(), GDAL_OF_MULTIDIM_RASTER | GDAL_OF_READONLY,
                                             drivers, nullptr, nullptr)),
        GDALClose);
    Require(dataset != nullptr, "cannot open NetCDF measurement for " + context);
    const auto root = dataset->GetRootGroup();
    const auto swath = root ? root->OpenGroup(burst.swath_id) : nullptr;
    const auto group = swath ? swath->OpenGroup(group_name.str()) : nullptr;
    const auto array = group ? group->OpenMDArray(std::string(layer)) : nullptr;
    Require(array != nullptr, "missing NetCDF array " + context);
    const auto& dimensions = array->GetDimensions();
    Require(dimensions.size() == 2 && dimensions[0]->GetName() == "azimuthExtent" &&
                dimensions[1]->GetName() == "rangeExtent" && dimensions[0]->GetSize() == burst.rows &&
                dimensions[1]->GetSize() == burst.columns,
            "NetCDF/XML dimension or axis mismatch for " + context);
    etad::Grid grid{burst.rows, burst.columns, std::vector<double>(GridSize(burst.rows, burst.columns))};
    const std::array<GUInt64, 2> start{0, 0};
    const std::array<size_t, 2> count{grid.rows, grid.columns};
    Require(array->Read(start.data(), count.data(), nullptr, nullptr, GDALExtendedDataType::Create(GDT_Float64),
                        grid.values.data()),
            "cannot read " + context);
    bool has_no_data{};
    bool has_scale{};
    bool has_offset{};
    const double no_data = array->GetNoDataValueAsDouble(&has_no_data);
    const double scale = array->GetScale(&has_scale);
    const double offset = array->GetOffset(&has_offset);
    for (auto& value : grid.values) {
        if (!std::isfinite(value) || (has_no_data && value == no_data)) {
            value = INVALID;
        } else {
            value = value * (has_scale ? scale : 1.0) + (has_offset ? offset : 0.0);
            if (!std::isfinite(value)) {
                value = INVALID;
            }
        }
    }
    return grid;
}

etad::InSarLayers Sentinel1EtadProduct::LoadInSarBurstLayers(int b_index, double frequency_hz) const {
    CheckFrequency(frequency_hz);
    const auto& burst = etad::GetBurst(metadata_, b_index);
    const auto troposphere = LoadLayer(b_index, "troposphericCorrectionRg");
    const auto geodetic = LoadLayer(b_index, "geodeticCorrectionRg");
    const auto ionosphere = LoadLayer(b_index, "ionosphericCorrectionRg");
    const auto height = LoadLayer(b_index, "height");
    return etad::ComputeInSarLayers(troposphere, geodetic, ionosphere, height, frequency_hz,
                                    etad::RangeCalibration(metadata_, burst.swath_id));
}
}  // namespace alus::s1tbx
