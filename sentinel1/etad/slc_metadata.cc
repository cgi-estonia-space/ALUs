/** Native Sentinel-1 SLC metadata parsing. SPDX-License-Identifier: GPL-3.0-or-later */
#include "slc_metadata.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/posix_time/time_parsers.hpp>
#include <boost/filesystem/path.hpp>
#include <pugixml.hpp>

#include "ceres-core/core/i_virtual_dir.h"

namespace {
constexpr double LIGHT_SPEED_METRES_PER_SECOND = 299792458.0;

void Require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("SLC metadata: " + message);
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

size_t PositiveSize(const std::string& text) {
    Require(!text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char character) {
                return std::isdigit(character) != 0;
            }),
            "invalid positive integer '" + text + "'");
    size_t consumed{};
    const auto value = std::stoull(text, &consumed);
    Require(consumed == text.size() && value > 0 && value <= std::numeric_limits<size_t>::max(),
            "invalid positive integer '" + text + "'");
    return static_cast<size_t>(value);
}

size_t PositiveSize(const pugi::xml_node& node, const char* field) { return PositiveSize(Text(node, field)); }

int PositiveInteger(const pugi::xml_node& node, const char* field) {
    size_t consumed{};
    const auto text = Text(node, field);
    const int value = std::stoi(text, &consumed);
    Require(consumed == text.size() && value > 0, "invalid positive integer '" + text + "'");
    return value;
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

std::string Upper(std::string value) {
    boost::algorithm::trim(value);
    boost::algorithm::to_upper(value);
    return value;
}

std::string ProductName(std::filesystem::path path) {
    while (boost::algorithm::iequals(path.extension().string(), ".zip") ||
           boost::algorithm::iequals(path.extension().string(), ".safe")) {
        path = path.stem();
    }
    return path.filename().string();
}

std::string ReadFile(alus::ceres::IVirtualDir& directory, const std::string& path) {
    const auto file = directory.GetFile(path);
    std::ifstream stream(file.string(), std::ios::binary);
    Require(stream.good(), "cannot read annotation " + path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
}  // namespace

namespace alus::s1tbx::slc {

Metadata ParseAnnotation(std::string_view xml, std::string_view product_name) {
    pugi::xml_document document;
    const auto result = document.load_buffer(xml.data(), xml.size());
    Require(result, "annotation XML: " + std::string(result.description()));
    const auto root = document.child("product");
    Require(root, "annotation root must be product");

    Metadata metadata;
    metadata.product_name = boost::algorithm::trim_copy(std::string(product_name));
    Require(!metadata.product_name.empty(), "empty product name");

    const auto header = root.child("adsHeader");
    metadata.mission = Upper(Text(header, "missionId"));
    metadata.mode = Upper(Text(header, "mode"));
    metadata.swath_id = Upper(Text(header, "swath"));
    metadata.polarisation = Upper(Text(header, "polarisation"));
    metadata.absolute_orbit = PositiveInteger(header, "absoluteOrbitNumber");
    metadata.start_time = Time(header, "startTime");
    metadata.stop_time = Time(header, "stopTime");
    Require(metadata.mission.rfind("S1", 0) == 0 && metadata.mode == "IW" &&
                Upper(Text(header, "productType")) == "SLC",
            "only Sentinel-1 IW SLC annotations are supported");
    Require(metadata.swath_id == "IW1" || metadata.swath_id == "IW2" || metadata.swath_id == "IW3",
            "invalid IW swath");
    Require(metadata.polarisation == "VV" || metadata.polarisation == "VH" || metadata.polarisation == "HH" ||
                metadata.polarisation == "HV",
            "invalid polarisation");
    Require(metadata.start_time <= metadata.stop_time, "invalid annotation time interval");

    metadata.radar_frequency_hz = Number(root.child("generalAnnotation").child("productInformation"), "radarFrequency");
    Require(metadata.radar_frequency_hz > 0, "radar frequency must be positive, in Hz");

    const auto image = root.child("imageAnnotation").child("imageInformation");
    auto& raster = metadata.raster;
    raster.azimuth_time_interval = Number(image, "azimuthTimeInterval");
    raster.first_range_time = Number(image, "slantRangeTime");
    const double range_pixel_spacing = Number(image, "rangePixelSpacing");
    raster.total_lines = PositiveSize(image, "numberOfLines");
    raster.total_samples = PositiveSize(image, "numberOfSamples");
    Require(raster.azimuth_time_interval > 0 && raster.first_range_time > 0 && range_pixel_spacing > 0,
            "sampling values must be positive");
    raster.range_time_interval = 2.0 * range_pixel_spacing / LIGHT_SPEED_METRES_PER_SECOND;

    const auto timing = root.child("swathTiming");
    raster.lines_per_burst = PositiveSize(timing, "linesPerBurst");
    raster.samples_per_burst = PositiveSize(timing, "samplesPerBurst");
    const auto burst_list = timing.child("burstList");
    Require(burst_list.attribute("count"), "missing burstList@count");
    const size_t declared_count = PositiveSize(burst_list.attribute("count").value());
    for (const auto& burst : burst_list.children("burst")) {
        metadata.bursts.push_back({Time(burst, "azimuthTime")});
    }
    Require(metadata.bursts.size() == declared_count, "count mismatch in burstList");
    Require(raster.samples_per_burst == raster.total_samples, "samplesPerBurst does not match numberOfSamples");
    Require(raster.lines_per_burst <= std::numeric_limits<size_t>::max() / metadata.bursts.size() &&
                raster.lines_per_burst * metadata.bursts.size() == raster.total_lines,
            "burst lines do not match numberOfLines");
    for (size_t i = 1; i < metadata.bursts.size(); ++i) {
        Require(metadata.bursts[i - 1].first_line_time < metadata.bursts[i].first_line_time,
                "burst first-line times must be strictly increasing");
    }
    return metadata;
}

Metadata ReadMetadata(const std::filesystem::path& path, std::string_view swath_id, std::string_view polarisation) {
    auto input = std::filesystem::absolute(path).lexically_normal();
    if (input.filename().empty()) {
        input = input.parent_path();
    }
    Require(std::filesystem::exists(input), "input does not exist: " + input.string());
    if (boost::algorithm::iequals(input.filename().string(), "manifest.safe")) {
        input = input.parent_path();
    }
    Require(std::filesystem::is_directory(input) || boost::algorithm::iequals(input.extension().string(), ".zip"),
            "expected SAFE directory, manifest.safe, or ZIP");
    auto directory = ceres::IVirtualDir::Create(boost::filesystem::path(input.string()));
    Require(directory != nullptr, "cannot open " + input.string());

    std::string root;
    if (!directory->Exists("manifest.safe")) {
        std::vector<std::string> roots;
        for (const auto& entry : directory->List("")) {
            if (directory->Exists(entry + "/manifest.safe")) {
                roots.push_back(entry);
            }
        }
        Require(roots.size() == 1, "expected one SAFE root containing manifest.safe");
        root = roots.front() + "/";
    }
    const std::string product_name =
        root.empty() ? ProductName(input) : ProductName(std::filesystem::path(root).parent_path());
    const std::string requested_swath = Upper(std::string(swath_id));
    const std::string requested_polarisation = Upper(std::string(polarisation));
    Require(!requested_swath.empty() && !requested_polarisation.empty(), "swath and polarisation are required");

    std::vector<Metadata> matches;
    const std::string annotation_folder = root + "annotation";
    for (const auto& name : directory->List(annotation_folder)) {
        if (!boost::algorithm::iequals(std::filesystem::path(name).extension().string(), ".xml")) {
            continue;
        }
        auto metadata = ParseAnnotation(ReadFile(*directory, annotation_folder + "/" + name), product_name);
        if (metadata.swath_id == requested_swath && metadata.polarisation == requested_polarisation) {
            matches.push_back(std::move(metadata));
        }
    }
    Require(matches.size() == 1, "expected one " + requested_swath + "/" + requested_polarisation + " annotation");
    return std::move(matches.front());
}

}  // namespace alus::s1tbx::slc
