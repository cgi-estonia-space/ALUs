/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1ProductReaderPlugin.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
 *
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
#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"

#include <optional>
#include <stdexcept>
#include <string>

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>

#include "s1tbx-io/sentinel1/sentinel1_constants.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader.h"
#include "snap-engine-utilities/engine-utilities/gpf/reader_utils.h"
#include "snap-engine-utilities/engine-utilities/util/zip_utils.h"

namespace alus::s1tbx {

bool Sentinel1ProductReaderPlugIn::IsLevel1(const boost::filesystem::path& path) {
    if (snapengine::ZipUtils::IsZip(path)) {
        if (snapengine::ZipUtils::FindInZip(path, "s1", ".tiff")) {
            return true;
        }
        std::string name = path.filename().string();
        boost::algorithm::to_upper(name);
        return boost::algorithm::contains(name, "_1AS") || boost::algorithm::contains(name, "_1AD") ||
               boost::algorithm::contains(name, "_1SS") || boost::algorithm::contains(name, "_1SD");
    }
    boost::filesystem::path annotation_folder = path.parent_path().append(ANNOTATION);
    return boost::filesystem::exists(annotation_folder) && boost::filesystem::is_directory(annotation_folder) &&
           CheckFolder(annotation_folder, ".xml");
}

bool Sentinel1ProductReaderPlugIn::IsLevel2(const boost::filesystem::path& path) {
    if (snapengine::ZipUtils::IsZip(path)) {
        return snapengine::ZipUtils::FindInZip(path, "s1", ".nc");
    }
    boost::filesystem::path measurement_folder = path.parent_path().append(MEASUREMENT);
    return boost::filesystem::exists(measurement_folder) && boost::filesystem::is_directory(measurement_folder) &&
           CheckFolder(measurement_folder, ".nc");
}

bool Sentinel1ProductReaderPlugIn::IsLevel0(const boost::filesystem::path& path) {
    if (snapengine::ZipUtils::IsZip(path)) {
        return snapengine::ZipUtils::FindInZip(path, "s1", ".dat");
    }
    return CheckFolder(path.parent_path(), ".dat");
}

bool Sentinel1ProductReaderPlugIn::CheckFolder(const boost::filesystem::path& folder, std::string_view extension) {
    for (const boost::filesystem::directory_entry& f : boost::filesystem::directory_iterator(folder)) {
        auto name = f.path().filename().string();
        boost::algorithm::to_lower(name);

        if (boost::filesystem::is_regular_file(f)) {
            for (auto const& prefix : ANNOTATION_PREFIXES) {
                if (boost::algorithm::starts_with(name, prefix) &&
                    (extension.empty() || boost::algorithm::ends_with(name, extension))) {
                    return true;
                }
            }
        }
    }
    return false;
}
void Sentinel1ProductReaderPlugIn::ValidateInput(const boost::filesystem::path& path) {
    if (snapengine::ZipUtils::IsZip(path)) {
        if (!snapengine::ZipUtils::FindInZip(path, "s1", ".tiff")) {
            throw std::runtime_error("measurement folder is missing in product");
        }
    } else {
        if (!boost::filesystem::exists(path.parent_path().append(ANNOTATION))) {
            throw std::runtime_error("annotation folder is missing in product");
        }
    }
}
snapengine::DecodeQualification Sentinel1ProductReaderPlugIn::GetDecodeQualification(const std::any& input) {
    std::optional<boost::filesystem::path> path = snapengine::ReaderUtils::GetPathFromInput(input);
    if (path) {
        boost::filesystem::path clear_path = path.value();
        if (boost::filesystem::is_directory(path.value())) {
            //            todo: make sure this is correct
            clear_path = boost::filesystem::canonical(
                boost::filesystem::path(std::string(Sentinel1Constants::PRODUCT_HEADER_NAME)), clear_path);
            if (!boost::filesystem::exists(clear_path)) {
                return snapengine::DecodeQualification::UNABLE;
            }
        }

        std::string filename = clear_path.filename().string();
        boost::algorithm::to_lower(filename);
        if (filename == std::string(Sentinel1Constants::PRODUCT_HEADER_NAME)) {
            if (IsLevel1(clear_path) || IsLevel2(clear_path) || IsLevel0(clear_path)) {
                return snapengine::DecodeQualification::INTENDED;
            }
        }
        if (boost::algorithm::ends_with(filename, ".zip") && boost::algorithm::starts_with(filename, "s1") &&
            (snapengine::ZipUtils::FindInZip(clear_path, "s1", std::string(Sentinel1Constants::PRODUCT_HEADER_NAME)) ||
             snapengine::ZipUtils::FindInZip(clear_path, "rs2",
                                             std::string(Sentinel1Constants::PRODUCT_HEADER_NAME)))) {
            return snapengine::DecodeQualification::INTENDED;
        }
        if (boost::algorithm::starts_with(filename, "s1") && boost::algorithm::ends_with(filename, ".safe") &&
            boost::filesystem::is_directory(clear_path)) {
            auto manifest = boost::filesystem::canonical(
                boost::filesystem::path(std::string(Sentinel1Constants::PRODUCT_HEADER_NAME)), clear_path);
            if (boost::filesystem::exists(manifest)) {
                if (IsLevel1(manifest) || IsLevel2(manifest) || IsLevel0(manifest)) {
                    return snapengine::DecodeQualification::INTENDED;
                }
            }
        }
    }
    // todo zip stream

    return snapengine::DecodeQualification::UNABLE;
}

std::shared_ptr<snapengine::IProductReader> Sentinel1ProductReaderPlugIn::CreateReaderInstance() {
    return std::make_shared<Sentinel1ProductReader>(shared_from_this());
}
}  // namespace alus::s1tbx
