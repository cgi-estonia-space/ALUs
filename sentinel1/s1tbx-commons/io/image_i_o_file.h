/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.io.ImageIOFile.java
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
#pragma once

#include <fstream>
#include <memory>
#include <string>
#include <string_view>

#include <boost/filesystem.hpp>

#include "custom/dimension.h"
#include "snap-core/core/datamodel/index_coding.h"

namespace alus::snapengine {
class IndexCoding;
namespace custom {
class IImageReader;
}  // namespace custom
}  // namespace alus::snapengine
namespace alus::s1tbx {

/**
 * Reader for ImageIO File
 *
 * NOTE: Unlike ESA SNAP variant using streams, ported solution will get GDAL reader hardwired (still keeping changes in
 * mind if we decide to change it in the future)
 */
class ImageIOFile {
private:
    static constexpr bool USE_FILE_CACHE =
        false;  // Config.instance().preferences().getBoolean("s1tbx.readers.useFileCache", false);
    std::string name_;
    int scene_width_ = 0;
    int scene_height_ = 0;
    int data_type_;
    int num_images_ = 1;
    int num_bands_ = 1;
    // todo: purpose and how to port ImageInfo needs more investigation, currently looks like something similar in
    // https://www.boost.org/doc/libs/1_75_0/libs/gil/doc/html/io.html?highlight=reader std::shared_ptr<ImageInfo>
    // image_info_ = nullptr;
    std::shared_ptr<snapengine::IndexCoding> index_coding_ = nullptr;
    bool is_indexed_ = false;
    boost::filesystem::path product_input_file_;
    //  keeping stream option for the future, our alternative solution uses input for stream as input for
    //  reader(band_file_path_)
    //    std::ifstream stream_;
    std::string band_file_path_;
    std::shared_ptr<snapengine::custom::IImageReader> reader_ = nullptr;
    // todo: purpose and how to exactly port ImageReader needs more investigation, currently looking into
    // https://www.boost.org/doc/libs/1_75_0/libs/gil/doc/html/io.html?highlight=reader

public:
    // THIS CONSTRUCTOR IS NOT NEEDED FOR SENTINEL1
    ImageIOFile(std::string_view name, const std::shared_ptr<snapengine::custom::Dimension>& band_dimensions,
                std::string_view img_path, const std::shared_ptr<snapengine::custom::IImageReader>& iio_reader,
                const boost::filesystem::path& product_input_file);

    ImageIOFile(std::string_view name,
                [[maybe_unused]] const std::shared_ptr<snapengine::custom::Dimension>& band_dimensions,
                std::string_view img_path, const std::shared_ptr<snapengine::custom::IImageReader>& iio_reader, int num_images,
                int num_bands, int data_type, const boost::filesystem::path& product_input_file);

    void InitReader();

    [[nodiscard]] std::string GetName() const { return name_; }
    [[nodiscard]] int GetNumImages() const { return num_images_; }
    [[nodiscard]] int GetNumBands() const { return num_bands_; }

    void ReadImageIORasterBand(int source_offset_x, int source_offset_y, int source_step_x, int source_step_y,
                               const std::shared_ptr<snapengine::ProductData>& dest_buffer, int dest_offset_x,
                               int dest_offset_y, int dest_width, int dest_height, int image_i_d,
                               int band_sample_offset);

    void Close();

    std::shared_ptr<snapengine::custom::IImageReader> GetReader();
};

}  // namespace alus::s1tbx
