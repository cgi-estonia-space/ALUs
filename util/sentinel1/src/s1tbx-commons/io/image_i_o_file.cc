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
#include "s1tbx-commons/io/image_i_o_file.h"

#include <stdexcept>

#include "custom/i_image_reader.h"

namespace alus::s1tbx {

ImageIOFile::ImageIOFile(std::string_view name,
                         [[maybe_unused]] const std::shared_ptr<snapengine::custom::Dimension>& band_dimensions,
                         std::string_view img_path, const std::shared_ptr<snapengine::custom::IImageReader>& iio_reader,
                         int num_images, int num_bands, int data_type,
                         const boost::filesystem::path& product_input_file) {
    name_ = name;
    band_file_path_ = img_path;
    //    todo: think if we need to allocate using dimensions!? think if add local variable which can be used by reader
    //    to allocate? stream_ = input_stream; if (stream_ == nullptr) {
    //        throw std::runtime_error("Unable to open");
    //    }
    reader_ = iio_reader;
    // currently just avoiding streams
    InitReader();
    num_images_ = num_images;
    num_bands_ = num_bands;
    data_type_ = data_type;
    product_input_file_ = product_input_file;
}

// THIS FUNCTIONALITY IS NOT NEEDED FOR SENTINEL1
// ImageIOFile::ImageIOFile(
//    std::string_view name,
//    /* const std::ifstream& input_stream,*/ const std::shared_ptr<snapengine::custom::Dimension>& band_dimensions,
//    std::string_view img_path, const std::shared_ptr<snapengine::custom::IImageReader>& iio_reader,
//    const boost::filesystem::path& product_input_file) {
//    name_ = name;
//    band_file_path_ = img_path;
//    //    stream_ = input_stream;
//    //    if (stream_ == nullptr) {
//    //        throw std::runtime_error("Unable to open");
//    //    }
//    product_input_file_ = product_input_file;
//    CreateReader(iio_reader);
//    reader_ = iio_reader;
//}

// ImageIOFile::ImageIOFile(const boost::filesystem::path& input_file, const
// std::shared_ptr<snapengine::custom::IImageReader>& iio_reader,
//                         const boost::filesystem::path& product_input_file)
//    : ImageIOFile(input_file.filename().string(), /*ImageIO::CreateImageInputStream(input_file),*/ iio_reader,
//                  product_input_file) {}

// std::istream ImageIOFile::CreateImageInputStream(const std::istream& in_stream,
//                                                 const std::shared_ptr<snapengine::custom::Dimension>&
//                                                 band_dimensions) {
//    //        long size = band_dimensions->width*band_dimensions->height;
//    //        return use_file_cache_ || size > 500000000 ? new FileCacheImageInputStream(in_stream, CreateCacheDir())
//    :
//    //        new MemoryCacheImageInputStream(in_stream);
//}

// void ImageIOFile::ReadImageIORasterBand(const int source_offset_x, const int source_offset_y, const int
// source_step_x,
//                                        const int source_step_y,
//                                        const std::shared_ptr<snapengine::ProductData>& dest_buffer,
//                                        const int dest_offset_x, const int dest_offset_y, const int dest_width,
//                                        const int dest_height, const int image_i_d, const int band_sample_offset) {
//    throw std::runtime_error("not yet ported from snap java version");
//}
void ImageIOFile::Close() {
    //    todo: close streams/readers etc..
    throw std::runtime_error("not yet implemented");
}

std::shared_ptr<snapengine::custom::IImageReader> ImageIOFile::GetReader() {
    if (reader_) {
        return reader_;
    }
    throw std::runtime_error("no reader created");
}
void ImageIOFile::InitReader() {
    if (reader_) {
        reader_->SetInputPath(band_file_path_);
    }
}

// SNAP VERSION NEEDS THREAD PROTECTION, THIS FUNCTIONALITY IS NOT NEEDED FOR SENTINEL1
// void ImageIOFile::CreateReader(const std::shared_ptr<snapengine::custom::IImageReader>& iio_reader){
//
//    reader_ = iio_reader;
//    InitReader();
//
//    num_images_ = reader_->GetNumImages();
//    if(num_images_ < 0){
//        num_images_ = 1;
//    }
//    num_bands_ = 3;
//
//    data_type_ = snapengine::ProductData::TYPE_INT32;
//
//    auto its = reader_->GetRawImageType(0);
//    if (its) {
//        num_bands_ = reader_->GetRawImageType(0)->GetNumBands();
//        data_type_ = BufferImageTypeToProductType(its->GetBufferedImageType());
//
//        if (its->GetBufferedImageType() == BufferedImage::TYPE_BYTE_INDEXED) {
//            is_indexed_= true;
//            CreateIndexedImageInfo(its->GetColorModel());
//        }
//    }
//}

}  // namespace alus::s1tbx
