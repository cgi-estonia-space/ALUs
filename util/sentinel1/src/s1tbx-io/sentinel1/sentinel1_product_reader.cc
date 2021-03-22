/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1ProductReader.java
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
#include "sentinel1_product_reader.h"

#include <algorithm>
#include <iostream>

#include "abstract_product_reader.h"
#include "custom/i_image_reader.h"
#include "s1tbx-io/sentinel1/i_sentinel1_directory.h"
#include "s1tbx-io/sentinel1/sentinel1_constants.h"
#include "s1tbx-io/sentinel1/sentinel1_level1_directory.h"
#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"
#include "snap-core/datamodel/quicklooks/quicklook.h"

namespace alus::s1tbx {

void Sentinel1ProductReader::Close() {
    if (data_dir_) {
        data_dir_->Close();
        data_dir_ = nullptr;
    }
    AbstractProductReader::Close();
}

std::shared_ptr<snapengine::Product> Sentinel1ProductReader::ReadProductNodesImpl() {
    try {
        boost::filesystem::path input_path = GetPathFromInput(GetInput());
        if (boost::filesystem::is_directory(input_path)) {
            // todo:compare to snap outcome
            input_path = input_path.append(Sentinel1Constants::PRODUCT_HEADER_NAME);
        }
        if (!boost::filesystem::exists(input_path)) {
            throw std::runtime_error(input_path.string() + " not found");
        }

        if (Sentinel1ProductReaderPlugIn::IsLevel2(input_path)) {
            //            data_dir_ = std::make_shared<Sentinel1Level2Directory>(input_path);
            throw std::runtime_error("currently not supported");
        } else if (Sentinel1ProductReaderPlugIn::IsLevel1(input_path)) {
            data_dir_ = std::make_shared<Sentinel1Level1Directory>(input_path);
        } else if (Sentinel1ProductReaderPlugIn::IsLevel0(input_path)) {
            //            data_dir_ = std::make_shared<Sentinel1Level0Directory>(input_path);
            throw std::runtime_error("currently not supported");
        }
        if (data_dir_ == nullptr) {
            Sentinel1ProductReaderPlugIn::ValidateInput(input_path);
        }
        data_dir_->ReadProductDirectory();
        std::shared_ptr<snapengine::Product> product = data_dir_->CreateProduct();
        product->SetFileLocation(input_path);
        product->SetProductReader(SharedFromBase<Sentinel1ProductReader>());
        // todo: add support if needed
        //        if (std::dynamic_pointer_cast<Sentinel1Level2Directory>(data_dir_)) {
        //            std::dynamic_pointer_cast<Sentinel1Level2Directory>(data_dir_)->AddGeoCodingToBands(product);
        //        }
        AddCommonSARMetadata(product);

        SetQuicklookBandName(product);
        AddQuicklook(product, snapengine::Quicklook::DEFAULT_QUICKLOOK_NAME, GetQuicklookFile());

        product->SetModified(false);

        return product;
    } catch (const std::exception& e) {
        HandleReaderException(e);
    }

    return nullptr;
}
boost::filesystem::path Sentinel1ProductReader::GetQuicklookFile() {
    if (std::dynamic_pointer_cast<Sentinel1Level1Directory>(data_dir_)) {
        std::shared_ptr<Sentinel1Level1Directory> level1_directory =
            std::dynamic_pointer_cast<Sentinel1Level1Directory>(data_dir_);
        try {
            if (level1_directory->Exists(level1_directory->GetRootFolder() + "preview/quick-look.png")) {
                return level1_directory->GetFile(level1_directory->GetRootFolder() + "preview/quick-look.png");
            }
        } catch (const std::exception& e) {
            std::cerr << "Unable to load quicklook " << level1_directory->GetProductName() << std::endl;
        }
    }
    return "";
}
void Sentinel1ProductReader::ReadBandRasterDataImpl(
    int source_offset_x, int source_offset_y, [[maybe_unused]] int source_width, [[maybe_unused]] int source_height,
    int source_step_x, int source_step_y, std::shared_ptr<snapengine::Band> dest_band, int dest_offset_x,
    int dest_offset_y, int dest_width, int dest_height, const std::shared_ptr<snapengine::ProductData>& dest_buffer,
    [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    std::shared_ptr<BandInfo> band_info = data_dir_->GetBandInfo(dest_band);
    if (band_info && band_info->img_) {
        if (data_dir_->IsSLC()) {
            ReadSLCRasterBand(source_offset_x, source_offset_y, source_step_x, source_step_y, dest_buffer,
                              dest_offset_x, dest_offset_y, dest_width, dest_height, band_info);
        } else {
            //            currently only slc
            //            band_info->img_->ReadImageIORasterBand(source_offset_x, source_offset_y, source_step_x,
            //            source_step_y,
            //                                                   dest_buffer, dest_offset_x, dest_offset_y, dest_width,
            //                                                   dest_height, band_info->image_i_d_,
            //                                                   band_info->band_sample_offset_);
        }
    } else {
        //    } else if ( std::dynamic_pointer_cast<Sentinel1Level2Directory>(data_dir_)) {
        //        todo:check if this works like expected
        throw std::runtime_error("not yet ported/supported ");
        //        final Sentinel1Level2Directory s1L1Dir = (Sentinel1Level2Directory) dataDir;
        //        if (s1L1Dir.getOCNReader() == null) {
        //            throw new IOException("Sentinel1OCNReader not found");
        //        }
        //
        //        s1L1Dir.getOCNReader().readData(sourceOffsetX, sourceOffsetY, sourceWidth, sourceHeight,
        //                                        sourceStepX, sourceStepY, destBand, destOffsetX,
        //                                        destOffsetY, destWidth, destHeight, destBuffer);
    }
}
void Sentinel1ProductReader::ReadSLCRasterBand(const int source_offset_x, const int source_offset_y,
                                               const int source_step_x, const int source_step_y,
                                               const std::shared_ptr<snapengine::ProductData>& dest_buffer,
                                               const int dest_offset_x, const int dest_offset_y, int dest_width,
                                               int dest_height, const std::shared_ptr<BandInfo>& band_info) {
    int length;
    std::vector<int32_t> src_array;
    std::shared_ptr<snapengine::custom::Rectangle> dest_rect =
        std::make_shared<snapengine::custom::Rectangle>(dest_offset_x, dest_offset_y, dest_width, dest_height);

    if (use_cache_) {
        //        todo: make sure gdal uses cache (probably some option)
        //        const DataCache.DataKey datakey = new DataCache.DataKey(band_info->img_, dest_rect);
        //        DataCache.Data cached_data = Cache.get(datakey);
        //        if (cached_data && cached_data.valid_) {
        //            src_array = cached_data->int_array_;
        //            length = src_array.size();
        //        } else {
        //            cached_data = ReadRect(datakey, band_info, source_offset_x, source_offset_y, source_step_x,
        //            source_step_y, dest_rect);
        //
        //            src_array = cached_data->int_array;
        //            length = src_array.size();
        //        }
        throw std::runtime_error("currently explicit cache support not yet implemented");
    } else {
        //        todo: make sure gdal does not use cache (probably some option)
        //        DataCache.Data cached_data = ReadRect(nullptr, band_info, source_offset_x, source_offset_y,
        //        source_step_x, source_step_y, dest_rect);
        //
        //        src_array = cached_data.int_array;

        // now use gdal to read rectangle of data!?
        src_array = ReadRect(band_info, source_offset_x, source_offset_y, source_step_x, source_step_y, dest_rect);
        length = src_array.size();
    }

    auto dest_array = std::any_cast<std::vector<int16_t>>(dest_buffer->GetElems());
    //    todo: check if we can read in int16_t (vs. currently 32bits if look into ReadRect) right away and awoid these
    //    shifts here!? currently doing direct port and not understanding why this is done like that
    if (!band_info->is_imaginary_) {
        if (source_step_x == 1) {
            //            todo: optimize if needed and if we have time (c++ has some direct solutions), currently made
            //            rather direct port from java version
            int i = 0;
            for (int src_val : src_array) {
                dest_array.at(i++) = static_cast<int16_t>(src_val);
            }
        } else {
            for (int i = 0; i < length; i += source_step_x) {
                dest_array.at(i) = static_cast<int16_t>(src_array.at(i));
            }
        }
    } else {
        if (source_step_x == 1) {
            int i = 0;
            for (int src_val : src_array) {
                dest_array.at(i++) = static_cast<int16_t>(src_val >> 16);
            }
        } else {
            for (int i = 0; i < length; i += source_step_x) {
                dest_array.at(i) = static_cast<int16_t>(src_array.at(i) >> 16);
            }
        }
    }
}

std::vector<int32_t> Sentinel1ProductReader::ReadRect(const std::shared_ptr<BandInfo>& band_info, int source_offset_x,
                                                      int source_offset_y, int source_step_x, int source_step_y,
                                                      const std::shared_ptr<snapengine::custom::Rectangle>& dest_rect) {
    // now source is subsample region and dest_rect is actual area in original data
    // we need to read destionation from subsampled original
    // currently if we use gdal we should do it in 1 go using rasterio
    // java version subsamples using model and calculates pixels of subsampled target and then creates array just
    // exactly the size it reads subsampled data (this is good check if we have done ok)

    // this and other related stuff comes from band_info and is probably set up during reader init there, simpler to
    // modify later---->>>>>>>>>
    /*std::string file_name{"placeholder"};
    GDALAllRegister();
    auto* dataset = static_cast<GDALDataset*>(GDALOpen(file_name.data(), GA_ReadOnly));
    CHECK_GDAL_PTR(dataset);*/
    // also add other stuff for dataset setup...
    // if possible make alternative using boost::GIL tiff reader
    // this comes from band_info----<<<<<<<<

    // this part should be handled by specific reader method
    //    sampleModel.getSamples(0, 0, dest_width, dest_height, bandInfo.bandSampleOffset, src_array,
    //    data.getDataBuffer());

    //    todo: check this over band_info->band_sample_offset_
    // took "nBandCount" from img = new ImageIOFile(name, imgStream, GeoTiffUtils.getTiffIIOReader(imgStream), 1, 1,
    // ProductData.TYPE_INT32, productInputFile);
    // last option might be band_info->band_sample_offset_ (probably ok to use 0 since single band)
    std::shared_ptr<snapengine::custom::Rectangle> source_rectangle = nullptr;
    if (source_step_x == 1 && source_step_y == 1) {
        //        todo: make sure we have copy constructor rectangle
        source_rectangle = std::make_shared<snapengine::custom::Rectangle>(dest_rect);
    }

    //    nPixelSpace = eBufType (bytes until next pixel)
    //    nLineSpace = eBufType * nBufXSize (bytes until next line)
    // if 1 every column and row will be used
    //    source_x_subsampling  period 1 uses every row/column
    //    source_y_subsampling
    //    subsampling_x_offset  //offset for 1st pixel of region origin
    //    subsampling_y_offset

    //    * The number of subsampled pixels in a scanline is given by
    //    * <p>
    //    * <code>truncate[(width - subsamplingXOffset + sourceXSubsampling - 1)
    //                     * / sourceXSubsampling]</code>.

    //    source_offset_x % sourceStepX

    //[period - (region offset modulo period)] modulo period)
    // todo:this might be already handled elsewhere
    if (source_step_x < 0 || source_step_y < 0) {
        throw std::invalid_argument("sampling step needs to be bigger than 0");
    }

    auto subsampled_data_width = static_cast<std::size_t>(
        std::trunc((source_rectangle->width - source_offset_x % source_step_x + source_step_x - 1) / source_step_x));
    auto subsampled_data_height = static_cast<std::size_t>(
        std::trunc((source_rectangle->height - source_offset_y % source_step_y + source_step_y - 1) / source_step_y));

    // these must already take account of subsampling logic, actual region which gets filled with data, after source
    // data has been subsampled
    auto dest_width = std::min(static_cast<std::size_t>(dest_rect->width), subsampled_data_width);
    auto dest_height = std::min(static_cast<std::size_t>(dest_rect->height), subsampled_data_height);
    std::vector<int32_t> data(dest_width * dest_height);
    //    todo:might need more elegant type mapping from gdalint32 to int32_t?

    // DONE?! java version calculates subsampled data based on destinatuon rectangle attributes and subsampling logic

    // DONE?! calculate destination rectangle with height and databuffer size based on subsampled source
    // with/height/size

    // provide subsampling logic to actual value reading as inputs to skip pixels where needed (last 3 values while
    // reading)

    //    todo: check against actual values in intellij using debugger!

    //    todo: check if we can use GDALDataType::GDT_Int16 for reading
    // java version uses bandInfo.bandSampleOffset, but we have only one band so it should be ok to use default 0 for
    // last parameter GDAL doc states: "The method also takes care of image decimation / replication if the buffer size
    // (nBufXSize x nBufYSize) is different than the size of the region being accessed (nXSize x nYSize)." not sure if
    // it uses origin offsets correctly when doing decimation

    //(std::vector<int>{1}).data() vs nullptr for taking all bands (we only have single band anyway)

    /*
     * auto buffer_type = GDALDataType::GDT_Int32;
    CHECK_GDAL_ERROR(dataset->RasterIO(GF_Read, source_rectangle->x, source_rectangle->y, source_rectangle->width,
                                       source_rectangle->height, data.data(), dest_width, dest_height, buffer_type, 1,
                                       nullptr, 0, 0, 0));
    */

    //    !!!MOVE GDAL BEHIND INTERFACE here
    // read into data using whichever reader it has
    //    todo: provide parameters needed to read using gdal..

    // TODO: TO MAKE THINGS SIMPLE FIRST VERSION USES DESTINATION RECTANGLE MEANING SUBSET IS NOT DEFINED!!!!!!!!!!!,
    // LATER ADD SUPPORT TO DESTINATION RECTANGLE since no subset is defined destination and source are the same, this
    // means also dest_width and dest_height should be same as source_rectangle.width/height (good check for formula)
    band_info->img_->GetReader()->ReadSubSampledData(source_rectangle, data);

    // /*buffer_type*source_step_x, buffer_type * dest_width*source_step_y*/
    //    todo: check over gdal destruction patterns
    // this gets moved?
    return data;
}
Sentinel1ProductReader::Sentinel1ProductReader(const std::shared_ptr<snapengine::IProductReaderPlugIn>& reader_plug_in)
    : SARReader(reader_plug_in) {}

}  // namespace alus::s1tbx
