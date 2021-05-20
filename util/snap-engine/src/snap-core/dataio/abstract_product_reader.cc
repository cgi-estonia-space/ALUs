/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.AbstractProductReader.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
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
#include "abstract_product_reader.h"

#include <any>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "ceres-core/i_progress_monitor.h"
#include "guardian.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/product_data.h"

namespace alus::snapengine {

std::any AbstractProductReader::GetInput() { return input_; }

void AbstractProductReader::ReadBandRasterData(std::shared_ptr<Band> dest_band, int dest_offset_x, int dest_offset_y,
                                               int dest_width, int dest_height,
                                               std::shared_ptr<ProductData> dest_buffer,
                                               std::shared_ptr<ceres::IProgressMonitor> pm) {
    Guardian::AssertNotNull("dest_band", dest_band);
    Guardian::AssertNotNull("dest_buffer", dest_buffer);

    if (dest_buffer->GetNumElems() < dest_width * dest_height) {
        throw std::invalid_argument("destination buffer too small");
    }
    if (dest_buffer->GetNumElems() > dest_width * dest_height) {
        throw std::invalid_argument("destination buffer too big");
    }

    int source_offset_x = 0;
    int source_offset_y = 0;
    int source_step_x = 1;
    int source_step_y = 1;
    if (GetSubsetDef() != nullptr) {
        source_step_x = GetSubsetDef()->GetSubSamplingX();
        source_step_y = GetSubsetDef()->GetSubSamplingY();
        if (!GetSubsetDef()->GetRegionMap().empty() &&
            (GetSubsetDef()->GetRegionMap().find(dest_band->GetName()) != GetSubsetDef()->GetRegionMap().end())) {
            source_offset_x = GetSubsetDef()->GetRegionMap().at(dest_band->GetName()).x;
            source_offset_y = GetSubsetDef()->GetRegionMap().at(dest_band->GetName()).y;
        } else if (GetSubsetDef()->GetRegion() != nullptr) {
            source_offset_x = GetSubsetDef()->GetRegion()->x;
            source_offset_y = GetSubsetDef()->GetRegion()->y;
        }
    }
    source_offset_x += source_step_x * dest_offset_x;
    source_offset_y += source_step_y * dest_offset_y;
    int source_width = source_step_x * (dest_width - 1) + 1;
    int source_height = source_step_y * (dest_height - 1) + 1;

    ReadBandRasterDataImpl(source_offset_x, source_offset_y, source_width, source_height, source_step_x, source_step_y,
                           dest_band, dest_offset_x, dest_offset_y, dest_width, dest_height, dest_buffer, pm);
}

void AbstractProductReader::Close() {
    std::cerr << "AbstractProductReader.close(): " << ToString() << std::endl;
    input_ = nullptr;
    subset_def_ = nullptr;
}

bool AbstractProductReader::IsMetadataIgnored() {
    bool ignore_metadata = false;
    if (subset_def_ != nullptr) {
        ignore_metadata = subset_def_->IsIgnoreMetadata();
    }
    return ignore_metadata;
}

bool AbstractProductReader::IsNodeAccepted(std::string_view name) {
    return GetSubsetDef() == nullptr || GetSubsetDef()->IsNodeAccepted(name);
}

std::shared_ptr<Product> AbstractProductReader::ReadProductNodes(std::any input,
                                                                 std::shared_ptr<ProductSubsetDef> subset_def) {
    // (nf, 26.09.2007) removed (input == null) check, null inputs (= no sources) shall be allowed
    if (!input.has_value() && !IsInstanceOfValidInputType(input)) {
        throw std::invalid_argument("invalid input source: " + std::string(input.type().name()));
    }
    SetInput(input);
    SetSubsetDef(subset_def);

    //    todo: do we want these logs? also class name might not be correct, might need more work based on intentsions
    auto start = std::chrono::high_resolution_clock::now();
    std::cerr << "Start reading the product from input '" << input.type().name() << "' using the '"
              << typeid(*this).name() << "' reader class. The subset is '" << subset_def << "'." << std::endl;

    std::shared_ptr<Product> product = ReadProductNodesImpl();
    ConfigurePreferredTileSize(product);
    product->SetModified(false);
    if (product->GetProductReader() == nullptr) {
        product->SetProductReader(shared_from_this());
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    std::cerr << "Finish reading the product from input '" << input.type().name() << "' using the '"
              << typeid(*this).name() << "' reader class. The time elapsed is " << elapsed_seconds.count()
              << " milliseconds." << std::endl;

    return product;
}

bool AbstractProductReader::IsInstanceOfValidInputType([[maybe_unused]] std::any input) {
    //    todo::replace plugin validation with some validator class?
    //    if (getReaderPlugIn() != null) {
    //        Class[] inputTypes = getReaderPlugIn().getInputTypes();
    //        for (Class inputType : inputTypes) {
    //            if (inputType.isInstance(input)) {
    //                return true;
    //            }
    //        }
    //        return false;
    //    }
    return true;
}

void AbstractProductReader::ConfigurePreferredTileSize(const std::shared_ptr<Product>& product) {
    // currently not sure where we keep config
    std::shared_ptr<custom::Dimension> new_size =
        //        todo: provide some config file to read default preferences from?
        //        GetConfiguredTileSize(product, Config.instance().preferences().get(SYSPROP_READER_TILE_WIDTH, 0),
        //        Config.instance().preferences().get(SYSPROP_READER_TILE_HEIGHT, 0));
        GetConfiguredTileSize(product, 0, 0);
    if (new_size != nullptr) {
        std::shared_ptr<custom::Dimension> old_size = product->GetPreferredTileSize();
        if (old_size == nullptr) {
            product->SetPreferredTileSize(new_size);
            std::cerr << "Product '" << std::static_pointer_cast<ProductNode>(product)->GetName()
                      << "': tile size set to " << new_size->width << " x " << new_size->height << " pixels"
                      << std::endl;
        } else if (old_size != new_size) {
            product->SetPreferredTileSize(new_size);
            std::cerr << "Product '" << std::static_pointer_cast<ProductNode>(product)->GetName()
                      << "': tile size set to " << new_size->width << " x " << new_size->height << " pixels"
                      << std::endl;
        }
    }
}

void AbstractProductReader::SetInput(std::any input) { input_ = input; }

std::shared_ptr<custom::Dimension> AbstractProductReader::GetConfiguredTileSize(std::shared_ptr<Product> product,
                                                                        std::string_view tile_width_str,
                                                                        std::string_view tile_height_str) {
    int tile_width = ParseTileSize(tile_width_str, product->GetSceneRasterWidth());
    int tile_height = ParseTileSize(tile_height_str, product->GetSceneRasterHeight());
    std::shared_ptr<custom::Dimension> new_size = nullptr;
    if (tile_width != 0 || tile_height != 0) {
        std::shared_ptr<custom::Dimension> old_size = product->GetPreferredTileSize();
        if (tile_width == 0) {
            // Note: tile_height will not be null
            tile_width = (old_size != 0 ? old_size->width : std::min(product->GetSceneRasterWidth(), tile_height));
        }
        if (tile_height == 0) {
            // Note: tile_width will not be null
            tile_height = (old_size != 0 ? old_size->height : std::min(product->GetSceneRasterHeight(), tile_width));
        }
        new_size = std::make_shared<custom::Dimension>(tile_width, tile_height);
    }
    return new_size;
}

int AbstractProductReader::ParseTileSize(std::string_view size_str, int max_size) {
    int size = 0;
    if (!size_str.empty()) {
        if (size_str == "*") {
            size = max_size;
        } else {
            try {
                size = stoi(std::string(size_str));
            } catch (std::invalid_argument& e) {
                // ignore
                std::cerr << "ParseTileSize string to integer operation got invalid argument exception, used parameter "
                          << size_str << ", returned error message: " << e.what() << std::endl;
            } catch (std::out_of_range& e) {
                // ignore
                std::cerr << "ParseTileSize string to integer operation got out of range exception, used parameter "
                          << size_str << ", returned error message: " << e.what() << std::endl;
            }
        }
    }
    return size;
}
// todo:improve this, inital solution is temporary
std::string AbstractProductReader::ToString() { return typeid(*this).name() /*+ "[input=" + input_ + "]"*/; }

// void AbstractProductReader::ReadTiePointGridRasterData(std::shared_ptr<TiePointGrid> tpg, int dest_offset_x,
//                                                       int dest_offset_y, int dest_width, int dest_height,
//                                                       std::shared_ptr<ProductData> dest_buffer,
//                                                       std::shared_ptr<ceres::IProgressMonitor> pm) {
//    IProductReader::ReadTiePointGridRasterData(tpg, dest_offset_x, dest_offset_y, dest_width, dest_height,
//    dest_buffer,
//                                               pm);

}  // namespace alus::snapengine
