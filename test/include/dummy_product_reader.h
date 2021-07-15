/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.ProductTest.java
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
#pragma once

#include <memory>
#include <stdexcept>

#include "ceres-core/core/i_progress_monitor.h"
#include "dummy_product_reader_plug_in.h"
#include "snap-core/core/dataio/abstract_product_reader.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/product.h"

namespace alus::snapengine {

class DummyProductReader : public AbstractProductReader {
protected:
    void ReadBandRasterDataImpl([[maybe_unused]] int source_offset_x, [[maybe_unused]] int source_offset_y,
                                [[maybe_unused]] int source_width, [[maybe_unused]] int source_height,
                                [[maybe_unused]] int source_step_x, [[maybe_unused]] int source_step_y,
                                [[maybe_unused]] std::shared_ptr<Band> dest_band, [[maybe_unused]] int dest_offset_x,
                                [[maybe_unused]] int dest_offset_y, [[maybe_unused]] int dest_width,
                                [[maybe_unused]] int dest_height,
                                [[maybe_unused]] const std::shared_ptr<ProductData>& dest_buffer,
                                [[maybe_unused]] std::shared_ptr<alus::ceres::IProgressMonitor> pm) override {
        throw std::runtime_error("not implemented");
    }

public:
    void Close() override {}
    std::shared_ptr<Product> ReadProductNodesImpl() override { throw std::runtime_error("not implemented"); }
    explicit DummyProductReader(const std::shared_ptr<IProductReaderPlugIn>& plug_in);
};

}  // namespace alus::snapengine