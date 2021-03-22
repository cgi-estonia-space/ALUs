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

#include <any>
#include <memory>

#include "gmock/gmock.h"

#include "snap-core/dataio/decode_qualification.h"
#include "snap-core/dataio/i_product_reader.h"
#include "snap-core/dataio/i_product_reader_plug_in.h"

#include "dummy_product_reader.h"

namespace alus::snapengine {
class DummyProductReaderPlugIn : std::enable_shared_from_this<DummyProductReaderPlugIn>, public IProductReaderPlugIn {
public:
    std::shared_ptr<IProductReader> CreateReaderInstance() override;
    DecodeQualification GetDecodeQualification([[maybe_unused]] const std::any& input) override {
        return DecodeQualification::UNABLE;
    }
};
}  // namespace alus::snapengine