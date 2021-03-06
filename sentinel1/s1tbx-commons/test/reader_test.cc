/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.test.ReaderTest.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include "s1tbx-commons/test/reader_test.h"

#include <any>
#include <stdexcept>

#include "s1tbx-commons/test/metadata_validator.h"
#include "snap-core/core/dataio/decode_qualification.h"
#include "snap-core/core/dataio/i_product_reader.h"
#include "snap-core/core/datamodel/band.h"
#include "snap-core/core/datamodel/product.h"
#include "snap-engine-utilities/engine-utilities/util/test_utils.h"

namespace alus::s1tbx {
std::shared_ptr<snapengine::Product> ReaderTest::TestReader(const boost::filesystem::path& input_path) {
    return TestReader(input_path, reader_plug_in_);
}

std::shared_ptr<snapengine::Product> alus::s1tbx::ReaderTest::TestReader(
    const boost::filesystem::path& input_path, const std::shared_ptr<Sentinel1ProductReaderPlugIn>& reader_plug_in) {
    if (!boost::filesystem::exists(input_path)) {
        //    todo: solve shared_from_this()
        snapengine::TestUtils::SkipTest(std::make_any<std::shared_ptr<ReaderTest>>(shared_from_this()),
                                        input_path.filename().string() + " not found");
        return nullptr;
    }

    const snapengine::DecodeQualification can_read = reader_plug_in->GetDecodeQualification(input_path);
    if (can_read != snapengine::DecodeQualification::INTENDED) {
        throw std::runtime_error("Reader not intended");
    }

    const std::shared_ptr<snapengine::IProductReader> reader = reader_plug_in->CreateReaderInstance();
    std::shared_ptr<snapengine::Product> product = reader->ReadProductNodes(input_path, nullptr);
    if (product == nullptr) {
        throw std::runtime_error("Unable to read product");
    }
    return product;
}

void ReaderTest::ValidateProduct(const std::shared_ptr<snapengine::Product>& product) const {
    snapengine::TestUtils::VerifyProduct(product, verify_time_, verify_geocoding_);
}
void ReaderTest::ValidateMetadata(const std::shared_ptr<snapengine::Product>& product) {
    const auto metadata_validator = std::make_shared<MetadataValidator>(product);
    metadata_validator->Validate();
}
void ReaderTest::ValidateMetadata(const std::shared_ptr<snapengine::Product>& product,
                                  const std::shared_ptr<ValidationOptions>& options) {
    const auto metadata_validator = std::make_shared<MetadataValidator>(product, options);
    metadata_validator->Validate();
}
void ReaderTest::ValidateBands(const std::shared_ptr<snapengine::Product>& trg_product,
                               const std::vector<std::string>& band_names) {
    const std::vector<std::shared_ptr<snapengine::Band>> bands = trg_product->GetBands();
    if (band_names.size() != bands.size()) {
        throw std::runtime_error("Expecting " + std::to_string(band_names.size()) + " bands but found " +
                                 std::to_string(bands.size()));
    }
    for (const auto& band_name : band_names) {
        auto band = trg_product->GetBand(band_name);
        if (band == nullptr) {
            throw std::runtime_error("Band " + band_name + " not found");
        }
        if (!band->GetUnit().has_value()) {
            throw std::runtime_error("Band " + band_name + " is missing a unit");
        }
        if (!band->IsNoDataValueUsed()) {
            throw std::runtime_error("Band " + band_name + " is not using a nodata value");
        }
    }
}
ReaderTest::ReaderTest(const std::shared_ptr<Sentinel1ProductReaderPlugIn>& reader_plug_in)
    : reader_plug_in_(reader_plug_in), reader_(reader_plug_in->CreateReaderInstance()) {}

}  // namespace alus::s1tbx
