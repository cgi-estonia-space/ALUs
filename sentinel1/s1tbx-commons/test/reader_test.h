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
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <boost/filesystem.hpp>

#include "s1tbx-io/sentinel1/sentinel1_product_reader_plug_in.h"

namespace alus::snapengine {

// pre-declare
class Product;
class IProductReader;
}  // namespace alus::snapengine

namespace alus::s1tbx {

// pre-declare
class ValidationOptions;

/**
 * Paths to common SAR input data
 */
class ReaderTest : public std::enable_shared_from_this<ReaderTest> {
protected:
    const std::shared_ptr<Sentinel1ProductReaderPlugIn> reader_plug_in_;
    const std::shared_ptr<snapengine::IProductReader> reader_;
    bool verify_time_ = true;
    bool verify_geocoding_ = true;

    std::shared_ptr<snapengine::Product> TestReader(
        const boost::filesystem::path& input_path, const std::shared_ptr<Sentinel1ProductReaderPlugIn>& reader_plug_in);

    void ValidateMetadata(const std::shared_ptr<snapengine::Product>& product,
                          const std::shared_ptr<ValidationOptions>& options);

    // todo: check what this is
    //    static {
    //        TestUtils.initTestEnvironment();
    //    }

public:
    explicit ReaderTest(const std::shared_ptr<Sentinel1ProductReaderPlugIn>& reader_plug_in);

    std::shared_ptr<snapengine::Product> TestReader(const boost::filesystem::path& input_path);
    void ValidateProduct(const std::shared_ptr<snapengine::Product>& product);
    void ValidateMetadata(const std::shared_ptr<snapengine::Product>& product);
    void ValidateBands(const std::shared_ptr<snapengine::Product>& trg_product, std::vector<std::string> band_names);
};
}  // namespace alus::s1tbx
