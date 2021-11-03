/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.util.TestUtils.java
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
#include <string_view>

#include <boost/filesystem.hpp>

#include "snap-core/core/datamodel/product_data_utc.h"

namespace alus {
namespace snapengine {

// pre-declare
class Product;
class Band;
class MetadataElement;

/**
 * Utilities for Operator unit tests
 */
class TestUtils {
private:
    static constexpr bool FAIL_ON_SKIP = false;
    static constexpr bool FAIL_ON_LARGE_TEST_PRODUCT = false;
    static constexpr bool FAIL_ON_ALL_NO_DATA = false;
    static constexpr int LARGE_DIMENSION = 100;

    static const Utc& no_time_;

    inline static bool test_environment_initialized_ = false;

    static void AddGeoCoding(const std::shared_ptr<Product>& product);
    static void CompareMetadata(const std::shared_ptr<Product>& test_product,
                                const std::shared_ptr<Product>& expected_product,
                                std::vector<std::string> exemption_list);

public:
    static constexpr std::string_view SKIPTEST{"skipTest"};
    //    static final Logger log = SystemUtils.LOG;

    static void InitTestEnvironment();

    static std::shared_ptr<Product> ReadSourceProduct(const boost::filesystem::path& input_file);

    static void VerifyProduct(const std::shared_ptr<Product>& product, bool verify_times, bool verify_geo_coding);

    static void VerifyProduct(const std::shared_ptr<Product>& product, bool verify_times, bool verify_geo_coding,
                              bool verify_band_data);

    static std::shared_ptr<Product> CreateProduct(std::string_view type, int w, int h);

    static std::shared_ptr<Band> CreateBand(const std::shared_ptr<Product>& test_product, std::string_view band_name,
                                            int w, int h);

    static void AttributeEquals(const std::shared_ptr<MetadataElement>& elem, std::string_view name, double true_value);

    static void AttributeEquals(const std::shared_ptr<MetadataElement>& elem, std::string_view name,
                                std::string_view true_value);

    static void CompareProducts(const std::shared_ptr<Product>& target_product,
                                const std::shared_ptr<Product>& expected_product);

    static void ComparePixels(const std::shared_ptr<Product>& target_product, std::string_view band_name,
                              std::vector<float> expected);

    static void ComparePixels(const std::shared_ptr<Product>& target_product, std::string_view band_name, int x, int y,
                              std::vector<float> expected);

    static void CompareProducts(const std::shared_ptr<Product>& target_product, std::string_view expected_path,
                                std::vector<std::string> exemption_list);

    static void CompareArrays(std::vector<float> actual, std::vector<float> expected, float threshold);

    static bool ContainsProductType(std::vector<std::string> product_type_exemptions, std::string_view product_type);

    static bool SkipTest(const std::any& obj, std::string_view msg);
};

}  // namespace snapengine
}  // namespace alus
