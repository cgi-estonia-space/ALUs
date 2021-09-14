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
#include "snap-engine-utilities/util/test_utils.h"

#include <stdexcept>

#include <boost/algorithm/string.hpp>
#include <boost/math/special_functions/relative_difference.hpp>

#include "alus_log.h"
#include "ceres-core/i_progress_monitor.h"
#include "snap-core/dataio/product_i_o.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/metadata_attribute.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-core/datamodel/tie_point_geo_coding.h"
#include "snap-core/datamodel/tie_point_grid.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-engine-utilities/datamodel/unit.h"

namespace alus {
namespace snapengine {

const Utc& TestUtils::no_time_ = *new Utc();

void TestUtils::InitTestEnvironment() {
    if (test_environment_initialized_) {
        return;
    }

    try {
        //        todo: think this through later
        //        SystemUtils.init3rdPartyLibs(GPT.class);
        test_environment_initialized_ = true;
    } catch (const std::exception& e) {
        LOGE << e.what();
    }
}

std::shared_ptr<Product> TestUtils::ReadSourceProduct(const boost::filesystem::path& input_file) {
    if (!boost::filesystem::exists(input_file)) {
        throw std::runtime_error(boost::filesystem::canonical(input_file).string() + " not found");
    }
    // todo: if we add more readers in the future, then we can switch to CommonReaders, current port made a shortcut
    //    const std::shared_ptr<Product> product = CommonReaders::ReadProduct(input_file);
    const std::shared_ptr<Product> product = ProductIO::ReadProduct(input_file);
    if (product == nullptr) {
        throw std::runtime_error("Unable to read " + input_file.filename().string());
    }
    return product;
}

void TestUtils::VerifyProduct(const std::shared_ptr<Product>& product, bool verify_times, bool verify_geo_coding) {
    VerifyProduct(product, verify_times, verify_geo_coding, false);
}

void TestUtils::VerifyProduct(const std::shared_ptr<Product>& product, bool verify_times, bool verify_geo_coding,
                              bool verify_band_data) {
    if (product == nullptr) {
        throw std::runtime_error("product is null");
    }

    if (verify_geo_coding && product->GetSceneGeoCoding() == nullptr) {
        LOGW << "Geocoding is null for " << boost::filesystem::canonical(product->GetFileLocation());
    }

    if (product->GetMetadataRoot() == nullptr) {
        throw std::runtime_error("metadataroot is null");
    }
    if (product->GetNumBands() == 0 && verify_band_data) {
        throw std::runtime_error("numbands is zero");
    }
    if (product->GetProductType().empty()) {
        throw std::runtime_error("productType is null");
    }

    if (product->GetSceneRasterWidth() == 0 || product->GetSceneRasterHeight() == 0 ||
        product->GetSceneRasterWidth() == AbstractMetadata::NO_METADATA ||
        product->GetSceneRasterHeight() == AbstractMetadata::NO_METADATA) {
        throw std::runtime_error("product scene raster dimensions are " +
                                 std::to_string(product->GetSceneRasterWidth()) + " x " +
                                 std::to_string(product->GetSceneRasterHeight()));
    }
    if (verify_times) {
        if (product->GetStartTime() == nullptr || product->GetStartTime()->GetMjd() == no_time_.GetMjd()) {
            throw std::runtime_error("startTime is null");
        }
        if (product->GetEndTime() == nullptr || product->GetEndTime()->GetMjd() == no_time_.GetMjd()) {
            throw std::runtime_error("endTime is null");
        }
    }
    if (verify_band_data && FAIL_ON_ALL_NO_DATA) {
        for (auto const& b : product->GetBands()) {
            if (b->GetUnit()->empty()) {
                throw std::runtime_error("band " + b->GetName() + " has null unit");
            }

            // readPixels gets computeTiles to be executed
            const int w = b->GetRasterWidth() / 2;
            const int h = b->GetRasterHeight() / 2;
            if (FAIL_ON_LARGE_TEST_PRODUCT && (w > LARGE_DIMENSION * 2 || h > LARGE_DIMENSION * 2)) {
                throw std::runtime_error("Test product too large " + std::to_string(w) + "," + std::to_string(h));
            }
            const int x0 = w / 2;
            const int y0 = h / 2;

            bool all_no_data = true;
            for (int y = y0; y < y0 + h; ++y) {
                const std::vector<float> float_values(w);
                b->ReadPixels(x0, y, w, 1, float_values, std::make_shared<ceres::NullProgressMonitor>());
                for (float f : float_values) {
                    if (!(f == b->GetNoDataValue() || f == 0 || isnan(f))) {
                        all_no_data = false;
                    }
                }
            }
            if (all_no_data) {
                throw std::runtime_error("band " + b->GetName() + " is all no data value");
            }
        }
    }
}

std::shared_ptr<Product> TestUtils::CreateProduct(std::string_view type, int w, int h) {
    //    const auto product = std::make_shared<Product>("name", type, w, h);
    const auto product = Product::CreateProduct("name", type, w, h);

    product->SetStartTime(AbstractMetadata::ParseUtc("10-MAY-2008 20:30:46.890683"));
    product->SetEndTime(AbstractMetadata::ParseUtc("10-MAY-2008 20:35:46.890683"));
    product->SetDescription(std::make_optional("description"));

    AddGeoCoding(product);

    AbstractMetadata::AddAbstractedMetadataHeader(product->GetMetadataRoot());

    return product;
}

std::shared_ptr<Band> TestUtils::CreateBand(const std::shared_ptr<Product>& test_product, std::string_view band_name,
                                            int w, int h) {
    const std::shared_ptr<Band> band = test_product->AddBand(band_name, ProductData::TYPE_INT32);
    band->SetUnit(snapengine::Unit::AMPLITUDE);
    std::vector<int> int_values(w * h);
    for (int i = 0; i < w * h; i++) {
        int_values.at(i) = i + 1;
    }
    band->SetData(ProductData::CreateInstance(int_values));
    return band;
}

void TestUtils::AddGeoCoding(const std::shared_ptr<Product>& product) {
    const auto lat_grid =
        std::make_shared<TiePointGrid>("lat", 2, 2, 0.5f, 0.5f, product->GetSceneRasterWidth(),
                                       product->GetSceneRasterHeight(), std::vector<float>{10.0f, 10.0f, 5.0f, 5.0f});
    const auto lon_grid = std::make_shared<TiePointGrid>(
        "lon", 2, 2, 0.5f, 0.5f, product->GetSceneRasterWidth(), product->GetSceneRasterHeight(),
        std::vector<float>{10.0f, 10.0f, 5.0f, 5.0f}, TiePointGrid::DISCONT_AT_360);
    const auto tp_geo_coding = std::make_shared<TiePointGeoCoding>(lat_grid, lon_grid);

    product->AddTiePointGrid(lat_grid);
    product->AddTiePointGrid(lon_grid);
    product->SetSceneGeoCoding(tp_geo_coding);
}

void TestUtils::AttributeEquals(const std::shared_ptr<MetadataElement>& elem, std::string_view name,
                                double true_value) {
    const double val = elem->GetAttributeDouble(name, 0);
    if (boost::math::epsilon_difference(val, true_value) != 0) {
        if (boost::math::epsilon_difference(static_cast<float>(val), static_cast<float>(true_value)) != 0) {
            throw std::runtime_error(std::string(name) + " is " + std::to_string(val) + ", expecting " +
                                     std::to_string(true_value));
        }
    }
}

void TestUtils::AttributeEquals(const std::shared_ptr<MetadataElement>& elem, std::string_view name,
                                std::string_view true_value) {
    const std::string val(elem->GetAttributeString(name, ""));
    if (val != true_value) {
        throw std::runtime_error(std::string(name) + " is " + val + ", expecting " + std::string(true_value));
    }
}

void TestUtils::CompareMetadata(const std::shared_ptr<Product>& test_product,
                                const std::shared_ptr<Product>& expected_product,
                                std::vector<std::string> exemption_list) {
    const std::shared_ptr<MetadataElement> test_abs_root = AbstractMetadata::GetAbstractedMetadata(test_product);
    if (test_abs_root == nullptr) {
        throw std::runtime_error("Metadata is null");
    }

    const std::shared_ptr<MetadataElement> expected_abs_root =
        AbstractMetadata::GetAbstractedMetadata(expected_product);
    if (expected_abs_root == nullptr) {
        throw std::runtime_error("Metadata is null");
    }

    if (!exemption_list.empty()) {
        std::sort(exemption_list.begin(), exemption_list.end());
    }

    const std::vector<std::shared_ptr<MetadataAttribute>> attrib_list = expected_abs_root->GetAttributes();
    for (const auto& expected_attrib : attrib_list) {
        if (!exemption_list.empty() &&
            std::binary_search(exemption_list.begin(), exemption_list.end(), expected_attrib->GetName())) {
            continue;
        }

        std::shared_ptr<MetadataAttribute> result = test_abs_root->GetAttribute(expected_attrib->GetName());
        if (result == nullptr) {
            throw std::runtime_error("Metadata attribute " + expected_attrib->GetName() + " is missing");
        }
        const std::shared_ptr<ProductData> expected_data = result->GetData();
        if (!expected_data->EqualElems(expected_attrib->GetData())) {
            auto condition1 = expected_data->ToString();
            auto condition2 = result->GetData()->ToString();
            boost::algorithm::trim(condition1);
            boost::algorithm::trim(condition2);
            if ((expected_data->GetType() == ProductData::TYPE_FLOAT32 ||
                 expected_data->GetType() == ProductData::TYPE_FLOAT64) &&
                boost::math::epsilon_difference(expected_data->GetElemDouble(), result->GetData()->GetElemDouble()) ==
                    0) {
            } else if (boost::iequals(condition1, condition2)) {
                // exactly like snap has it
            } else {
                throw std::runtime_error("Metadata attribute " + expected_attrib->GetName() + " expecting " +
                                         expected_attrib->GetData()->ToString() + " got " +
                                         result->GetData()->ToString());
            }
        }
    }
}

void TestUtils::CompareProducts(const std::shared_ptr<Product>& target_product,
                                const std::shared_ptr<Product>& expected_product) {
    // compare updated metadata
    CompareMetadata(target_product, expected_product, std::vector<std::string>{});

    if (target_product->GetNumBands() != expected_product->GetNumBands()) {
        throw std::runtime_error("Different number of bands");
    }

    if (!target_product->IsCompatibleProduct(expected_product.get(), 0)) throw std::runtime_error("Geocoding is different");

    for (const auto& expected_t_p_g : expected_product->GetTiePointGrids()) {
        const std::shared_ptr<TiePointGrid> trg_t_p_g = target_product->GetTiePointGrid(expected_t_p_g->GetName());
        if (trg_t_p_g == nullptr) {
            throw std::runtime_error("TPG " + expected_t_p_g->GetName() + " not found");
        }

        const std::vector<float> expected_tie_points = expected_t_p_g->GetTiePoints();
        const std::vector<float> trg_tie_points = trg_t_p_g->GetTiePoints();

        if (trg_tie_points != expected_tie_points) {
            throw std::runtime_error("TPGs are different in file " +
                                     boost::filesystem::canonical(expected_product->GetFileLocation()).string());
        }
    }

    for (const auto& expected_band : expected_product->GetBands()) {
        const std::shared_ptr<Band> trg_band = target_product->GetBand(expected_band->GetName());
        if (trg_band == nullptr) {
            throw std::runtime_error("Band " + expected_band->GetName() + " not found");
        }

        const std::vector<float> float_values(2500);
        trg_band->ReadPixels(40, 40, 50, 50, float_values, std::make_shared<ceres::NullProgressMonitor>());

        const std::vector<float> expected_values(2500);
        expected_band->ReadPixels(40, 40, 50, 50, expected_values, std::make_shared<ceres::NullProgressMonitor>());

        if (float_values != expected_values) {
            throw std::runtime_error("Pixels are different in file " +
                                     boost::filesystem::canonical(expected_product->GetFileLocation()).string());
        }
    }
}

void TestUtils::ComparePixels(const std::shared_ptr<Product>& target_product, std::string_view band_name,
                              std::vector<float> expected) {
    ComparePixels(target_product, band_name, 0, 0, expected);
}

void TestUtils::ComparePixels(const std::shared_ptr<Product>& target_product, std::string_view band_name, int x, int y,
                              std::vector<float> expected) {
    const std::shared_ptr<Band> band = target_product->GetBand(band_name);
    if (band == nullptr) {
        throw std::runtime_error(std::string(band_name) + " not found");
    }

    std::vector<float> actual(expected.size());
    band->ReadPixels(x, y, expected.size(), 1, actual, std::make_shared<ceres::NullProgressMonitor>());

    for (std::size_t i = 0; i < expected.size(); ++i) {
        if ((std::abs(expected.at(i) - actual.at(i)) > 0.0001)) {
            std::string msg = "actual:";
            for (float an_actual : actual) {
                msg += std::to_string(an_actual) + ", ";
            }
            LOGE << msg;
            msg = "expected:";

            for (float an_expected : expected) {
                msg += std::to_string(an_expected) + ", ";
            }
            LOGE << msg;
            throw std::runtime_error("Mismatch [" + std::to_string(i) + "] " + std::to_string(actual.at(i)) +
                                     " is not " + std::to_string(expected.at(i)) + " for " + target_product->GetName() +
                                     " band:" + std::string(band_name));
        }
    }
}

void TestUtils::CompareProducts(const std::shared_ptr<Product>& target_product, std::string_view expected_path,
                                std::vector<std::string> exemption_list) {
    const std::shared_ptr<Band> target_band = target_product->GetBandAt(0);
    if (target_band == nullptr) {
        throw std::runtime_error("targetBand at 0 is null");
    }

    // readPixels: execute computeTiles()
    const std::vector<float> float_values(2500);
    target_band->ReadPixels(40, 40, 50, 50, float_values, std::make_shared<ceres::NullProgressMonitor>());

    // compare with expected outputs:
    const boost::filesystem::path expected_file{std::string(expected_path)};
    if (!boost::filesystem::exists(expected_file)) {
        throw std::runtime_error("Expected file not found " + expected_file.filename().string());
    }

    //    todo: ok temporary solution could just provide SAFE reader to avoid porting more
    const std::shared_ptr<IProductReader> reader2 = ProductIO::GetProductReaderForInput(expected_file);

    const std::shared_ptr<Product> expected_product = reader2->ReadProductNodes(expected_file, nullptr);
    const std::shared_ptr<Band> expected_band = expected_product->GetBandAt(0);

    const std::vector<float> expected_values(2500);
    expected_band->ReadPixels(40, 40, 50, 50, expected_values, std::make_shared<ceres::NullProgressMonitor>());
    if (float_values != expected_values) {
        throw std::runtime_error("Pixels are different in file " + std::string(expected_path));
    }

    // compare updated metadata
    CompareMetadata(target_product, expected_product, exemption_list);
}

void TestUtils::CompareArrays(std::vector<float> actual, std::vector<float> expected, float threshold) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("The actual array and expected array have different lengths");
    }

    for (std::size_t i = 0; i < actual.size(); ++i) {
        if ((std::abs(expected.at(i) - actual.at(i)) > threshold)) {
            std::string msg = "actual:";
            for (float an_actual : actual) {
                msg += std::to_string(an_actual) + ", ";
            }
            LOGE << msg;

            msg = "expected:";
            for (float an_expected : expected) {
                msg += std::to_string(an_expected) + ", ";
            }
            LOGE << msg;

            throw std::runtime_error("Mismatch [" + std::to_string(i) + "] " + std::to_string(actual.at(i)) +
                                     " is not " + std::to_string(expected.at(i)));
        }
    }
}

bool TestUtils::ContainsProductType(std::vector<std::string> product_type_exemptions, std::string_view product_type) {
    if (!product_type_exemptions.empty()) {
        for (const auto& str : product_type_exemptions) {
            if (product_type.find(str) != std::string::npos) {
                return true;
            }
        }
    }
    return false;
}

bool TestUtils::SkipTest(const std::any& obj, std::string_view msg) {
    LOGW << obj.type().name() << " skipped " << msg;
    if (FAIL_ON_SKIP) {
        throw std::runtime_error(std::string(obj.type().name()) + " skipped " + std::string(msg));
    }
    return true;
}

}  // namespace snapengine
}  // namespace alus
