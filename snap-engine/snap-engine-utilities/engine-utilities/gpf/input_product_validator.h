/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.InputProductValidator.java
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
#include <string_view>
#include <vector>

namespace alus::snapengine {

class MetadataElement;
class Product;

class InputProductValidator {
private:
    std::shared_ptr<Product> product_;
    std::shared_ptr<MetadataElement> abs_root_;

    static constexpr std::string_view SHOULD_BE_SAR_PRODUCT = "Input should be a SAR product";
    static constexpr std::string_view SHOULD_NOT_BE_LEVEL0 = "Level-0 RAW products are not supported";
    static constexpr std::string_view SHOULD_BE_COREGISTERED = "Input should be a coregistered stack.";
    static constexpr std::string_view SHOULD_BE_SLC = "Input should be a single look complex SLC product.";
    static constexpr std::string_view SHOULD_BE_GRD = "Input should be a detected product.";
    static constexpr std::string_view SHOULD_BE_S1 = "Input should be a Sentinel-1 product.";
    static constexpr std::string_view SHOULD_BE_DEBURST = "Source product should first be deburst.";
    static constexpr std::string_view SHOULD_BE_MULTISWATH_SLC =
        "Source product should be multi sub-swath SLC burst product.";
    static constexpr std::string_view SHOULD_BE_QUAD_POL = "Input should be a full pol SLC product.";
    static constexpr std::string_view SHOULD_BE_CALIBRATED = "Source product should be calibrated.";
    static constexpr std::string_view SHOULD_NOT_BE_CALIBRATED = "Source product has already been calibrated.";
    static constexpr std::string_view SHOULD_BE_MAP_PROJECTED = "Source product should be map projected.";
    static constexpr std::string_view SHOULD_NOT_BE_MAP_PROJECTED = "Source product should not be map projected.";
    static constexpr std::string_view SHOULD_BE_COMPATIBLE =
        "Source products do not have compatible dimensions and geocoding.";

    static constexpr float GEOGRAPHIC_ERROR = 1.0e-3F;

    static bool Contains(const std::vector<std::string>& list, std::string_view tag);

public:
    explicit InputProductValidator(const std::shared_ptr<Product>& product);
    bool IsSARProduct();
    void CheckIfSARProduct();
    void CheckIfCoregisteredStack();
    bool IsComplex();
    void CheckIfSLC();
    void CheckIfGRD();
    bool IsMultiSwath();
    bool IsSentinel1Product();
    void CheckIfSentinel1Product();
    void CheckMission(const std::vector<std::string>& valid_missions);
    void CheckProductType(const std::vector<std::string>& valid_product_types);
    void CheckAcquisitionMode(const std::vector<std::string>& valid_modes);
    bool IsTOPSARProduct();
    void CheckIfTOPSARBurstProduct(bool shouldbe);
    void CheckIfMultiSwathTOPSARProduct();
    bool IsDebursted();
    bool IsFullPolSLC();
    void CheckIfQuadPolSLC();
    static bool IsMapProjected(const std::shared_ptr<Product>& product);
    bool IsCalibrated();
    void CheckIfCalibrated(bool should_be);
    void CheckIfTanDEMXProduct();
    void CheckIfCompatibleProducts(std::vector<std::shared_ptr<Product>> source_products);
};

}  // namespace alus::snapengine