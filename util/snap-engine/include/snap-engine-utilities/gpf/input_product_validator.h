#pragma once

#include <memory>
#include <string_view>
#include <vector>

namespace alus {
namespace snapengine {

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

    static bool Contains(std::vector<std::string> list, std::string_view tag);

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
    void CheckMission(std::vector<std::string> valid_missions);
    void CheckProductType(std::vector<std::string> valid_product_types);
    void CheckAcquisitionMode(std::vector<std::string> valid_modes);
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

}  // namespace snapengine
}  // namespace alus
