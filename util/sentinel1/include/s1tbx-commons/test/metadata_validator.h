#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace alus::snapengine {
class Product;
class MetadataElement;
}  // namespace alus::snapengine

namespace alus::s1tbx {

class ValidationOptions {
public:
    bool validate_orbit_state_vectors_ = true;
};

class MetadataValidator {
private:
    const std::shared_ptr<snapengine::Product> product_;
    const std::shared_ptr<snapengine::MetadataElement> abs_root_;
    const std::shared_ptr<ValidationOptions> validation_options_;

    void VerifySRGR();
    void VerifyOrbitStateVectors();
    void VerifyDopplerCentroids();
    void VerifyStr(std::string_view tag);
    void VerifyStr(std::string_view tag, std::vector<std::string> allowed_str);
    void VerifyDouble(std::string_view tag);
    void VerifyInt(std::string_view tag);
    void VerifyUTC(std::string_view tag);

public:
    MetadataValidator(const std::shared_ptr<snapengine::Product>& product);
    MetadataValidator(const std::shared_ptr<snapengine::Product>& product,
                      const std::shared_ptr<ValidationOptions>& options);
    void Validate();
    void ValidateOptical();
    void ValidateSAR();
};
}  // namespace alus::s1tbx
