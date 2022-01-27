/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.test.MetadataValidator.java
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
    void VerifyStr(std::string_view tag, const std::vector<std::string>& allowed_str);
    void VerifyDouble(std::string_view tag);
    void VerifyInt(std::string_view tag);
    void VerifyUTC(std::string_view tag);

public:
    explicit MetadataValidator(const std::shared_ptr<snapengine::Product>& product);
    MetadataValidator(const std::shared_ptr<snapengine::Product>& product,
                      const std::shared_ptr<ValidationOptions>& options);
    void Validate();
    void ValidateSAR();

    static void ValidateOptical();
};
}  // namespace alus::s1tbx
