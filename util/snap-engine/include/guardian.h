/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.util.Guardian.java ported
 * for native code. Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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

#include <boost/date_time/posix_time/posix_time.hpp>

#include "band.h"
#include "product.h"
#include "product_data.h"

namespace alus {
namespace snapengine {
class Guardian {
public:
    static void AssertNotNullOrEmpty(std::string_view expr_text, std::string_view text);
    template <typename T>
    static void AssertNotNull(std::string_view expr_text, T expr_value) {
        if (expr_value == nullptr) {
            throw std::invalid_argument(std::string(expr_text) + " argument is nullptr");
        }
    }
};

}  // namespace snapengine
}  // namespace alus
