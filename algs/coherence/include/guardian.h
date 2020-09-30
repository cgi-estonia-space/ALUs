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

#include "product_data.h"

namespace alus {
namespace snapengine {
class Guardian {
   public:
    static void AssertNotNullOrEmpty(std::string_view expr_text, std::string_view text);
    static void AssertNotNull(std::string_view expr_text, boost::posix_time::time_input_facet* expr_value);
    static void AssertNotNull(std::string_view expr_text, std::shared_ptr<ProductData> &expr_value);
};

}  // namespace snapengine
}  // namespace alus
