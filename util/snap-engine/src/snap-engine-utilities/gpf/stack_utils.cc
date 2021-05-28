/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.engine_utilities.gpf.StackUtils.java
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

#include "stack_utils.h"

#include <string>
#include <memory>

#include "snap-core/datamodel/product.h"
#include "snap-engine-utilities/datamodel/metadata/abstract_metadata.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-engine-utilities/gpf/operator_utils.h"
#include "snap-core/util/string_utils.h"

namespace alus::snapengine{

std::string StackUtils::CreateBandTimeStamp(std::shared_ptr<Product>& product) {
    std::shared_ptr<MetadataElement> abs_root = AbstractMetadata::GetAbstractedMetadata(product);
    if (abs_root != nullptr) {
        std::string date_string = OperatorUtils::GetAcquisitionDate(abs_root);
        if (!date_string.empty()) date_string = '_' + date_string;
        return StringUtils::CreateValidName(date_string, {'_', '.'}, '_');
    }
    return "";
}

}
