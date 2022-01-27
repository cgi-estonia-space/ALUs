/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.commons.io.ImageIOFile.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include "s1tbx-commons/io/band_info.h"

#include <utility>

#include "snap-engine-utilities/engine-utilities/datamodel/unit.h"

namespace alus::s1tbx {

BandInfo::BandInfo(const std::shared_ptr<snapengine::Band>& band, std::shared_ptr<ImageIOFile> img_file, const int id,
                   const int offset)
    : image_i_d_(id),
      band_sample_offset_(offset),
      img_(std::move(img_file)),
      is_imaginary_(band->GetUnit().has_value() && (band->GetUnit().value() == snapengine::Unit::IMAGINARY)) {
    if (is_imaginary_) {
        imaginary_band_ = band;
    } else {
        real_band_ = band;
    }
}
}  // namespace alus::s1tbx
