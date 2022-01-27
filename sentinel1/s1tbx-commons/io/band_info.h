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
#pragma once

#include <memory>

#include "image_i_o_file.h"
#include "snap-core/core/datamodel/band.h"

namespace alus::s1tbx {

class BandInfo {
public:
    const int image_i_d_;
    const int band_sample_offset_;
    const std::shared_ptr<ImageIOFile> img_;
    const bool is_imaginary_;
    std::shared_ptr<snapengine::Band> real_band_;
    std::shared_ptr<snapengine::Band> imaginary_band_;

    BandInfo(const std::shared_ptr<snapengine::Band>& band, std::shared_ptr<ImageIOFile> img_file, int id, int offset);

    void SetRealBand(const std::shared_ptr<snapengine::Band>& band) { real_band_ = band; }

    void SetImaginaryBand(const std::shared_ptr<snapengine::Band>& band) { imaginary_band_ = band; }
};

}  // namespace alus::s1tbx
