/**
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

#include "gdal_priv.h"

#include "dataset.h"
#include "geocoding.h"

namespace alus {
namespace snapengine {
namespace old {

class Product {
public:
    std::unique_ptr<alus::snapengine::geocoding::Geocoding> geocoding_;
    std::shared_ptr<GDALDataset> dataset_;
    const char* FILE_FORMAT_;
};
}  // namespace old
}  // namespace snapengine
}  // namespace alus