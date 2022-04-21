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

#include "execute.h"

#include "gdal_management.h"

namespace alus::resample {

Execute::Execute(Parameters params) : params_{std::move(params)} { gdalmanagement::Initialize(); }

void Execute::Run(alus::cuda::CudaInit&, size_t) {}

Execute::~Execute() { gdalmanagement::Deinitialize(); }
}  // namespace alus::resample