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

#include <optional>

#include "device_padded_image.h"
#include "sar_metadata.h"

namespace alus::palsar {

void WriteImage(const DevicePaddedImage& img, const char* path, bool complex, std::optional<SARMetadata> metadata = {});

void WriteComplexImg(const DevicePaddedImage& img, const char* path);

void WriteIntensityImg(const DevicePaddedImage& img, const char* path);

void WriteComplexPaddedImg(const DevicePaddedImage& img, const char* path);

void WriteIntensityPaddedImg(const DevicePaddedImage& img, const char* path);

void ExportMetadata(const SARMetadata& metadata, const char* path);
}  // namespace alus::palsar
