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

#include "device_padded_image.h"
#include "palsar_metadata.h"

namespace alus::palsar {
struct IQ8 {
    uint8_t i;
    uint8_t q;
};

/**
 * Both functions take as an input pointer to Signal Data Record of the IMG file
 * Layout of Signal Data Records in IMG files(720 bytes after SAR Data File Descriptor Record)
 *   1        2
 * __________________
 * |____|___________|
 * |____|___________|
 * |____|___________|
 * |____|___________|
 * |____|___________|
 * (... (repeated until azimuth size))
 *
 * Columns are:
 * 1) Signal Data Record header - fixed size of 412 bytes
 * 2) SAR RAW SIGNAL DATA - total byte size is SAR DATA record length, number of IQ8 samples divided by 2
 *
 */

// Remove signal data record header and adjust range pixel position if slant ranges to the first pixels differ between
// range rows
void FormatRawIQ8(const IQ8* d_signal_in, IQ8* d_signal_out, ImgFormat img, const std::vector<uint32_t>& offsets);

// Calculate the mean values of I and Q band over the whole dataset
void CalculateDCBias(const IQ8* d_signal_data, ImgFormat img, size_t multiprocessor_count, SARResults& result);

// Convert unsigned 8(real value max 5) bit complex data to floating point complex data by applying calculated DC
// offsets
void ApplyDCBias(const IQ8* d_signal_data, const SARResults& results, ImgFormat img, DevicePaddedImage& output);
}  // namespace alus::palsar
