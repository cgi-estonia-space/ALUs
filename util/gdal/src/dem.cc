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
#include "dem.h"

namespace alus {

Dem::Dem(alus::Dataset<double> ds) : m_ds_{std::move(ds)} {
    // m_ds.LoadRasterBand(1);
}

std::vector<double> Dem::GetLocalDemFor(Dataset<double>& image, unsigned int x_0, unsigned int y_0, unsigned int width,
                                        unsigned int height) {
    auto const& data_buffer = m_ds_.GetHostDataBuffer();
    auto const band_x_size = m_ds_.GetRasterSizeX();
    std::vector<double> altitudes;
    altitudes.reserve(data_buffer.size());
    for (unsigned int i_x = 0; i_x < width; i_x++) {
        for (unsigned int i_y = 0; i_y < height; i_y++) {
            auto const coordinates =
                image.GetPixelCoordinatesFromIndex(static_cast<int>(x_0 + i_x), static_cast<int>((y_0 + i_y)));
            auto const dem_indexes =
                m_ds_.GetPixelIndexFromCoordinates(std::get<0>(coordinates), std::get<1>(coordinates));
            double value = data_buffer.at(band_x_size * std::get<1>(dem_indexes) + std::get<0>(dem_indexes));

            if (value < 0) value = 0;

            altitudes.push_back(value);
        }
    }

    return altitudes;
}

alus::Dataset<double>* Dem::GetDataset() { return &this->m_ds_; }

}  // namespace alus
