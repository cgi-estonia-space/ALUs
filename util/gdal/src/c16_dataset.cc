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

#include "c16_dataset.h"

#include <cstdint>
#include <map>
#include <vector>

#include "dataset.h"

namespace alus {

template <typename BufferType>
C16Dataset<BufferType>::C16Dataset(std::string_view filename) {
    dataset_ = std::make_unique<Dataset<Iq16>>(filename);
}

template <typename BufferType>
void C16Dataset<BufferType>::ReadRectangle(Rectangle /*rectangle*/, int /*band_nr*/, BufferType* /*data_buffer*/) {
    throw std::runtime_error("Sorry mate, c16 dataset does not work with single bands. Not yet atleast");
}

template <typename BufferType>
void C16Dataset<BufferType>::ReadRectangle(Rectangle rectangle, std::map<int, BufferType*>& bands) {
    if (bands.size() != 2) {
        throw std::runtime_error("C16 reader only reads C16 products with 2 bands in them.");
    }

    std::vector<Iq16> pairs(rectangle.height * rectangle.width);
    BufferType* i_band = bands.at(1);
    BufferType* q_band = bands.at(2);

    dataset_->ReadRectangle(rectangle, 1, pairs.data());

    for (auto& pair : pairs) {
        *i_band = pair.i;
        *q_band = pair.q;

        i_band++;
        q_band++;
    }
}

template <typename BufferType>
void C16Dataset<BufferType>::TryToCacheImage() {
    dataset_->TryToCacheImage();
}

template <typename BufferType>
void C16Dataset<BufferType>::LoadRasterBand(int /*band_nr*/) {
    throw std::runtime_error("Loading raster bands is disabled on C16 dataset.");
}

template <typename BufferType>
std::vector<BufferType> const& C16Dataset<BufferType>::GetHostDataBuffer() const {
    throw std::runtime_error("C16 Dataset does not have data buffers");
}

template <typename BufferType>
BufferType* C16Dataset<BufferType>::GetDeviceDataBuffer() {
    throw std::runtime_error("C16 Dataset does not have data buffers");
}

template <typename BufferType>
long unsigned int C16Dataset<BufferType>::GetBufferByteSize() {  // NOLINT
    throw std::runtime_error("C16 Dataset does not have data buffers");
}

template <typename BufferType>
size_t C16Dataset<BufferType>::GetBufferElemCount() {
    throw std::runtime_error("C16 Dataset does not have data buffers");
}

template class C16Dataset<double>;
template class C16Dataset<float>;
template class C16Dataset<int16_t>;
template class C16Dataset<int>;

}  // namespace alus
