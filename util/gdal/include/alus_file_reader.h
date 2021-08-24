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

#include <vector>
#include <map>

#include "shapes.h"

namespace alus {

/**
 * First of all we load onto a buffer, which can be cpu or gpu buffer. After that we ask data.
 */
template <typename BufferType>
class AlusFileReader {
   public:
    virtual void ReadRectangle(Rectangle rectangle, int band_nr, BufferType *data_buffer) = 0;
    virtual void ReadRectangle(Rectangle rectangle, std::map<int, BufferType*>& bands) = 0;
    virtual void LoadRasterBand(int band_nr) = 0;
    virtual std::vector<BufferType> const& GetHostDataBuffer() const = 0;
    virtual BufferType *GetDeviceDataBuffer() = 0;
    virtual long unsigned int GetBufferByteSize() = 0;
    virtual size_t GetBufferElemCount() = 0;
    virtual void TryToCacheImage() = 0;
};

}