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

#include <map>
#include <memory>

#include "alus_file_reader.h"
#include "dataset.h"
#include "gdal_util.h"

namespace alus {

template <typename BufferType>
class C16Dataset : public AlusFileReader<BufferType> {
public:
    explicit C16Dataset(std::string_view filename);
    C16Dataset(C16Dataset<BufferType> const&) = delete;
    C16Dataset<BufferType>& operator=(C16Dataset<BufferType> const&) = delete;

    void ReadRectangle(Rectangle rectangle, int band_nr, BufferType* data_buffer) override;
    void ReadRectangle(Rectangle rectangle, std::map<int, BufferType*>& bands) override;

    void LoadRasterBand(int band_nr) override;
    std::vector<BufferType> const& GetHostDataBuffer() const override;
    BufferType* GetDeviceDataBuffer() override;
    uint64_t GetBufferByteSize() override;
    size_t GetBufferElemCount() override;

    Dataset<Iq16>* GetDataset() { return dataset_.get(); }
    void TryToCacheImage() override;
    void SetReadingArea(Rectangle new_area) override { dataset_->SetReadingArea(new_area); };

private:
    std::unique_ptr<Dataset<Iq16>> dataset_;
};

}  // namespace alus
