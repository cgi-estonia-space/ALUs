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

#include <cstdint>
#include <cstdlib>
#include <string>

namespace alus::palsar {
/**
 * Record is the smallest subdivision of PALSAR file,
 * this class provides a small view object to records and can be used to parse elements from the records
 */
class Record {
public:
    Record(const char* data_ptr_, size_t record_offset, size_t record_size, const char* record_name)
        : file_data_(data_ptr_), record_name_(record_name), record_offset_(record_offset), record_size_(record_size) {}

    // Member functions to read a specific datatype from a record,
    // all ReadXX member function offsets follow the PALSAR file documentation which uses 1-based index for offsets,
    // and only substract 1 internally to follow C's zero based indexing
    double ReadF16(uint32_t data_offset);
    double ReadF8(uint32_t data_offset);
    double ReadD22(uint32_t data_offset);
    uint32_t ReadI4(uint32_t data_offset);
    uint64_t ReadI6(uint32_t data_offset);
    uint64_t ReadI8(uint32_t data_offset);
    std::string ReadAn(uint32_t data_offset, size_t n);
    uint8_t ReadB1(uint32_t data_offset);
    uint32_t ReadB4(uint32_t data_offset);

private:
    const char* Data() { return file_data_ + record_offset_; }

    void VerifyOffset(size_t offset, size_t field_size);

    void PrintBytes(size_t offset, size_t field_size);

    const char* file_data_;
    const char* record_name_;
    size_t record_offset_;
    size_t record_size_;
};
}  // namespace alus::palsar
