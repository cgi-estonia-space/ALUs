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

#include "palsar_record.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "alus_log.h"

namespace alus::palsar {
double Record::ReadF16(uint32_t data_offset) {
    VerifyOffset(data_offset, 16);
    std::string s(Data() + data_offset - 1, 16);
    double res = 0.0;
    try {
        res = std::stod(s);
    } catch (std::exception& ex) {
        PrintBytes(data_offset, 16);
        throw;
    }
    return res;
}

double Record::ReadF8(uint32_t data_offset) {
    VerifyOffset(data_offset, 8);
    std::string s(Data() + data_offset - 1, 8);
    double res = 0.0;
    try {
        res = std::stod(s);
    } catch (std::exception& ex) {
        PrintBytes(data_offset, 8);
        throw;
    }
    return res;
}

double Record::ReadD22(uint32_t data_offset) {
    VerifyOffset(data_offset, 22);
    std::string s(Data() + data_offset - 1, 22);
    double res = 0.0;
    try {
        res = std::stod(s);
    } catch (std::exception& ex) {
        PrintBytes(data_offset, 22);
        throw;
    }
    return res;
}

uint32_t Record::ReadI4(uint32_t data_offset) {
    VerifyOffset(data_offset, 4);
    std::string s(Data() + data_offset - 1, 4);
    uint32_t res = 0.0;
    try {
        res = std::stoi(s);
    } catch (std::exception& ex) {
        PrintBytes(data_offset, 4);
        throw;
    }
    return res;
}

uint64_t Record::ReadI6(uint32_t data_offset) {
    VerifyOffset(data_offset, 6);
    std::string s(Data() + data_offset - 1, 6);
    int64_t res = 0.0;
    try {
        res = std::stoll(s);
    } catch (std::exception& ex) {
        PrintBytes(data_offset, 6);
        throw;
    }
    return res;
}

uint64_t Record::ReadI8(uint32_t data_offset) {
    VerifyOffset(data_offset, 8);
    std::string s(Data() + data_offset - 1, 8);
    int64_t res = 0.0;
    try {
        res = std::stoll(s);
    } catch (std::exception& ex) {
        PrintBytes(data_offset, 8);
        throw;
    }
    return res;
}

std::string Record::ReadAn(uint32_t data_offset, size_t n) {
    VerifyOffset(data_offset, n);
    return {Data() + data_offset - 1, n};
}

uint8_t Record::ReadB1(uint32_t data_offset) {
    VerifyOffset(data_offset, 1);
    return *(Data() + data_offset - 1);
}

uint32_t Record::ReadB4(uint32_t data_offset) {
    VerifyOffset(data_offset, 4);
    uint32_t r;
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
    memcpy(&r, Data() + data_offset - 1, 4);
    return __builtin_bswap32(r);  // in file stored as big endian
}

void Record::VerifyOffset(size_t offset, size_t field_size) {
    if (offset - 1 + field_size >= record_size_) {
        std::string err_msg = " offset " + std::to_string(offset) + " with field size " + std::to_string(field_size)

                              + " out of bounds of" + record_name_;
        throw std::logic_error(err_msg);
    }
}

void Record::PrintBytes(size_t offset, size_t field_size) {
    const char* ptr = Data() + offset - 1;
    LOGE << "Conversion failed in " << record_name_;
    std::string line;
    line.resize(32);
    for (size_t i = 0; i < field_size; i++) {
        snprintf(line.data(), line.size(), "offset = %zu - (%02X) - %c", offset + i - 1, ptr[i], ptr[i]);
        LOGE << line;
    }
}

}  // namespace alus::palsar