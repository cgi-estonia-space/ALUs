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

#include "binary_output.h"

#include <cstdio>

#include "alus_log.h"
#include "checks.h"

namespace alus::palsar {
void WriteRawBinary(const void* data, size_t elem_size, size_t n_elem, const char* path) {
    LOGD << "Writing binary file @ " << path;
    FILE* fp = fopen(path, "w");
    CHECK_NULLPTR(fp);
    fwrite(data, elem_size, n_elem, fp);
    fclose(fp);
}
}  // namespace alus::palsar
