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

#include <array>
#include <string_view>

namespace alus::resample {
// They are duplicate of NPP library's typedefs. It is redundant, but for the future there can be other methods rather
// than NPP library's methods only.
enum class Method {
    NEAREST_NEIGHBOUR,
    LINEAR,
    CUBIC,
    CUBIC2P_BSPLINE,
    CUBIC2P_CATMULLROM,
    CUBIC2P_C05C03,
    SUPER,
    LANCZOS,
    LANCZOS3,
    SMOOTH_EDGE,
    ROW_NEIGHBOUR
};

constexpr std::array<std::string_view, 11> METHOD_STRINGS{
    "nearest-neighbour", "linear", "cubic",   "cubic2p-bspline", "cubic2p-catmullrom",
    "cubic2p-c05c03",    "super",  "lanczos", "lanczos3", "smooth-edge", "row-neighbour"
};

}  // namespace alus::resample