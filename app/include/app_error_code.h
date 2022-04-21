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

namespace alus::app::errorcode {

constexpr int ALG_SUCCESS{0};
constexpr int ARGUMENT_PARSE{1};
constexpr int ALGORITHM_EXCEPTION{2};
constexpr int GENERAL_EXCEPTION{3};
constexpr int UNKNOWN_EXCEPTION{4};
constexpr int GPU_DEVICE_ERROR{5};

}  // namespace alus::app::errorcode