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

#include <cstddef>
#include <functional>
#include <memory>
#include <string_view>

namespace alus::gdalmanagement {

/**
 * Initializes currently needed drivers and ALUS error handler callback for GDAL.
 * Requires log system to be setup before.
 */
void Initialize();

/**
 * Sets GDAL default error handler back in place.
 */
void Deinitialize();

/**
 * It is strongly advised to use other GDAL mechanisms to set cache - https://gdal.org/user/configoptions.html
 * This call duplicates GDAL API 'GDALSetCacheMax64()'.
 */
void SetCacheMax(size_t bytes);

using ErrorCallback = std::function<void(std::string_view)>;
using ErrorCallbackGuard = std::unique_ptr<const ErrorCallback, void(*)(const ErrorCallback*)>;

/**
 * One can install custom error handle/callback for GDAL system messages.
 * This is not thread safe/thread bounded, therefore concurrent/multiple calls
 * might lead to unexpected results/behavior.
 *
 * @return A guard object that de registers the callback.
 */
ErrorCallbackGuard SetErrorHandle(const ErrorCallback& handler);

}  // namespace alus::gdalmanagement