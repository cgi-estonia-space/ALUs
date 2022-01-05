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

#include <boost/log/trivial.hpp>

namespace alus::common::log {

enum class Level { VERBOSE, DEBUG, INFO, WARNING, ERROR, SILENT };

void Initialize();
void SetLevel(Level level);

}  // namespace alus::common::log

#define LOGV BOOST_LOG_TRIVIAL(trace)
#define LOGD BOOST_LOG_TRIVIAL(debug)
#define LOGI BOOST_LOG_TRIVIAL(info)
#define LOGW BOOST_LOG_TRIVIAL(warning)
#define LOGE BOOST_LOG_TRIVIAL(error)