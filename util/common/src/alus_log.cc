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

#include "alus_log.h"

#include <memory>

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

namespace {
std::unique_ptr<std::map<alus::common::log::Level, boost::log::trivial::severity_level>> boost_level_map{}; // NOSONAR

void CreateLogLevelMap() {
    boost_level_map = std::make_unique<std::map<alus::common::log::Level, boost::log::trivial::severity_level>>();
    boost_level_map->emplace(alus::common::log::Level::VERBOSE, boost::log::trivial::severity_level::trace);
    boost_level_map->emplace(alus::common::log::Level::DEBUG, boost::log::trivial::severity_level::debug);
    boost_level_map->emplace(alus::common::log::Level::INFO, boost::log::trivial::severity_level::info);
    boost_level_map->emplace(alus::common::log::Level::WARNING, boost::log::trivial::severity_level::warning);
    boost_level_map->emplace(alus::common::log::Level::ERROR, boost::log::trivial::severity_level::error);
}
}  // namespace

namespace alus::common::log {

void Initialize() {
    CreateLogLevelMap();
    SetLevel(Level::VERBOSE);
}

void SetLevel(Level level) {
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost_level_map->at(level));
}
}  // namespace alus::common::log