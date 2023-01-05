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

#include <chrono>
#include <iostream>
#include <memory>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>  // Might be falsely flagged by CLion as unnecessary.
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>

#include <boost/log/attributes/attribute.hpp>
#include <boost/log/attributes/attribute_cast.hpp>
#include <boost/log/attributes/attribute_value.hpp>
#include <boost/log/attributes/attribute_value_impl.hpp>

namespace {
std::unique_ptr<std::map<alus::common::log::Level, boost::log::trivial::severity_level>> boost_level_map{};  // NOSONAR

const auto uptime_start = std::chrono::system_clock::now();
auto log_format = alus::common::log::Format::DEFAULT;

void CreateLogLevelMap() {
    boost_level_map = std::make_unique<std::map<alus::common::log::Level, boost::log::trivial::severity_level>>();
    boost_level_map->emplace(alus::common::log::Level::VERBOSE, boost::log::trivial::severity_level::trace);
    boost_level_map->emplace(alus::common::log::Level::DEBUG, boost::log::trivial::severity_level::debug);
    boost_level_map->emplace(alus::common::log::Level::INFO, boost::log::trivial::severity_level::info);
    boost_level_map->emplace(alus::common::log::Level::WARNING, boost::log::trivial::severity_level::warning);
    boost_level_map->emplace(alus::common::log::Level::ERROR, boost::log::trivial::severity_level::error);
}

uint32_t ElapsedTime() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now() - uptime_start).count();
}

class ElapsedTimeAttributeValue : public boost::log::attribute::impl {
public:
    // The method generates a new attribute value
    boost::log::attribute_value get_value() { return boost::log::attributes::make_attribute_value(ElapsedTime()); }
};

class ElapsedTimeAttribute : public boost::log::attribute {
public:
    ElapsedTimeAttribute() : boost::log::attribute(new ElapsedTimeAttributeValue()) {}
    // Attribute casting support
    explicit ElapsedTimeAttribute(boost::log::attributes::cast_source const& source)
        : boost::log::attribute(source.as<ElapsedTimeAttributeValue>()) {}
};
}  // namespace

namespace alus::common::log {

void Initialize(Format f) {
    log_format = f;
    if (f == Format::CREODIAS) {
        boost::log::add_console_log(
            std::cout, boost::log::keywords::format =
                           (boost::log::expressions::stream
                            << "{ \"time\": \""
                            << boost::log::expressions::format_date_time<boost::posix_time::ptime>(
                                   "TimeStamp", "%Y-%m-%d %H:%M:%S.%f")
                            << "\", "
                            << "\"level\": \"" << boost::log::trivial::severity << "\", "
                            << "\"message\": \"" << boost::log::expressions::smessage << "\", "
                            << "\"elapsed\": " << boost::log::expressions::attr<uint32_t>("ElapsedTime") << ", "
                            << "\"unit\": \"s\""
                            << " }"));
        boost::log::add_common_attributes();
        boost::log::core::get()->add_global_attribute("ElapsedTime", ElapsedTimeAttribute());
    }

    CreateLogLevelMap();
    SetLevel(Level::VERBOSE);
}

void SetLevel(Level level) {
    if (log_format == Format::CREODIAS) {
        boost::log::core::get()->set_filter(
            (boost::log::trivial::severity == boost::log::trivial::severity_level::debug ||
             boost::log::trivial::severity == boost::log::trivial::severity_level::info ||
             boost::log::trivial::severity == boost::log::trivial::severity_level::error) &&
            boost::log::trivial::severity >= boost_level_map->at(level));
    } else {
        boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost_level_map->at(level));
    }
}
}  // namespace alus::common::log