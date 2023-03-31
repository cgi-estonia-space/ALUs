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

#include "metadata_record.h"

namespace alus::common::metadata {

void Container::Add(std::string_view key, std::string value) {
    metadata_.insert(std::make_pair(key, std::move(value)));
}

void Container::AddOrAppend(std::string_view key, std::string value) {
    auto entry = metadata_.find(std::string(key));
    if (entry == metadata_.end()) {
        Add(key, std::move(value));
        return;
    }

    entry->second += " " + value;
}

void Container::AddWhenMissing(std::string_view key, std::string value) {
    if (metadata_.count(std::string(key))) {
        return;
    }

    Add(key, std::move(value));
}

}  // namespace alus::common::metadata