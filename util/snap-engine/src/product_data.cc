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

#include "product_data.h"

namespace alus::snapengine {

Utc::Utc(int days, int seconds, int microseconds) : days_{days}, seconds_{seconds}, microseconds_{microseconds} {}

double Utc::getMjd() const {
    return getDaysFraction() + SECONDS_TO_DAYS * (getSecondsFraction() + MICROS_TO_SECONDS * getMicroSecondsFraction());
}

}  // namespace alus::snapengine
