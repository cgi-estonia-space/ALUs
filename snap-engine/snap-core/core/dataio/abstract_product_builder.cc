/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.AbstractProductBuilder.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "snap-core/core/dataio/abstract_product_builder.h"
namespace alus::snapengine {

AbstractProductBuilder::AbstractProductBuilder(bool source_product_owner)
    : AbstractProductReader(nullptr), source_product_owner_(source_product_owner) {}



}  // namespace alus::snapengine
