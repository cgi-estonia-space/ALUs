/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.ProductIOPlugInManager.java
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
#pragma once

#include <memory>
#include <vector>

#include "snap-core/dataio/i_product_reader_plug_in.h"

namespace alus::snapengine {

/**
 * THIS IS MODIFIED SHORTCUT VERSION TO INJECT PLUGINS, BUT TO KEEP SNAP-ENGINE AND S1TBX DECOUPLED
 */
class ProductIOPlugInManager {
private:
    std::vector<std::shared_ptr<IProductReaderPlugIn>> reader_plug_ins_;

    ProductIOPlugInManager() {}

public:
    ProductIOPlugInManager(ProductIOPlugInManager const&) = delete;
    void operator=(ProductIOPlugInManager const&) = delete;

    static ProductIOPlugInManager& GetInstance() {
        static ProductIOPlugInManager instance;
        return instance;
    }

    /**
     * Gets all registered reader plug-ins. In the case that no reader plug-ins are registered, an empty iterator is
     * returned.
     *
     * @return an iterator containing all registered reader plug-ins
     */
    std::vector<std::shared_ptr<IProductReaderPlugIn>> GetAllReaderPlugIns() { return reader_plug_ins_; }

    /**
     * Registers the specified reader plug-in by adding it to this manager. If the given reader plug-in is
     * <code>null</code>, nothing happens.
     *
     * @param readerPlugIn the reader plug-in to be added to this manager
     */
    void AddReaderPlugIn(std::shared_ptr<IProductReaderPlugIn> reader_plug_in) {
        if (reader_plug_in) {
            reader_plug_ins_.emplace_back(reader_plug_in);
        }
    }
};
}  // namespace alus::snapengine
