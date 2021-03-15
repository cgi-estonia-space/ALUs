/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.dataio.ProductIO.java
 * ported for native code.
 * Copied from a snap-engine (https://github.com/senbox-org/snap-engine) repository originally stated:
 *
 * Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)
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
#include <any>

#include <boost/filesystem.hpp>

namespace alus::snapengine {

//pre-declare
class IProductReader;
class Product;
class ProductSubsetDef;

/**
 * The <code>ProductIO</code> class provides several utility methods concerning data I/O for remote sensing data
 * products.
 * <p> For example, a product can be read in using a single method call:
 * <pre>
 *      Product product =  ProductIO.readProduct("test.prd");
 * </pre>
 * and written out in a similar way:
 * <pre>
 *      ProductIO.writeProduct(product, "HDF5", "test.h5", null);
 * </pre>
 *
 * original java version authors: Norman Fomferra, Sabine Embacher
 */
class ProductIO {
private:
    static std::shared_ptr<Product> ReadProductImpl(const boost::filesystem::path& file, const std::shared_ptr<ProductSubsetDef>& subset_def);
public:
    /**
     * Tries to find a product reader instance suitable for the given input.
     * The method returns {@code null}, if no
     * registered product reader can handle the given {@code input} value.
     * <p>
     * The {@code input} may be of any type, but most likely it will be a file path given by a {@code String} or
     * {@code File} value. Some readers may also directly support an {@link javax.imageio.stream.ImageInputStream} object.
     *
     * @param input the input object.
     *
     * @return a product reader for the given {@code input} or {@code null} if no registered reader can handle
     *         the it.
     *
     * @see ProductReaderPlugIn#getDecodeQualification(Object)
     * @see ProductReader#readProductNodes(Object, ProductSubsetDef)
     */
    static std::shared_ptr<IProductReader> GetProductReaderForInput(const std::any& input);

    /**
     * Reads the data product specified by the given file.
     * <p>The product returned will be associated with the reader appropriate for the given
     * file format (see also {@link Product#getProductReader() Product.productReader}).
     * <p>The method does not automatically read band data, thus
     * {@link Band#getRasterData() Band.rasterData} will always be null
     * for all bands in the product returned by this method.
     *
     * @param file the data product file
     *
     * @return a data model as an in-memory representation of the given product file or <code>null</code> if no
     *         appropriate reader was found for the given product file
     *
     * @throws IOException if an I/O error occurs
     * @see #readProduct(String)
     */
    static std::shared_ptr<Product> ReadProduct(const boost::filesystem::path& file);

};
}  // namespace alus::snapengine
