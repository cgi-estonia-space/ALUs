/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.util.math.FX.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General  License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General  License for
 * more details.
 *
 * You should have received a copy of the GNU General  License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include <functional>
#include <memory>
#include <string>

namespace alus {
namespace snapengine {

/**
 * Represents a function <i>f(x)</i>.
 *
 * original java version author Norman Fomferra
 */
class IFX {
public:
    virtual ~IFX();

    /**
     * The function <i>f(x) = x <sup>4</sup></i>
     */
    static const std::reference_wrapper<IFX> XXXX;
    /**
     * The function <i>f(x) = x <sup>3</sup></i>
     */
    static const std::reference_wrapper<IFX> XXX;
    /**
     * The function <i>f(x) = x <sup>2</sup></i>
     */
    static const std::reference_wrapper<IFX> XX;
    /**
     * The function <i>f(x) = x</i>
     */
    static const std::reference_wrapper<IFX> X;
    /**
     * The function <i>f(x) = 1</i>
     */
    static const std::reference_wrapper<IFX> ONE;

    /**
     * The function <i>y = f(x)</i>
     *
     * @param x the x parameter
     *
     * @return y
     */
    virtual double F(double x) = 0;

    /**
     * Returns the function as C code expression, e.g. <code>"pow(x, 3) * y"</code>
     * @return the C code
     */
    virtual std::string GetCCodeExpr() = 0;
};

}  // namespace snapengine
}  // namespace alus
