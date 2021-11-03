/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.util.math.Functions.java
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

#include <string>

#include "snap-core/core/util/math/i_f_x.h"
#include "snap-core/core/util/math/i_f_x_y.h"

namespace alus {
namespace snapengine {

/**
 * Represents a function <i>F(x,y)</i>.
 *
 * original java version author Norman Fomferra
 */
namespace functions {
//////////////////////////////////////////////////////
// Frequently used F(x) functions
//////////////////////////////////////////////////////

class FX_X4 final : public virtual IFX {
public:
    double F(double x) override { return (x * x * x * x); }
    std::string GetCCodeExpr() override { return "pow(x, 4)"; }
};

class FX_X3 final : public virtual IFX {
public:
    double F(double x) override { return (x * x * x); }
    std::string GetCCodeExpr() override { return "pow(x, 3)"; }
};

class FX_X2 final : public virtual IFX {
public:
    double F(double x) override { return (x * x); }
    std::string GetCCodeExpr() override { return "pow(x, 2)"; }
};

class FX_X final : public virtual IFX {
public:
    double F(double x) override { return x; }
    std::string GetCCodeExpr() override { return "x"; }
};

class FX_1 final : public virtual IFX {
public:
    double F([[maybe_unused]] double x) override { return 1.0; }
    std::string GetCCodeExpr() override { return "1"; }
};

//////////////////////////////////////////////////////
// Frequently used F(x,y) functions
//////////////////////////////////////////////////////

class FXY_X4Y4 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x * x) * (y * y * y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 4) * pow(y, 4)"; }
};

class FXY_X4Y3 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x * x) * (y * y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 4) * pow(y, 3)"; }
};

class FXY_X3Y4 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x) * (y * y * y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 3) * pow(y, 4)"; }
};

class FXY_X4Y2 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x * x) * (y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 4) * pow(y, 2)"; }
};

class FXY_X2Y4 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x) * (y * y * y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 2) * pow(y, 4)"; }
};

class FXY_X4Y final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x * x) * (y); }
    std::string GetCCodeExpr() override { return "pow(x, 4) * y"; }
};

class FXY_XY4 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x) * (y * y * y * y); }
    std::string GetCCodeExpr() override { return "x * pow(y, 4)"; }
};

class FXY_X4 final : public virtual IFXY {
public:
    double F(double x, [[maybe_unused]] double y) override { return (x * x * x * x); }
    std::string GetCCodeExpr() override { return "pow(x, 4)"; }
};

class FXY_Y4 final : public virtual IFXY {
public:
    double F([[maybe_unused]] double x, double y) override { return (y * y * y * y); }
    std::string GetCCodeExpr() override { return "pow(y, 4)"; }
};

class FXY_X3Y3 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x) * (y * y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 3) * pow(y, 3)"; }
};

class FXY_X3Y2 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x) * (y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 3) * pow(y, 2)"; }
};

class FXY_X2Y3 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x) * (y * y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 2) * pow(y, 3)"; }
};

class FXY_X3Y final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x * x) * (y); }
    std::string GetCCodeExpr() override { return "pow(x, 3) * y"; }
};

class FXY_XY3 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x) * (y * y * y); }
    std::string GetCCodeExpr() override { return "x * pow(y, 3)"; }
};

class FXY_X3 final : public virtual IFXY {
public:
    double F(double x, [[maybe_unused]] double y) override { return (x * x * x); }
    std::string GetCCodeExpr() override { return "pow(x, 3)"; }
};

class FXY_Y3 final : public virtual IFXY {
public:
    double F([[maybe_unused]] double x, double y) override { return (y * y * y); }
    std::string GetCCodeExpr() override { return "pow(y, 3)"; }
};

class FXY_X2Y2 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x) * (y * y); }
    std::string GetCCodeExpr() override { return "pow(x, 2) * pow(y, 2)"; }
};

class FXY_X2Y final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x * x) * (y); }
    std::string GetCCodeExpr() override { return "pow(x, 2) * y"; }
};

class FXY_XY2 final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x) * (y * y); }
    std::string GetCCodeExpr() override { return "x * pow(y, 2)"; }
};

class FXY_X2 final : public virtual IFXY {
public:
    double F(double x, [[maybe_unused]] double y) override { return (x) * (x); }
    std::string GetCCodeExpr() override { return "pow(x, 2)"; }
};

class FXY_XY final : public virtual IFXY {
public:
    double F(double x, double y) override { return (x) * (y); }
    std::string GetCCodeExpr() override { return "x * y"; }
};

class FXY_Y2 final : public virtual IFXY {
public:
    double F([[maybe_unused]] double x, double y) override { return (y * y); }
    std::string GetCCodeExpr() override { return "pow(y, 2)"; }
};

class FXY_X final : public virtual IFXY {
public:
    double F(double x, [[maybe_unused]] double y) override { return (x); }
    std::string GetCCodeExpr() override { return "x"; }
};

class FXY_Y final : public virtual IFXY {
public:
    double F([[maybe_unused]] double x, double y) override { return (y); }
    std::string GetCCodeExpr() override { return "y"; }
};

class FXY_1 final : public virtual IFXY {
public:
    double F([[maybe_unused]] double x, [[maybe_unused]] double y) override { return 1.0; }
    std::string GetCCodeExpr() override { return "1"; }
};

}  // namespace functions
}  // namespace snapengine
}  // namespace alus
