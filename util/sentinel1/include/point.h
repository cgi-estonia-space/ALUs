#pragma once

namespace alus::s1tbx{
class Point {
   private:
    double x_, y_, z_;

   public:
    Point();
    Point(double x, double y, double z);
    [[nodiscard]] double GetX() const { return x_; }
    [[nodiscard]] double GetY() const { return y_; }
    [[nodiscard]] double GetZ() const { return z_; }
    [[nodiscard]] Point Min(const Point &p) const;
    [[nodiscard]] double In(const Point &p) const;
    [[nodiscard]] double Norm() const;
    void SetX(double x);
    void SetY(double y);
    void SetZ(double z);
};
}  // namespace alus
