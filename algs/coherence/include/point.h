#pragma once

namespace alus {
class Point {
   private:
    double x_, y_, z_;

   public:
    Point();
    Point(double x, double y, double z);
    [[nodiscard]] double GetX() const { return x_; }
    [[nodiscard]] double GetY() const { return y_; }
    [[nodiscard]] double GetZ() const { return z_; }
    Point Min(Point p);
    double In(Point &p);
    double Norm();
    void SetX(double x);
    void SetY(double y);
    void SetZ(double z);
};
}  // namespace alus
