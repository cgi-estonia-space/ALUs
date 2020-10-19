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
#pragma once

#include <iostream>

#include "backgeocoding_io.h"
#include "backgeocoding.h"


namespace alus {
namespace backgeocoding{

/**
 * A helper class to manage the data intputs and threading to Backgeocoding class.
 */
class BackgeocodingController: public BackgeocodingIO{
private:
    Backgeocoding *backgeocoding_{nullptr};
    Rectangle slave_rect;

public:
    void ComputeImage();
    void ReadTile(Rectangle area, double *tile_i, double *tile_q) override ;
    BackgeocodingController();
    ~BackgeocodingController();
};

}//namespace
}//namespace
