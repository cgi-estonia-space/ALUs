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

namespace alus {

template <typename T>
T ** Allocate2DArray(int x, int y){
    int i=0, size=x*y, countX=0;
    T ** result = new T*[x];
    T * inside = new T[size];

    for(i=0; i<size; i+=y){
        result[countX] = &inside[i];
        countX++;
    }
    return result;
}

}//namespace
