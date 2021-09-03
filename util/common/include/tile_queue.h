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

#include <cstdlib>
#include <mutex>
#include <vector>

namespace alus {

template <class T>
class ThreadSafeTileQueue {
    std::vector<T> tiles_ = {};
    std::mutex mutex_;
    size_t queue_index_ = 0;

public:
    ThreadSafeTileQueue() = default;
    explicit ThreadSafeTileQueue(std::vector<T>&& tiles) { InsertData(std::move(tiles)); }

    void InsertData(std::vector<T>&& tiles) {
        std::unique_lock l(mutex_);
        queue_index_ = 0;
        tiles_ = std::move(tiles);
    }

    bool PopFront(T& dest) {
        std::unique_lock l(mutex_);
        if (queue_index_ == tiles_.size()) {
            return false;
        }
        dest = tiles_.at(queue_index_);
        queue_index_++;
        return true;
    }
};
}  // namespace alus