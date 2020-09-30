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

#include <condition_variable>
#include <mutex>
#include <queue>

namespace alus {
namespace multithreading {
template <class T>
class BlockingQueue {
   public:
    [[nodiscard]] bool GetSize() { return data_.size(); }
    [[nodiscard]] bool IsEmpty() { return data_.empty(); }

    void Push(T const& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        data_.push(value);
        notifier_.notify_one();
    }

    void Pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!data_.empty()) {
            data_.pop();
        }
        notifier_.notify_one();
    }

    [[nodiscard]] T& Front() {
        std::unique_lock<std::mutex> lock(mutex_);
        notifier_.wait(lock, [&] { return !data_.empty(); });
        T& data = data_.front();
        return data;
    }

    [[nodiscard]] T& Back() {
        std::unique_lock<std::mutex> lock(mutex_);
        notifier_.wait(lock, [&] { return !data_.empty(); });
        T& data = data_.back();
        return data;
    }

   private:
    std::queue<T> data_;
    std::mutex mutex_;
    std::condition_variable notifier_;
};
}  // namespace multithreading
}  // namespace alus
