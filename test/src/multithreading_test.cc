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
#include "gmock/gmock.h"

#include <atomic>
#include <thread>

#include "tile_queue.h"

namespace {

TEST(MultiThreadingTest, ThreadSafeQueueWorksCorrectly) {
    std::atomic<uint64_t> test_atomic = 0;
    constexpr uint64_t RERUNS = 100;
    constexpr uint64_t INCREMENTS = 10000;

    for (uint64_t i = 0; i < RERUNS; i++) {
        test_atomic = 0;
        const int value = 2;
        alus::ThreadSafeTileQueue<int> tile_queue;
        std::vector<int> data(INCREMENTS, value);
        tile_queue.InsertData(std::move(data));
        std::vector<std::thread> pool;
        for (uint64_t j = 0; j < 16; j++) {  // NOLINT
            pool.emplace_back([&]() {
                while (true) {
                    int value = 0;
                    if (!tile_queue.PopFront(value)) {
                        break;
                    }
                    test_atomic += value;
                }
            });
        }
        for (auto& t : pool) t.join();
        ASSERT_EQ(INCREMENTS * value, test_atomic);
    }
}
}  // namespace