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

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <atomic>

namespace {


TEST(MultiThreadingTest, MultiThreading) {
    std::atomic<uint64_t> test_atomic = 0;
    constexpr uint64_t RERUNS = 100;
    constexpr uint64_t INCREMENTS = 10000;

    /*
        a simple threadpool stress test
        The test against boost is left in for future reference,
        if we choose use another library or implement our own
     */
    for(uint64_t i = 0; i < RERUNS; i++) {

        test_atomic = 0;
        boost::asio::thread_pool thread_pool(4);
        for(uint64_t j = 0; j < INCREMENTS; j++) {
            boost::asio::post(thread_pool, [&test_atomic](){ test_atomic++; });
        }
        thread_pool.wait();
        ASSERT_EQ(INCREMENTS, test_atomic);
    }
}
}  // namespace