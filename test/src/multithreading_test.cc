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

#include <memory>
#include <numeric>

#include "alus_thread_pool.h"
#include "executable.h"

namespace {

class TestExecutable : public alus::multithreading::Executable {
   public:
    TestExecutable(std::vector<int> &test_vector, int index) : test_vector_(test_vector), index_(index) {}

   private:
    void Execute() override {
        sleep(4); // Artificially postpone write operation in order to test Synchronise method
        test_vector_[index_] = index_ + 1;
    }

   private:
    std::vector<int> &test_vector_;
    int index_;
};

TEST(MultiThreadingTest, MultiThreading) {
    int const THREAD_COUNT{4};
    int const EXPECTED_SUM{10};
    std::vector<int> test_vector(THREAD_COUNT, 0);

    alus::multithreading::ThreadPool thread_pool(THREAD_COUNT);
    for (int I = 0; I < THREAD_COUNT; ++I) {
        auto executable = std::make_shared<TestExecutable>(test_vector, I);
        thread_pool.Enqueue(executable);
    }
    thread_pool.Start();
    thread_pool.Synchronise();
    int actual_sum = std::accumulate(test_vector.begin(), test_vector.end(), 0);

    ASSERT_EQ(EXPECTED_SUM, actual_sum);
}
}  // namespace