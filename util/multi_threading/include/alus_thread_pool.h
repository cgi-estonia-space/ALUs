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

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <memory>

#include "blocking_queue.h"
#include "executable.h"
#include "thread_worker.h"

namespace alus {
namespace multithreading {
class ThreadPool {
   public:
    explicit ThreadPool(unsigned int desired_pool_size);
    ThreadPool();

    void Enqueue(std::shared_ptr<Executable> job);
    void Start();
    bool IsRunning();
    void Synchronise();

    virtual ~ThreadPool();

private:
    unsigned int pool_size_;
    std::mutex job_mutex_;
    std::atomic<bool> is_running_;
    std::vector<alus::multithreading::ThreadWorker*> workers_;
    alus::multithreading::BlockingQueue<std::shared_ptr<alus::multithreading::Executable>> jobs_;
    friend class alus::multithreading::ThreadWorker;
};
}  // namespace multithreading
}  // namespace alus