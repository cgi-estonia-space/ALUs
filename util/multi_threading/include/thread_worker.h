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
#include <thread>

#include "blocking_queue.h"
#include "executable.h"

namespace alus {
namespace multithreading {

class ThreadPool;
class ThreadWorker {
   public:
    explicit ThreadWorker(alus::multithreading::ThreadPool& thread_pool);
    void Join();

   private:
    alus::multithreading::ThreadPool& thread_pool_;
    std::thread thread_;
    std::atomic<bool> is_running_;

    void Work();
};
//alus::multithreading::BlockingQueue<alus::multithreading::Executable*>& ThreadPool::GetQueue() { return jobs_; }
}  // namespace multithreading
}  // namespace alus
