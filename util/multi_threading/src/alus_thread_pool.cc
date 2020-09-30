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
#include "alus_thread_pool.h"

#include <algorithm>

alus::multithreading::ThreadPool::ThreadPool(unsigned int desired_pool_size) : is_running_{false} {
    auto system_max_threads = std::thread::hardware_concurrency();
    pool_size_ = desired_pool_size < system_max_threads ? desired_pool_size : system_max_threads - 1;
    workers_.resize(pool_size_);
}
alus::multithreading::ThreadPool::ThreadPool() : is_running_{false} {
    auto system_max_threads = std::thread::hardware_concurrency();
    pool_size_ = system_max_threads > 1 ? std::thread::hardware_concurrency() / 2 : system_max_threads;
    workers_.resize(pool_size_);
}

alus::multithreading::ThreadPool::~ThreadPool() { Synchronise(); }
void alus::multithreading::ThreadPool::Enqueue(std::shared_ptr<alus::multithreading::Executable> job) {
    job_mutex_.lock();
    jobs_.Push(job);
    job_mutex_.unlock();
}
void alus::multithreading::ThreadPool::Start() {
    is_running_ = true;
    for (unsigned int i = 0; i < pool_size_; i++) {
        auto* thread_worker = new alus::multithreading::ThreadWorker(*this);
        workers_.push_back(thread_worker);
    }
}

bool alus::multithreading::ThreadPool::IsRunning() {
    if (is_running_.load()) {
        is_running_ = !jobs_.IsEmpty();
    }
    return is_running_.load();
}
void alus::multithreading::ThreadPool::Synchronise() {
    for (auto* worker : workers_) {
        if (worker) {
            worker->Join();
            delete worker;
        }
    }
    workers_.clear();
}
