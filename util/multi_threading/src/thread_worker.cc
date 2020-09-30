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
#include "thread_worker.h"

#include "alus_thread_pool.h"

void alus::multithreading::ThreadWorker::Join() {
    thread_.join();
}
alus::multithreading::ThreadWorker::ThreadWorker(alus::multithreading::ThreadPool& thread_pool)
    : thread_pool_(thread_pool), is_running_(true) {
    thread_ = std::thread(&alus::multithreading::ThreadWorker::Work, this);
}
void alus::multithreading::ThreadWorker::Work() {
    while (thread_pool_.IsRunning()) {
        std::shared_ptr<alus::multithreading::Executable> executable{};

        if (!thread_pool_.jobs_.IsEmpty()) {
            executable = thread_pool_.jobs_.Front();
            thread_pool_.jobs_.Pop();
        }

        if (executable) {
            executable->Execute();
        }
    }
}
