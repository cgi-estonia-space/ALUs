/**
 * This file is a filtered duplicate of a SNAP's
 * com.bc.ceres.core.NullProgressMonitor.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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

#include "ceres-core/i_progress_monitor.h"

namespace alus {
namespace ceres {

/**
 * A default progress monitor implementation suitable for subclassing.
 * <p>
 * This implementation supports cancellation. The default implementations of
 * the other methods do nothing.
 * <p>
 * This class has been more or less directly taken over from the
 * <a href="http://www.eclipse.org/">Eclipse</a> Core API.
 */
class NullProgressMonitor : public IProgressMonitor {

private:
    /**
     * Indicates whether cancel has been requested.
     */
    bool canceled_ = false;

public:
    /**
     * Constructs a new progress monitor.
     */
    NullProgressMonitor() : IProgressMonitor() {}

    /**
     * This implementation does nothing.
     * Subclasses may override this method to do interesting
     * processing when a task begins.
     */
    void BeginTask([[maybe_unused]] std::string_view task_name, [[maybe_unused]] int total_work) override{
        // do nothing
    }

    /**
     * This implementation does nothing.
     * Subclasses may override this method to do interesting
     * processing when a task is done.
     */
    void Done() override {
        // do nothing
    }

    /**
     * This implementation does nothing.
     * Subclasses may override this method.
     */
    void InternalWorked([[maybe_unused]] double work) override{
        // do nothing
    }

    /**
     * This implementation returns the value of the internal
     * state variable set by <code>setCanceled</code>.
     * Subclasses which override this method should
     * override <code>setCanceled</code> as well.
     */
    bool IsCanceled() override { return canceled_; }

    /**
     * This implementation sets the value of an internal state variable.
     * Subclasses which override this method should override
     * <code>isCanceled</code> as well.
     */
    void SetCanceled(bool canceled) override { canceled_ = canceled; }

    /**
     * This implementation does nothing.
     * Subclasses may override this method to do something
     * with the name of the task.
     */
    void SetTaskName([[maybe_unused]] std::string_view taskName) override {
        // do nothing
    }

    /**
     * This implementation does nothing.
     * Subclasses may override this method to do interesting
     * processing when a subtask begins.
     */
    void SetSubTaskName([[maybe_unused]] std::string_view sub_task_name) override {
        // do nothing
    }

    /**
     * This implementation does nothing.
     * Subclasses may override this method to do interesting
     * processing when some work has been completed.
     */
    void Worked([[maybe_unused]] int work) override {
        // do nothing
    }
};
}  // namespace ceres
}  // namespace alus
