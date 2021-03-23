/**
 * This file is a filtered duplicate of a SNAP's
 * com.bc.ceres.core.ProgressMonitor.java
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

#include <memory>
#include <string_view>

namespace alus {
namespace ceres {

/**
 * The <code>ProgressMonitor</code> interface is implemented
 * by objects that monitor the progress of an activity; the methods
 * in this interface are invoked by code that performs the activity.
 * <p>
 * All activity is broken down into a linear sequence of tasks against
 * which progress is reported. When a task begins, a <code>beginTask(String, int)
 * </code> notification is reported, followed by any number and mixture of
 * progress reports (<code>worked()</code>) and subtask notifications
 * (<code>subTask(String)</code>).  When the task is eventually completed, a
 * <code>done()</code> notification is reported.  After the <code>done()</code>
 * notification, the progress monitor cannot be reused;  i.e., <code>
 * beginTask(String, int)</code> cannot be called again after the call to
 * <code>done()</code>.
 * <p>
 * A request to cancel an operation can be signaled using the
 * <code>setCanceled</code> method.  Operations taking a progress
 * monitor are expected to poll the monitor (using <code>isCanceled</code>)
 * periodically and abort at their earliest convenience.  Operation can however
 * choose to ignore cancelation requests.
 * <p>
 * Since notification is synchronous with the activity itself, the listener should
 * provide a fast and robust implementation. If the handling of notifications would
 * involve blocking operations, or operations which might throw uncaught exceptions,
 * the notifications should be queued, and the actual processing deferred (or perhaps
 * delegated to a separate thread).
 * <p>
 * Clients may implement this interface.
 * <p>
 * This interface has been more or less directly taken over from the <a href="http://www.eclipse.org/">Eclipse</a> Core
 * API.
 */
class IProgressMonitor {
public:
    /**
     * Constant indicating an unknown amount of work.
     */
    static constexpr int UNKNOWN = -1;
    /**
     * Notifies that the main task is beginning.  This must only be called once
     * on a given progress monitor instance.
     *
     * @param taskName  the name (or description) of the main task
     * @param totalWork the total number of work units into which
     *                  the main task is been subdivided. If the value is <code>UNKNOWN</code>
     *                  the implementation is free to indicate progress in a way which
     *                  doesn't require the total number of work units in advance.
     */
    virtual void BeginTask(std::string_view task_name, int total_work) = 0;

    /**
     * Notifies that the work is done; that is, either the main task is completed
     * or the user canceled it. This method may be called more than once
     * (implementations should be prepared to handle this case).
     */
    virtual void Done() = 0;

    /**
     * Internal method to handle scaling correctly. This method
     * must not be called by a client. Clients should
     * always use the method <code>worked(int)</code>.
     *
     * @param work the amount of work done
     */
    virtual void InternalWorked(double work) = 0;

    /**
     * Returns whether cancelation of current operation has been requested.
     * Long-running operations should poll to see if cancelation
     * has been requested.
     *
     * @return <code>true</code> if cancellation has been requested,
     *         and <code>false</code> otherwise
     * @see #setCanceled(boolean)
     */
    virtual bool IsCanceled() = 0;

    /**
     * Sets the cancel state to the given value.
     *
     * @param canceled <code>true</code> indicates that cancelation has
     *                 been requested (but not necessarily acknowledged);
     *                 <code>false</code> clears this flag
     * @see #isCanceled()
     */
    virtual void SetCanceled(bool canceled) = 0;

    /**
     * Sets the task name to the given value. This method is used to
     * restore the task label after a nested operation was executed.
     * Normally there is no need for clients to call this method.
     *
     * @param taskName the name (or description) of the main task
     * @see #beginTask(String, int)
     */
    virtual void SetTaskName(std::string_view task_name) = 0;

    /**
     * Notifies that a subtask of the main task is beginning.
     * Subtasks are optional; the main task might not have subtasks.
     *
     * @param subTaskName the name (or description) of the subtask
     */
    virtual void SetSubTaskName(std::string_view sub_task_name) = 0;

    /**
     * Notifies that a given number of work unit of the main task
     * has been completed. Note that this amount represents an
     * installment, as opposed to a cumulative amount of work done
     * to date.
     *
     * @param work the number of work units just completed
     */
    virtual void Worked(int work) = 0;

    virtual ~IProgressMonitor() = default;
};
}  // namespace ceres
}  // namespace alus
