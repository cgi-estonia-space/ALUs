/**
 * This file is a filtered duplicate of a SNAP's
 * com.bc.ceres.core.VirtualDir.java
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

#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

namespace alus::ceres {
/**
 * A read-only directory that can either be a directory in the file system or a ZIP-file.
 * Files having '.gz' extensions are automatically decompressed.
 *
 * @author original java version author: Norman Fomferra
 * @since Ceres 0.11
 */
class IVirtualDir {
private:
    static boost::filesystem::path GetBaseTempDir();

protected:
    static constexpr int TEMP_DIR_ATTEMPTS = 10000;

public:
    IVirtualDir() = default;
    IVirtualDir(const IVirtualDir&) = delete;
    IVirtualDir& operator=(const IVirtualDir&) = delete;
    virtual ~IVirtualDir() = default;
    /**
     * Creates an instance of a virtual directory object from a given directory or ZIP-file.
     *
     * @param file A directory or a ZIP-file.
     * @return The virtual directory instance, or {@code null} if {@code file} is not a directory or a ZIP-file or
     * the ZIP-file could not be opened for read access..
     */
    static std::shared_ptr<IVirtualDir> Create(const boost::filesystem::path& file);

    virtual bool IsCompressed() = 0;

    /**
     * Returns an std::vector of strings naming the files and directories in the
     * directory denoted by the given relative directory path.
     * <p>
     * There is no guarantee that the name strings in the resulting std::vector
     * will appear in any specific order; they are not, in particular,
     * guaranteed to appear in alphabetical order.
     *
     * @param path The relative directory path.
     * @return An std::vector of strings naming the files and directories in the
     * directory denoted by the given relative directory path.
     * The std::vector will be empty if the directory is empty.
     * @throws IOException If the directory given by the relative path does not exists.
     */
    virtual std::vector<std::string> List(std::string_view path) = 0;

    /**
     * Gets the file for the given relative path.
     *
     * @param path The relative file or directory path.
     * @return Gets the file or directory for the specified file path.
     * @throws IOException If the file or directory does not exist or if it can't be extracted from a ZIP-file.
     */
    virtual boost::filesystem::path GetFile(std::string_view path) = 0;

    virtual bool Exists(std::string_view path) = 0;

    /**
     * Opens an input stream for the given relative file path.
     * Files having '.gz' extensions are automatically decompressed.
     *
     * @param path The relative file path.
     * @return An input stream for the specified relative path.
     * @throws IOException If the file does not exist or if it can't be opened for reading.
     */
    virtual void GetInputStream(std::string_view path, std::fstream& stream) = 0;

    static boost::filesystem::path CreateUniqueTempDir();

    /**
     * Deletes the directory <code>treeRoot</code> and all the content recursively.
     *
     * @param treeRoot directory to be deleted
     */
    static void DeleteFileTree(const boost::filesystem::path& tree_root);
    /**
     * Closes access to this virtual directory.
     */
    virtual void Close() = 0;
};
}  // namespace alus::ceres