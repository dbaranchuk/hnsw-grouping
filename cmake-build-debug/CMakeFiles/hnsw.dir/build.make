# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.12

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/dbaranchuk/clion-2018.2.1/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/dbaranchuk/clion-2018.2.1/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dbaranchuk/hnsw-grouping

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dbaranchuk/hnsw-grouping/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/hnsw.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hnsw.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hnsw.dir/flags.make

CMakeFiles/hnsw.dir/hnswalg.cpp.o: CMakeFiles/hnsw.dir/flags.make
CMakeFiles/hnsw.dir/hnswalg.cpp.o: ../hnswalg.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dbaranchuk/hnsw-grouping/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hnsw.dir/hnswalg.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hnsw.dir/hnswalg.cpp.o -c /home/dbaranchuk/hnsw-grouping/hnswalg.cpp

CMakeFiles/hnsw.dir/hnswalg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hnsw.dir/hnswalg.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dbaranchuk/hnsw-grouping/hnswalg.cpp > CMakeFiles/hnsw.dir/hnswalg.cpp.i

CMakeFiles/hnsw.dir/hnswalg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hnsw.dir/hnswalg.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dbaranchuk/hnsw-grouping/hnswalg.cpp -o CMakeFiles/hnsw.dir/hnswalg.cpp.s

CMakeFiles/hnsw.dir/utils.cpp.o: CMakeFiles/hnsw.dir/flags.make
CMakeFiles/hnsw.dir/utils.cpp.o: ../utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dbaranchuk/hnsw-grouping/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/hnsw.dir/utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/hnsw.dir/utils.cpp.o -c /home/dbaranchuk/hnsw-grouping/utils.cpp

CMakeFiles/hnsw.dir/utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hnsw.dir/utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dbaranchuk/hnsw-grouping/utils.cpp > CMakeFiles/hnsw.dir/utils.cpp.i

CMakeFiles/hnsw.dir/utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hnsw.dir/utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dbaranchuk/hnsw-grouping/utils.cpp -o CMakeFiles/hnsw.dir/utils.cpp.s

# Object files for target hnsw
hnsw_OBJECTS = \
"CMakeFiles/hnsw.dir/hnswalg.cpp.o" \
"CMakeFiles/hnsw.dir/utils.cpp.o"

# External object files for target hnsw
hnsw_EXTERNAL_OBJECTS =

lib/libhnsw.a: CMakeFiles/hnsw.dir/hnswalg.cpp.o
lib/libhnsw.a: CMakeFiles/hnsw.dir/utils.cpp.o
lib/libhnsw.a: CMakeFiles/hnsw.dir/build.make
lib/libhnsw.a: CMakeFiles/hnsw.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dbaranchuk/hnsw-grouping/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library lib/libhnsw.a"
	$(CMAKE_COMMAND) -P CMakeFiles/hnsw.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hnsw.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hnsw.dir/build: lib/libhnsw.a

.PHONY : CMakeFiles/hnsw.dir/build

CMakeFiles/hnsw.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hnsw.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hnsw.dir/clean

CMakeFiles/hnsw.dir/depend:
	cd /home/dbaranchuk/hnsw-grouping/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dbaranchuk/hnsw-grouping /home/dbaranchuk/hnsw-grouping /home/dbaranchuk/hnsw-grouping/cmake-build-debug /home/dbaranchuk/hnsw-grouping/cmake-build-debug /home/dbaranchuk/hnsw-grouping/cmake-build-debug/CMakeFiles/hnsw.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hnsw.dir/depend

