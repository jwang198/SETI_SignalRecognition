# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus

# Include any dependencies generated for this target.
include CMakeFiles/dp_algo.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/dp_algo.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/dp_algo.dir/flags.make

CMakeFiles/dp_algo.dir/dp_algo.cpp.o: CMakeFiles/dp_algo.dir/flags.make
CMakeFiles/dp_algo.dir/dp_algo.cpp.o: dp_algo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/dp_algo.dir/dp_algo.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dp_algo.dir/dp_algo.cpp.o -c /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus/dp_algo.cpp

CMakeFiles/dp_algo.dir/dp_algo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dp_algo.dir/dp_algo.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus/dp_algo.cpp > CMakeFiles/dp_algo.dir/dp_algo.cpp.i

CMakeFiles/dp_algo.dir/dp_algo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dp_algo.dir/dp_algo.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus/dp_algo.cpp -o CMakeFiles/dp_algo.dir/dp_algo.cpp.s

CMakeFiles/dp_algo.dir/dp_algo.cpp.o.requires:

.PHONY : CMakeFiles/dp_algo.dir/dp_algo.cpp.o.requires

CMakeFiles/dp_algo.dir/dp_algo.cpp.o.provides: CMakeFiles/dp_algo.dir/dp_algo.cpp.o.requires
	$(MAKE) -f CMakeFiles/dp_algo.dir/build.make CMakeFiles/dp_algo.dir/dp_algo.cpp.o.provides.build
.PHONY : CMakeFiles/dp_algo.dir/dp_algo.cpp.o.provides

CMakeFiles/dp_algo.dir/dp_algo.cpp.o.provides.build: CMakeFiles/dp_algo.dir/dp_algo.cpp.o


# Object files for target dp_algo
dp_algo_OBJECTS = \
"CMakeFiles/dp_algo.dir/dp_algo.cpp.o"

# External object files for target dp_algo
dp_algo_EXTERNAL_OBJECTS =

dp_algo: CMakeFiles/dp_algo.dir/dp_algo.cpp.o
dp_algo: CMakeFiles/dp_algo.dir/build.make
dp_algo: /usr/local/lib/libopencv_videostab.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_ts.a
dp_algo: /usr/local/lib/libopencv_superres.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_stitching.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_contrib.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_nonfree.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_ocl.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_gpu.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_photo.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_objdetect.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_legacy.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_video.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_ml.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_calib3d.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_features2d.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_highgui.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_imgproc.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_flann.2.4.12.dylib
dp_algo: /usr/local/lib/libopencv_core.2.4.12.dylib
dp_algo: CMakeFiles/dp_algo.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable dp_algo"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dp_algo.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/dp_algo.dir/build: dp_algo

.PHONY : CMakeFiles/dp_algo.dir/build

CMakeFiles/dp_algo.dir/requires: CMakeFiles/dp_algo.dir/dp_algo.cpp.o.requires

.PHONY : CMakeFiles/dp_algo.dir/requires

CMakeFiles/dp_algo.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/dp_algo.dir/cmake_clean.cmake
.PHONY : CMakeFiles/dp_algo.dir/clean

CMakeFiles/dp_algo.dir/depend:
	cd /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus /Users/Jason/Desktop/Spark@SETI/timeseries_cplusplus/CMakeFiles/dp_algo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/dp_algo.dir/depend

