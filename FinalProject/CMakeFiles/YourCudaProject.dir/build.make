# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /srv/home/pthombre/GPUProject/repo759/FinalProject

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /srv/home/pthombre/GPUProject/repo759/FinalProject

# Include any dependencies generated for this target.
include CMakeFiles/YourCudaProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/YourCudaProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/YourCudaProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/YourCudaProject.dir/flags.make

CMakeFiles/YourCudaProject.dir/main.cu.o: CMakeFiles/YourCudaProject.dir/flags.make
CMakeFiles/YourCudaProject.dir/main.cu.o: CMakeFiles/YourCudaProject.dir/includes_CUDA.rsp
CMakeFiles/YourCudaProject.dir/main.cu.o: main.cu
CMakeFiles/YourCudaProject.dir/main.cu.o: CMakeFiles/YourCudaProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv/home/pthombre/GPUProject/repo759/FinalProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/YourCudaProject.dir/main.cu.o"
	/opt/apps/cuda/x86_64/11.8.0/default/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/YourCudaProject.dir/main.cu.o -MF CMakeFiles/YourCudaProject.dir/main.cu.o.d -x cu -c /srv/home/pthombre/GPUProject/repo759/FinalProject/main.cu -o CMakeFiles/YourCudaProject.dir/main.cu.o

CMakeFiles/YourCudaProject.dir/main.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/YourCudaProject.dir/main.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/YourCudaProject.dir/main.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/YourCudaProject.dir/main.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o: CMakeFiles/YourCudaProject.dir/flags.make
CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o: CMakeFiles/YourCudaProject.dir/includes_CUDA.rsp
CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o: NonLinearElasticity.cu
CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o: CMakeFiles/YourCudaProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/srv/home/pthombre/GPUProject/repo759/FinalProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o"
	/opt/apps/cuda/x86_64/11.8.0/default/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o -MF CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o.d -x cu -c /srv/home/pthombre/GPUProject/repo759/FinalProject/NonLinearElasticity.cu -o CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o

CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target YourCudaProject
YourCudaProject_OBJECTS = \
"CMakeFiles/YourCudaProject.dir/main.cu.o" \
"CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o"

# External object files for target YourCudaProject
YourCudaProject_EXTERNAL_OBJECTS =

YourCudaProject: CMakeFiles/YourCudaProject.dir/main.cu.o
YourCudaProject: CMakeFiles/YourCudaProject.dir/NonLinearElasticity.cu.o
YourCudaProject: CMakeFiles/YourCudaProject.dir/build.make
YourCudaProject: CMakeFiles/YourCudaProject.dir/linkLibs.rsp
YourCudaProject: CMakeFiles/YourCudaProject.dir/objects1.rsp
YourCudaProject: CMakeFiles/YourCudaProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/srv/home/pthombre/GPUProject/repo759/FinalProject/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CUDA executable YourCudaProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/YourCudaProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/YourCudaProject.dir/build: YourCudaProject
.PHONY : CMakeFiles/YourCudaProject.dir/build

CMakeFiles/YourCudaProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/YourCudaProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/YourCudaProject.dir/clean

CMakeFiles/YourCudaProject.dir/depend:
	cd /srv/home/pthombre/GPUProject/repo759/FinalProject && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /srv/home/pthombre/GPUProject/repo759/FinalProject /srv/home/pthombre/GPUProject/repo759/FinalProject /srv/home/pthombre/GPUProject/repo759/FinalProject /srv/home/pthombre/GPUProject/repo759/FinalProject /srv/home/pthombre/GPUProject/repo759/FinalProject/CMakeFiles/YourCudaProject.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/YourCudaProject.dir/depend

