cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES 86)

project(GPURowhammer LANGUAGES CXX CUDA)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3 -Xcicc -O0 -Xptxas -O3 -gencode=arch=compute_86,code=sm_86")

find_package(nvtx3 REQUIRED PATHS "./rmm_lib/lib/cmake/nvtx3" NO_DEFAULT_PATH)
find_package(RMM REQUIRED PATHS "./rmm_lib/lib/cmake/rmm" NO_DEFAULT_PATH)
include_directories(./rmm_lib/include)
link_directories(./rmm_lib/lib)

include_directories(include)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(alloc SHARED ./exploit/allocator.cu)
target_link_libraries(alloc PRIVATE rmm::rmm)

# Reverse Engineering
add_library(re_lib ./re_gddr/drama_conflict_prober.cu ./include/utils.cu)

# Get Conflict Set
add_executable(conf_set ./re_gddr/conf_set_main.cu)
target_link_libraries(conf_set PUBLIC re_lib)

# Get Row Set
add_executable(row_set ./re_gddr/row_set_main.cu)
target_link_libraries(row_set PUBLIC re_lib)

# Get Bank Set
add_executable(get_banks ./re_gddr/get_banks_main.cu)
target_link_libraries(get_banks PUBLIC re_lib)

# Get Timing values
add_executable(gen_time ./re_gddr/gen_time_main.cu)
target_link_libraries(gen_time PUBLIC re_lib)

# Get Timing values
add_executable(gen_time_same ./re_gddr/gen_time_same_main.cu)
target_link_libraries(gen_time_same PUBLIC re_lib)

add_executable(load_modifiers ./re_gddr/load_modifiers_main.cu)
target_link_libraries(load_modifiers PUBLIC re_lib)

# Hammer
add_library(hammer_lib ./include/cuda_helpers.cu ./include/hammer_util.cu ./include/hammer_kernels.cu ./include/utils.cu)
target_link_libraries(hammer_lib PUBLIC Threads::Threads)

## Simple hammer
add_executable(simple_hammer ./hammer/simple_hammer.cu )
target_link_libraries(simple_hammer PRIVATE hammer_lib)
target_link_libraries(simple_hammer PRIVATE Threads::Threads)

add_executable(sync_delay ./hammer/synchronization.cu)
target_link_libraries(sync_delay PUBLIC hammer_lib)
target_link_libraries(sync_delay PRIVATE Threads::Threads)

add_executable(mt_sync_delay ./hammer/synchronization_multi_thread.cu)
target_link_libraries(mt_sync_delay PUBLIC hammer_lib)
target_link_libraries(mt_sync_delay PRIVATE Threads::Threads)

add_executable(st_sync_delay ./hammer/synchronization_single_thread.cu)
target_link_libraries(st_sync_delay PUBLIC hammer_lib)
target_link_libraries(st_sync_delay PRIVATE Threads::Threads)

add_executable(gpu_hammer ./hammer/gpu_hammer.cu)
target_link_libraries(gpu_hammer PUBLIC hammer_lib)
target_link_libraries(gpu_hammer PRIVATE Threads::Threads)

add_executable(trr_sampler ./hammer/trr_sampler_size.cu)
target_link_libraries(trr_sampler PUBLIC hammer_lib)
target_link_libraries(trr_sampler PRIVATE Threads::Threads)

add_executable(trh ./hammer/trh.cu)
target_link_libraries(trh PUBLIC hammer_lib)
target_link_libraries(trh PRIVATE Threads::Threads)

add_executable(hammer_manual_agg_left ./exploit/hammer_manual_agg_left.cu)
target_link_libraries(hammer_manual_agg_left PUBLIC hammer_lib)
target_link_libraries(hammer_manual_agg_left PRIVATE Threads::Threads CUDA::cuda_driver)
target_link_libraries(hammer_manual_agg_left PRIVATE rmm::rmm)
target_link_libraries(hammer_manual_agg_left PRIVATE alloc)

add_executable(hammer_manual_agg_right ./exploit/hammer_manual_agg_right.cu)
target_link_libraries(hammer_manual_agg_right PUBLIC hammer_lib)
target_link_libraries(hammer_manual_agg_right PRIVATE Threads::Threads CUDA::cuda_driver)
target_link_libraries(hammer_manual_agg_right PRIVATE rmm::rmm)
target_link_libraries(hammer_manual_agg_right PRIVATE alloc)

add_executable(hammer_mem_manage ./exploit/hammer_mem_manage.cu)
target_link_libraries(hammer_mem_manage PUBLIC hammer_lib)
target_link_libraries(hammer_mem_manage PRIVATE Threads::Threads CUDA::cuda_driver)
target_link_libraries(hammer_mem_manage PRIVATE rmm::rmm)
target_link_libraries(hammer_mem_manage PRIVATE alloc)

