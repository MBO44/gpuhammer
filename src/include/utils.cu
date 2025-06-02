#include <utils.cuh>
#include <iostream>

/**
 * @brief Returns the GPU clock value in nanoseconds.
 *
 * @param time GPU clock value
 * @return uint64_t time in nanoseconds
 */
uint64_t toNS(uint64_t time)
{
  /* System variable that should be constant in a run. */
  static long double clock_rate = []()
  {
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    std::cout << "Stable Max Clock Rate: "
              << ((long double)(device_prop.clockRate)) * 1000 << '\n';
    // std::cout << "Using Current Clock Rate: " << ((long double)(val)) <<
    // '\n';
    return ((long double)(device_prop.clockRate)) * 1000;
  }();

  // TODO: Later we might need dynamic clockRate for attack./
  return (time / clock_rate) * 1000000000.0;
}

/**
 * @brief Get the dimension of the suitable kernel for size
 *
 * @param size
 * @return std::tuple<int, int> first is Blocks and seconds is threads
 */
std::tuple<int, int> get_dim_from_size(uint64_t size)
{
  /* Constant in the system */
  static uint64_t maxThreads = []()
  {
    struct cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, 0);
    return device_prop.maxThreadsDim[2];
  }();

  /* Depending on size, dispatch dimension needed to handle the payload */
  int numBlocks = (size + (maxThreads - 1)) / maxThreads;
  int numThreads = size > maxThreads ? maxThreads : size;
  return std::make_tuple<>(numBlocks, numThreads);
}
