#include <fstream>
#include <stdint.h>
#include <stdio.h>
#include <tuple>
#include <vector>
#ifndef GPU_ROWHAMMER_UTILS_CUH
#define GPU_ROWHAMMER_UTILS_CUH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

uint64_t toNS(uint64_t time);

std::tuple<int, int> get_dim_from_size(uint64_t size);

/* Returns vector in string form, thats it. */
template <typename T> std::string vector_str(const std::vector<T> &vec)
{
  std::ostringstream oss;
  oss << '[';
  std::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(oss, ", "));
  oss << ']';
  return oss.str();
}


#endif /* GPU_ROWHAMMER_UTILS_CUH */