#include "rbce_n_conflict_exp.cuh"
#include <algorithm>
#include <cuda_helpers.cuh>
#include <iostream>
namespace rbce
{

N_Conflict::N_Conflict(uint64_t N, uint64_t LAYOUT_SIZE, uint64_t EXP_RANGE,
                       uint64_t EXP_IT, uint64_t STEP_SIZE)
    : N{N}, LAYOUT_SIZE{LAYOUT_SIZE}, EXP_RANGE{EXP_RANGE}, EXP_IT{EXP_IT},
      STEP_SIZE{STEP_SIZE}
{
  cudaMalloc(&(this->ADDR_LAYOUT), this->LAYOUT_SIZE);
  cudaMalloc(&(this->TIME_ARR_DEVICE), sizeof(uint64_t) * this->EXP_IT);
  cudaMalloc(&(this->ADDR_LST_DEVICE), this->N * sizeof(uint8_t *));
  this->ADDR_LST_HOST = new uint8_t *[N];
  this->TIME_ARR_HOST = new uint64_t[this->EXP_IT];

  struct cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  this->CLOCK_RATE = device_prop.clockRate;
}

N_Conflict::~N_Conflict()
{
  cudaFree(this->ADDR_LAYOUT);
  cudaFree(this->TIME_ARR_DEVICE);
  cudaFree(this->ADDR_LST_DEVICE);
  delete[] this->ADDR_LST_HOST;
  delete[] this->TIME_ARR_HOST;
}

uint64_t N_Conflict::get_addr_lst_elm(uint64_t idx)
{
  if (idx >= this->N)
    throw std::out_of_range("Index greater than address list size");

  return this->ADDR_LST_HOST[idx] - this->ADDR_LAYOUT;
}

void N_Conflict::set_addr_lst_host(uint64_t idx, uint64_t ofs)
{
  if (idx >= this->N)
    throw std::out_of_range("Index greater than address list size");

  this->ADDR_LST_HOST[idx] = this->ADDR_LAYOUT + ofs;
}

uint64_t N_Conflict::repeat_n_addr_exp(std::ofstream *file)
{
  /* Copy the addresses to GPU usable memory */
  cudaMemcpy(this->ADDR_LST_DEVICE, this->ADDR_LST_HOST,
             this->N * sizeof(uint8_t *), cudaMemcpyHostToDevice);

  /* Run experiment EXP_IT times to avoid noise */
  for (uint64_t i = 0; i < this->EXP_IT; i++)
  {
    n_address_conflict_kernel<<<1, this->N>>>(this->ADDR_LST_DEVICE,
                                              this->TIME_ARR_DEVICE + i);
  }
  cudaDeviceSynchronize();

  /* Copy the time values from GPU to HOST usable memory */
  cudaMemcpy(this->TIME_ARR_HOST, this->TIME_ARR_DEVICE,
             sizeof(uint64_t) * this->EXP_IT, cudaMemcpyDeviceToHost);

  /* The true delay is consistent and noise will only cause the delay to go
     up, thus we take the minimum. Convert it to NS for better understanding.
  */
  uint64_t min = toNS(*std::min_element(this->TIME_ARR_HOST,
                                        this->TIME_ARR_HOST + this->EXP_IT));

  if (file)
  {
    *file << '(';
    for (int i = 0; i < this->N; i++)
      // For some reason uint8_t* not printable
      *file << (void *)(*(this->ADDR_LST_HOST + i)) << ", ";
    *file << ")\t" << min << '\n';
  }

  return min;
}

} // namespace rbce
