#include "rbce_n_conflict_exp.cuh"
#include <iostream>

int main(int argc, char *argv[])
{
  /* Argument EXP_RANGE, EXP_IT, STEP */
  rbce::N_Conflict nc_test(2, std::stoull(argv[1]), std::stoull(argv[2]),
                           std::stoull(argv[3]), std::stoull(argv[4]));

  /* Argument Threshold */
  uint64_t threshold = std::stoull(argv[5]);

  /* Offset to an address in Target Bank */
  uint64_t offset_to_bank = std::stoull(argv[6]);

  std::ofstream offset_file;
  offset_file.open(argv[7]); /* Argument File name */

  /* Initialize address pairs */
  nc_test.set_addr_lst_host(0, offset_to_bank);
  nc_test.set_addr_lst_host(1, offset_to_bank);
  uint64_t base_delay = nc_test.repeat_n_addr_exp();
  for (int i = 0; i < 100000; i++)
  {
	  base_delay = nc_test.repeat_n_addr_exp();
  }
  cudaDeviceSynchronize();
  uint64_t conflict_delay = base_delay + threshold;

  nc_test.loop_range(
      [&](uint64_t step)
      {
        nc_test.set_addr_lst_host(1, step);

        /* Found conflict */
        if (conflict_delay < nc_test.repeat_n_addr_exp())
        {
          /* Prepare for run of 'step' with itself */
          nc_test.set_addr_lst_host(0, step);

          /* Should be in reasonable range of base_delay as in same bank chip */
          if (std::abs((int32_t)(nc_test.repeat_n_addr_exp()) -
                       (int32_t)(base_delay)) <= 10)
          {
            offset_file << step << '\n';
          }

          /* Reset [0] */
          nc_test.set_addr_lst_host(0, offset_to_bank);
        }
      });

  offset_file.close();
  return 0;
}
