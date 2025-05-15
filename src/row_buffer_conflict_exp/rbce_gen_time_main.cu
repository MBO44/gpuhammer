#include "rbce_n_conflict_exp.cuh"
#include <iostream>

int main(int argc, char *argv[])
{
  /* Argument EXP_RANGE, EXP_IT, STEP */
  rbce::N_Conflict nc_test(2, std::stoull(argv[1]), std::stoull(argv[2]),
                           std::stoull(argv[3]), std::stoull(argv[4]));

  std::ofstream *time_file = new std::ofstream;
  time_file->open(argv[5]); /* Argument File name */

  /* Initialize address pairs */
  nc_test.set_addr_lst_host(0, 0);
  nc_test.set_addr_lst_host(1, 0);
  nc_test.repeat_n_addr_exp(); /* First kernel is slower */

  nc_test.loop_range(
      [&nc_test, time_file](uint64_t step)
      {
        nc_test.set_addr_lst_host(1, step);
        nc_test.repeat_n_addr_exp(time_file);
      });

  time_file->close();
  return 0;
}