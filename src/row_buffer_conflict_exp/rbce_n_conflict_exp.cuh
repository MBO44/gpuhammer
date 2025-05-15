#include <fstream>
#include <stdint.h>
#ifndef GPU_ROWHAMMER_RBCE_N_CONFLICT_EXP_H
#define GPU_ROWHAMMER_RBCE_N_CONFLICT_EXP_H

namespace rbce
{

const auto default_cond = []() { return true; };

class N_Conflict
{
private:
  uint64_t *TIME_ARR_DEVICE; /* GPU side time array */
  uint64_t *TIME_ARR_HOST;   /* HOST side time array */
  uint8_t *ADDR_LAYOUT;      /* GPU memory layout */
  uint8_t **ADDR_LST_DEVICE; /* GPU side memory address array */
  uint8_t **ADDR_LST_HOST;   /* HOST side memory address array */

  /* Input Arguments */
  uint64_t N;           /* How many accesses are done in a kernel */
  uint64_t EXP_RANGE;   /* RANGE we run the expriment over */
  uint64_t LAYOUT_SIZE; /* SIZE of memory layout */
  uint64_t EXP_IT;      /* Number of iteration for a single access */
  uint64_t STEP_SIZE;   /* Steps we skip for each access */

  /* System dependent variable */
  uint64_t CLOCK_RATE; /* Stores the system clock rate */

public:
  N_Conflict(uint64_t N, uint64_t LAYOUT_SIZE, uint64_t EXP_RANGE,
             uint64_t EXP_IT, uint64_t STEP_SIZE);
  ~N_Conflict();

  uint8_t *get_addr_layout() { return ADDR_LAYOUT; };
  uint64_t get_exp_range() { return EXP_RANGE; };
  uint8_t **get_addr_lst_host() { return ADDR_LST_HOST; };
  uint64_t get_step() { return STEP_SIZE; };
  uint64_t get_addr_lst_elm(uint64_t idx);
  void set_exp_range(uint64_t EXP_RANGE) { this->EXP_RANGE = EXP_RANGE; };
  void set_addr_lst_host(uint64_t idx, uint64_t ofs);

  /**
   * @brief Runs the device code to access addresses stored in ADDR_LST_HOST
   * at the same time. The code is written with the assumption that you are
   * unning it within 1 single block with <= 32 threads.
   *
   * @param file writes the time values to file.
   * @return uint64_t time value of the access.
   */
  uint64_t repeat_n_addr_exp(std::ofstream *file = nullptr);

  /**
   * @brief Runs f through the experiment range with a step size start from i.
   *
   * https://stackoverflow.com/questions/24392000/define-a-for-loop-macro-in-c
   * @tparam FUNCTION lambda type holder
   * @param f function to run for each step
   * @param i initial step
   */
  template <typename FUNCTION>
  inline void loop_range(FUNCTION &&f, uint64_t i = 0)
  {
    for (; i < this->EXP_RANGE; i += this->STEP_SIZE)
    {
      std::forward<FUNCTION>(f)(i);
    }
  }

  /**
   * @brief Runs f through until cond is not met with a step size and start
   * from i.
   *
   * @tparam FUNCTION (uint64_t) -> any
   * @tparam COND (uint64_t) -> bool
   * @param f function to run for each step
   * @param cond custom condition to stop
   * @param i initial step
   */
  template <typename FUNCTION, typename COND>
  inline void loop_range(FUNCTION &&f, COND &&cond, uint64_t i = 0)
  {
    for (; std::forward<COND>(cond)(i); i += this->STEP_SIZE)
    {
      std::forward<FUNCTION>(f)(i);
    }
  }
};

} // namespace rbce
#endif /* GPU_ROWHAMMER_RBCE_N_CONFLICT_EXP_H */