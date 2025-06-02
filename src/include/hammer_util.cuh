#include <atomic>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#ifndef GPU_ROWHAMMER_HAMMER_UTIL_CUH
#define GPU_ROWHAMMER_HAMMER_UTIL_CUH

using RowList = std::vector<std::vector<uint8_t *>>;
using Row = std::vector<uint8_t *>;

enum MEM_PAT
{
  VICTIM_PAT = 0xAA, 
  AGGRES_PAT = 0x55
  // Alternative padding patterns:
  // VICTIM_PAT = 0x55,
  // AGGRES_PAT = 0xAA
};

extern std::string CLI_PREFIX;

RowList read_row_from_file(std::ifstream &file, const uint8_t *base_addr);

std::vector<uint64_t> get_random_victims(RowList &rows, uint64_t v_count);

std::vector<uint64_t> get_random_sequential_victims(RowList &rows,
                                                    uint64_t v_count);

std::vector<uint64_t> get_sequential_victims(RowList &rows, uint64_t row_id,
                                             uint64_t v_count);

std::vector<uint64_t> get_sequential_victims(RowList &rows, uint64_t row_id,
                                             uint64_t num_vic, uint64_t step);

std::vector<uint64_t> get_aggressors(std::vector<uint64_t> &victims);

std::vector<uint64_t> get_aggressors(RowList &rows, uint64_t row_id,
                                     uint64_t num_agg, uint64_t step);

void set_rows(RowList &rows, std::vector<uint64_t> &target_rows, uint8_t pat,
              uint64_t b_count);

void clear_L2cache_rows(RowList &rows, std::vector<uint64_t> &target_rows, uint64_t step);

bool verify_content(RowList &rows, std::vector<uint64_t> &victims,
                    std::vector<uint64_t> &aggressors, uint64_t b_count,
                    uint8_t pat);

bool verify_all_content(RowList &rows, std::vector<uint64_t> &victims,
                        std::vector<uint64_t> &aggressors, 
                        const uint64_t b_count, const uint8_t pat);
void sleep_for(uint64_t time, char time_type);

void evict_L2cache(uint8_t *layout);

void print_time (uint64_t time_ns);

void initialize_rows(RowList &rows, uint64_t b_count);

#endif /* GPU_ROWHAMMER_HAMMER_UTIL_CUH */