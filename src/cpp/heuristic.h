#pragma once

#ifndef H_HEURISTIC
#define H_HEURISTIC

enum {
  L2_HEURISTIC,
  L1_HEURISTIC,
  OCTILE_HEURISTIC,
  CUSTOM_HEURISTIC,
  INVALID_HEURISTIC = -1,
};

/**
 * Parses the heuristic input string.
 * @param str_heuristic heuristic input string
 * @return heuristic the used heuristic or -1 (INVALID_HEURISTIC) on failure
**/
int parse_heuristic(const char *str_heuristic);

#endif