#include "heuristic.h"
#include <string.h>

const char str_l2_heuristic[] = "l2";
const char str_l1_heuristic[] = "l1";
const char str_octile_heuristic[] = "octile";
const char str_custom_heuristic[] = "custom";

int parse_heuristic(const char *str_heuristic) {
  if (strcmp(str_heuristic, str_l2_heuristic) == 0) {
      return L2_HEURISTIC;
    } else if (strcmp(str_heuristic, str_l1_heuristic) == 0) {
      return L1_HEURISTIC;
    } else if (strcmp(str_heuristic, str_octile_heuristic) == 0) {
      return OCTILE_HEURISTIC;
    } else if (strcmp(str_heuristic, str_custom_heuristic) == 0) {
      return CUSTOM_HEURISTIC;
    } else {
      return INVALID_HEURISTIC;
    }
}