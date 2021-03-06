#include "prime-checker.hpp"

bool PrimeChecker::isPrime(uint16_t n) {
  bool retVal{true};
  if (n == 2) {
    return true;
  }
  if (n < 2 || 0 == n%2) {
    return false;
  } else {
    for (uint16_t i{3}; (i*i) <= n; i += 2) {
      if (0 == n%i) {
        return false;
      }    
    }
  }
  return retVal;
}
