#include <iostream>
#include "prime-checker.hpp"

int32_t main(int32_t argc, char **argv) {
  PrimeChecker pc;
  std::cout << "Hello World! Value = " << pc.isPrime(43) << std::endl;
  return 0;
}
