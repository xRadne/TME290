#include <iostream>
#include "cluon-complete.hpp"
#include "prime-checker.hpp"

int32_t main(int32_t, char **) {
  PrimeChecker pc;
  std::cout << "Hello World! Value = " << pc.isPrime(43) << std::endl;
  cluon::UDPSender sender{"127.0.0.1", 1234};
  sender.send("Hello UDP world!");
  return 0;
}
