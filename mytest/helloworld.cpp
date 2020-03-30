#include <chrono>
#include <iostream>

#include "cluon-complete.hpp"
#include "prime-checker.hpp"
#include "messages.hpp"

int32_t main(int32_t, char **) {
	PrimeChecker pc;
	std::cout << "Hello world = " << pc.isPrime(23) << std::endl;

	cluon::UDPSender sender{"225.0.0.111", 1238};
	
	uint16_t value;
	std::cout << "Enter a number to check: ";
	std::cin >> value;
	MyTestMessage1 msg;
	msg.myValue(value);
	cluon::ToProtoVisitor encoder;
	msg.accept(encoder);
	std::string data{encoder.encodedData()};
	sender.send(std::move(data));

	std::this_thread::sleep_for(std::chrono::duration<double>(5.0));

	cluon::UDPReceiver receiver("225.0.0.111", 1238,
		[](std::string &&data, std::string &&/*from*/,
			std::chrono::system_clock::time_point &&/*timepoint*/) noexcept {
				std::stringstream sstr{data};
				cluon::FromProtoVisitor decoder;
				decoder.decodeFrom(sstr);
				MyTestMessage1 receivedMsg;
				receivedMsg.accept(decoder);
				PrimeChecker pc;
				std::cout << receivedMsg.myValue() << " is " << (pc.isPrime(receivedMsg.myValue()) ? "" : "not") << " a prime" << std::endl;
		});

	while (receiver.isRunning()) {
		std::this_thread::sleep_for(std::chrono::duration<double>(1.0));
	}

	return 0;
}
