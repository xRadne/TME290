#include "cluon-complete.hpp"
#include "opendlv-extended-message-set.hpp"
#include "pathfollower.hpp"

int32_t main(int32_t argc, char **argv) {
  int32_t retCode{0};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if (0 == commandlineArguments.count("cid") || 0 == commandlineArguments.count("freq")) {
    std::cerr << argv[0] << " plans and follows a path based on observed cones input." << std::endl;
    std::cerr << "Usage:   " << argv[0] << " --freq=<Integration frequency> --cid=<OpenDaVINCI session> [--kp=<PID proportional constant. Default: 0.2> --ki=<PID integration constant. Default: 0.0> --kd=<PID derivative constant. Default: 0.0> --verbose]" << std::endl;
    std::cerr << "Example: " << argv[0] << " --freq=10 --cid=111" << std::endl;
    retCode = 1;
  } else {
    bool const VERBOSE{commandlineArguments.count("verbose") != 0};
    uint16_t const CID = std::stoi(commandlineArguments["cid"]);
    float const FREQ = std::stof(commandlineArguments["freq"]);
    double const K_P = (0 == commandlineArguments.count("kp")) ? 0.2 : std::stod(commandlineArguments["kp"]);
    double const K_I = (0 == commandlineArguments.count("ki")) ? 0.0 : std::stod(commandlineArguments["ki"]);
    double const K_D = (0 == commandlineArguments.count("kd")) ? 0.0 : std::stod(commandlineArguments["kd"]);

    PathFollower pathfollower(K_P, K_I, K_D);
    cluon::OD4Session od4{CID};

    auto onDistanceReading{[&pathfollower](cluon::data::Envelope &&envelope)
      {
        auto distanceReading = cluon::extractMessage<opendlv::proxy::DistanceReading>(std::move(envelope));
        uint32_t const senderStamp = envelope.senderStamp();
        if (senderStamp == 0) {
          pathfollower.setFrontUltrasonic(distanceReading);
        } else if (senderStamp == 1) {
          pathfollower.setRearUltrasonic(distanceReading);
        }
      }};

    auto onVoltageReading{[&pathfollower](cluon::data::Envelope &&envelope)
      {
        auto voltageReading = cluon::extractMessage<opendlv::proxy::VoltageReading>(std::move(envelope));
        uint32_t const senderStamp = envelope.senderStamp();
        if (senderStamp == 0) {
          pathfollower.setLeftIr(voltageReading);
        } else if (senderStamp == 1) {
          pathfollower.setRightIr(voltageReading);
        }
      }};

    auto onConesReading{[&pathfollower](cluon::data::Envelope &&envelope)
      {
        auto conesReading = cluon::extractMessage<opendlv::logic::perception::Cones>(std::move(envelope));
        pathfollower.setConesPixelPosition(conesReading);
      }};

    auto onKiwiReading{[&pathfollower](cluon::data::Envelope &&envelope)
      {
        auto kiwiReading = cluon::extractMessage<opendlv::logic::perception::Kiwi>(std::move(envelope));
        pathfollower.setKiwiObject(kiwiReading);
      }};
    
    auto atFrequency{[&VERBOSE, &pathfollower, &od4]() -> bool
      {
        pathfollower.step();
        
        auto groundSteeringAngle = pathfollower.getGroundSteeringAngle();
        auto pedalPositionRequest = pathfollower.getPedalPositionRequest();
        cluon::data::TimeStamp sampleTime = cluon::time::now();
        
        od4.send(groundSteeringAngle);
        od4.send(pedalPositionRequest);

        // For opendlv-development:
        auto aimPoint = pathfollower.getAimPoint();
        od4.send(aimPoint);

        if (VERBOSE) {
          std::cout 
            << "Actuation sent:" 
            << " GroundSteering=" << groundSteeringAngle.groundSteering()
            << " PedalPosition=" << pedalPositionRequest.position()
            << std::endl;
        }

        return true;
      }};

    od4.dataTrigger(opendlv::proxy::DistanceReading::ID(), onDistanceReading);
    od4.dataTrigger(opendlv::proxy::VoltageReading::ID(), onVoltageReading);
    od4.dataTrigger(opendlv::logic::perception::Cones::ID(), onConesReading);
    od4.dataTrigger(opendlv::logic::perception::Kiwi::ID(), onKiwiReading);
    od4.timeTrigger(FREQ, atFrequency);
  }
  return retCode;
}
