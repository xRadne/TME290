#ifndef PATHFOLLOWER
#define PATHFOLLOWER

#include <mutex>
#include <vector>

#include "cluon-complete.hpp"
#include "opendlv-extended-message-set.hpp"

struct Pixel
{
    int i;
    int j;
};

class PathFollower {
 private:
  PathFollower(PathFollower const &) = delete;
  PathFollower(PathFollower &&) = delete;
  PathFollower &operator=(PathFollower const &) = delete;
  PathFollower &operator=(PathFollower &&) = delete;

 public:
  PathFollower() noexcept;
  PathFollower(double, double, double) noexcept;
  ~PathFollower() = default;

 public:
  opendlv::proxy::GroundSteeringRequest getGroundSteeringAngle() noexcept;
  opendlv::proxy::PedalPositionRequest getPedalPositionRequest() noexcept;
  opendlv::logic::action::AimPoint getAimPoint() noexcept;
  void setConesPixelPosition(opendlv::logic::perception::Cones const &) noexcept;
  void setKiwiObject(opendlv::logic::perception::Kiwi const &) noexcept;
  void setFrontUltrasonic(opendlv::proxy::DistanceReading const &) noexcept;
  void setRearUltrasonic(opendlv::proxy::DistanceReading const &) noexcept;
  void setLeftIr(opendlv::proxy::VoltageReading const &) noexcept;
  void setRightIr(opendlv::proxy::VoltageReading const &) noexcept;
  void step() noexcept;

 private:
  double convertIrVoltageToDistance(float) const noexcept;

 private:
  double m_kp;
  double m_ki;
  double m_kd;
  opendlv::proxy::DistanceReading m_frontUltrasonicReading;
  opendlv::proxy::DistanceReading m_rearUltrasonicReading;
  opendlv::proxy::VoltageReading m_leftIrReading;
  opendlv::proxy::VoltageReading m_rightIrReading;

  opendlv::logic::perception::Kiwi m_kiwi;

  std::vector<Pixel> m_blueCones;
  std::vector<Pixel> m_yellowCones;
  std::vector<Pixel> m_orangeCones;

  opendlv::proxy::GroundSteeringRequest m_groundSteeringAngleRequest;
  opendlv::proxy::PedalPositionRequest m_pedalPositionRequest;
  opendlv::logic::action::AimPoint m_aimPoint;

  std::mutex m_frontUltrasonicReadingMutex;
  std::mutex m_rearUltrasonicReadingMutex;
  std::mutex m_leftIrReadingMutex;
  std::mutex m_rightIrReadingMutex;
  std::mutex m_blueConesMutex;
  std::mutex m_yellowConesMutex;
  std::mutex m_orangeConesMutex;
  std::mutex m_groundSteeringAngleRequestMutex;
  std::mutex m_pedalPositionRequestMutex;
  std::mutex m_kiwiMutex;
  std::mutex m_aimPointMutex;
  
  bool m_useFrontUltrasonicSensor;
  std::vector<double> m_steeringErrors;

  cluon::data::TimeStamp m_lastTimeStep;
  std::mutex m_lastTimeStepMutex;

  bool m_kiwiDetected;
  std::mutex m_kiwiDetectedMutex;
};

#endif
