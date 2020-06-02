#include <iostream>
#include <chrono>
#include <sstream>
#include <math.h>
#define PI 3.14159265

#include "pathfollower.hpp"

namespace constants {
  // standard motion parameters
  const float standardSpeed = 0.06f;
  const float maxSpeed = 0.1f;
  const float EMERGENCY_BREAK_DISTANCE_FRONT = 0.1f;
  
  // image parameters
  int const width = 1280;
  int const min_i = - width / 2;
  int const max_i = width / 2;
  int const height = 720;
  
  // task 1 cone and path processing parameters
  int const heightThreshold = 150;
  
  // task 2 following parameters
  const float maxBrakingDistance = 1.2f;
  const float minBrakingDistance = 0.8f;
  const float proportionalBrake = static_cast<float>(standardSpeed/(2*(maxBrakingDistance-minBrakingDistance)));
  
  // task 3 intersection parameters
  const float intersectionStoppingDistance = 1.5f;
  
  // kiwi detection parameters
  float const heightKiwiM = 0.109f;
  int const perceivedFocalLength = 1204;
  float const kiwiDistanceFactor = perceivedFocalLength * heightKiwiM;
}

PathFollower::PathFollower() noexcept: PathFollower(0.2, 0.0, 0.0)
{
}

PathFollower::PathFollower(double kp, double ki, double kd) noexcept:
  m_kp{kp},
  m_ki{ki},
  m_kd{kd},
  m_frontUltrasonicReading{},
  m_rearUltrasonicReading{},
  m_leftIrReading{},
  m_rightIrReading{},
  m_kiwi{},
  m_blueCones{},
  m_yellowCones{},
  m_orangeCones{},
  m_groundSteeringAngleRequest{},
  m_pedalPositionRequest{},
  m_aimPoint{},
  m_frontUltrasonicReadingMutex{},
  m_rearUltrasonicReadingMutex{},
  m_leftIrReadingMutex{},
  m_rightIrReadingMutex{},
  m_blueConesMutex{},
  m_yellowConesMutex{},
  m_orangeConesMutex{},
  m_groundSteeringAngleRequestMutex{},
  m_pedalPositionRequestMutex{},
  m_kiwiMutex{},
  m_aimPointMutex{},
  m_useFrontUltrasonicSensor{false},
  m_steeringErrors{},
  m_lastTimeStep{cluon::time::now()},
  m_lastTimeStepMutex{},
  m_kiwiDetected{false},
  m_kiwiDetectedMutex{}
{
}

void PathFollower::setFrontUltrasonic(opendlv::proxy::DistanceReading const &frontUltrasonicReading) noexcept
{
  std::lock_guard<std::mutex> lock(m_frontUltrasonicReadingMutex);
  m_frontUltrasonicReading = frontUltrasonicReading;

  m_useFrontUltrasonicSensor = true;
}

void PathFollower::setRearUltrasonic(opendlv::proxy::DistanceReading const &rearUltrasonicReading) noexcept
{
  std::lock_guard<std::mutex> lock(m_rearUltrasonicReadingMutex);
  m_rearUltrasonicReading = rearUltrasonicReading;
}

void PathFollower::setLeftIr(opendlv::proxy::VoltageReading const &leftIrReading) noexcept
{
  std::lock_guard<std::mutex> lock(m_leftIrReadingMutex);
  m_leftIrReading = leftIrReading;
}

void PathFollower::setRightIr(opendlv::proxy::VoltageReading const &rightIrReading) noexcept
{
  std::lock_guard<std::mutex> lock(m_rightIrReadingMutex);
  m_rightIrReading = rightIrReading;
}

opendlv::proxy::GroundSteeringRequest PathFollower::getGroundSteeringAngle() noexcept
{
  std::lock_guard<std::mutex> lock(m_groundSteeringAngleRequestMutex);
  return m_groundSteeringAngleRequest;
}

opendlv::proxy::PedalPositionRequest PathFollower::getPedalPositionRequest() noexcept
{
  std::lock_guard<std::mutex> lock(m_pedalPositionRequestMutex);
  return m_pedalPositionRequest;
}

opendlv::logic::action::AimPoint PathFollower::getAimPoint() noexcept
{
  std::lock_guard<std::mutex> lock(m_aimPointMutex);
  return m_aimPoint;
}

/////////////////////////////////////////////////////////////////////////
// To parse the cone strings:
template<class T=int>
std::vector<T> csvStringToVector(std::string string, char delimiter=',') {
  std::vector<T> v;

  T x;
  std::istringstream str_buf{string};
  while ( str_buf >> x ) {
    v.push_back(x);
    // If the next char in input is a delimiter, extract it. std::ws discards whitespace
    if ( ( str_buf >> std::ws).peek() == delimiter ) 
      str_buf.ignore();
  }
  
  return v;
}
/////////////////////////////////////////////////////////////////////////

// Assuming coordinate system with origin at bottom centre of image. Ie closest point to kiwi. 
float angleTo(Pixel pixel) {
  float dx = static_cast<float>(pixel.i);
  float dy = static_cast<float>(pixel.j);

  double angle = atan2(dy, dx);
  return static_cast<float>(angle - PI / 2);
}

Pixel meanConePixel(std::vector<Pixel> cones) {
  Pixel pixel;
  double sumI = 0.0, sumJ = 0.0;

  for (Pixel p : cones) {
    sumI += static_cast<double>(p.i);
    sumJ += static_cast<double>(p.j);
  }

  pixel.i = static_cast<int>(round(sumI / (double)cones.size()));
  pixel.j = static_cast<int>(round(sumJ / (double)cones.size()));
  return pixel;
}

/* Weighting -->
TOP OF IMAGE
\
 \
  \
BOTTOM OF IMAGE */
Pixel weightedMeanConePixel(std::vector<Pixel> cones) {
  int topOfImage = 240;
  Pixel pixel;
  double weightedSumI = 0.0, sumJ = 0.0;
  double sumWeights = 0.0;

  for (Pixel p : cones) {
    double w = static_cast<double>(topOfImage - p.j);
    weightedSumI += static_cast<double>(p.i) * w;
    sumJ += static_cast<double>(p.j);
    sumWeights += w;
  }

  pixel.i = static_cast<int>(round(weightedSumI / sumWeights));
  pixel.j = static_cast<int>(round(sumJ / (double)cones.size()));
  return pixel;
}

template<typename T=double>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
    auto first = v.cbegin() + m;
    auto last = v.cbegin() + n + 1;

    std::vector<T> vec(first, last);
    return vec;
}

double integrate(std::vector<double> values, double dt) {
  double sum = 0;
  for (double v : values) {
    sum += v;
  }
  return sum * dt;
}

void PathFollower::setConesPixelPosition(opendlv::logic::perception::Cones const &conesReading) noexcept
{
  // conesReading.coneId();
  std::vector<int> iCones = csvStringToVector(conesReading.i());
  std::vector<int> jCones = csvStringToVector(conesReading.j());

  std::vector<Pixel> cones;
  for (uint32_t k = 0; k < iCones.size(); k++) {
    Pixel pixel = {iCones[k], jCones[k]};
    cones.push_back(pixel);
  }

  if (conesReading.color() == 0) /* Blue */ {
    std::lock_guard<std::mutex> lock(m_blueConesMutex);
    m_blueCones = cones;
  } else if (conesReading.color() == 1) /* Yellow */ {
    std::lock_guard<std::mutex> lock(m_yellowConesMutex);
    m_yellowCones = cones;
  } else if (conesReading.color() == 2) /* Orange */ {
    std::lock_guard<std::mutex> lock(m_orangeConesMutex);
    m_orangeCones = cones;
  }
}

void PathFollower::setKiwiObject(opendlv::logic::perception::Kiwi const &kiwiReading) noexcept
{
  std::lock_guard<std::mutex> lock(m_kiwiMutex);
  m_kiwi = kiwiReading;
}

void PathFollower::step() noexcept
{
  /////////////////////////////////////////////////////////////////////////
  // Get cone positions:
  std::vector<Pixel> blueCones;
  std::vector<Pixel> yellowCones;
  std::vector<Pixel> orangeCones;
  {
    std::lock_guard<std::mutex> lock1(m_blueConesMutex);
    std::lock_guard<std::mutex> lock2(m_yellowConesMutex);
    std::lock_guard<std::mutex> lock3(m_orangeConesMutex);
    
    blueCones = m_blueCones;
    yellowCones = m_yellowCones;
    orangeCones = m_orangeCones;
  }

  /////////////////////////////////////////////////////////////////////////
  // Get kiwi object:
  int centreI_kiwi;
  int centreJ_kiwi;
  int width_kiwi;
  int height_kiwi;
  bool kiwiDetectedInLastStep;

  {
    std::lock_guard<std::mutex> lock1(m_kiwiMutex);
    std::lock_guard<std::mutex> lock2(m_kiwiDetectedMutex);

    centreI_kiwi = m_kiwi.centreI();
    centreJ_kiwi = m_kiwi.centreJ();
    width_kiwi = m_kiwi.width();
    height_kiwi = m_kiwi.height();

    kiwiDetectedInLastStep = m_kiwiDetected;
  }
  
  (void) centreJ_kiwi;

  /////////////////////////////////////////////////////////////////////////
  // Get sensor readings and old requests:
  opendlv::proxy::DistanceReading frontUltrasonicReading;
  {
    std::lock_guard<std::mutex> lock1(m_frontUltrasonicReadingMutex);

    frontUltrasonicReading = m_frontUltrasonicReading;
  }

  float frontDistanceSensor{0};
  if (!m_useFrontUltrasonicSensor) {
    frontDistanceSensor = 2.0f;
  } else {
    frontDistanceSensor = frontUltrasonicReading.distance();
  }

  /////////////////////////////////////////////////////////////////////////
  // Calculate the distance to the kiwi car:
  float frontDistanceKiwi;
  
  if (height_kiwi != 0) {
    frontDistanceKiwi = constants::kiwiDistanceFactor / static_cast<float>(height_kiwi);
  }
  else {
    frontDistanceKiwi = 100;
  }

  // Check if a kiwi was detected in this step or last step:
  bool kiwiDetected = !(width_kiwi == 0 || height_kiwi == 0);

  {
    std::lock_guard<std::mutex> lock(m_kiwiDetectedMutex);
    m_kiwiDetected = kiwiDetected;
  }

  if (!kiwiDetected && kiwiDetectedInLastStep) {
    // If detected a kiwi in last message, set kiwiDetected to true anyway for stability:
    kiwiDetected = true;
  }  

  /////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////
  // Get the time difference since last iteration:
  cluon::data::TimeStamp thisTimeStep = cluon::time::now();
  int64_t thisTimeStepUs = cluon::time::toMicroseconds(thisTimeStep);
  int64_t lastTimeStepUs;
  {
    std::lock_guard<std::mutex> lock(m_lastTimeStepMutex);

    lastTimeStepUs = cluon::time::toMicroseconds(m_lastTimeStep);
    m_lastTimeStep = thisTimeStep;
  }
  double DT = (thisTimeStepUs - lastTimeStepUs) * 0.000001; // in seconds 
  
  /////////////////////////////////////////////////////////////////////////
  //// BEHAVIOR                      //////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////

  float pedalPosition;
  float groundSteeringAngle;
  Pixel aimPixel{0, 0};

  // Emergency break behavior
  if (frontDistanceSensor < constants::EMERGENCY_BREAK_DISTANCE_FRONT || frontDistanceKiwi < constants::EMERGENCY_BREAK_DISTANCE_FRONT) {
    pedalPosition = 0.0f;
    if (width_kiwi == 0) {
      groundSteeringAngle = 0.0f;
    }
    else if (centreI_kiwi < 0){
      // if car is left steer right
      groundSteeringAngle = -0.5f;
    }
    else {
      // if car is right steer left
      groundSteeringAngle = 0.5f;
    }
  }

  // No cones detected behavior
  else if (blueCones.size() + yellowCones.size() + orangeCones.size() == 0) {
    pedalPosition = 0.5*constants::standardSpeed;
    groundSteeringAngle = 0.0f;
  }

  // Crossing behavior:
  else if (orangeCones.size() >= 3) { 
    // Stop if a Kiwi is detected in the right part of the image that is close enough:
    if (kiwiDetected && centreI_kiwi > 0 && frontDistanceKiwi < constants::intersectionStoppingDistance) {
      pedalPosition = 0; 
    } else {
      pedalPosition = constants::standardSpeed;
    }

    // Use the mean of the orange cones as aim pixel:
    Pixel orangeMean = weightedMeanConePixel(orangeCones);
    aimPixel = {orangeMean.i, orangeMean.j};
    float aimAngle = angleTo(aimPixel);

    groundSteeringAngle = 0.2f*aimAngle;

  }

  // Follow path between cones
  else {
    /////////////////////////////////////////////////////////////////////////
    // Calculate the ground steering angle:
    float aimAngle;
    {
      Pixel blueMean;
      Pixel yellowMean;

      // Ignore cones at the far top of the image

      for (int i = blueCones.size() - 1; i >= 0; i--) {
        if (blueCones[i].j > constants::heightThreshold) {
          blueCones.erase(blueCones.begin() + i);
        }
      }

      for (int i = yellowCones.size() - 1; i >= 0; i--) {
        if (yellowCones[i].j > constants::heightThreshold) {
          yellowCones.erase(yellowCones.begin() + i);
        }
      }
    
      // Weighted mean
      if (blueCones.size() > 0) {
        blueMean = weightedMeanConePixel(blueCones);
      } else {
        blueMean = Pixel{constants::max_i, constants::height/2};
      }

      if (yellowCones.size() > 0) {
        yellowMean = weightedMeanConePixel(yellowCones);
      } else {
        yellowMean = Pixel{constants::min_i, constants::height/2};
      }

      int meanI = static_cast<int>(round((blueMean.i + yellowMean.i) / 2.0));
      int meanJ = static_cast<int>(round((blueMean.j + yellowMean.j) / 2.0));
      aimPixel = {meanI, meanJ};

      aimAngle = angleTo(aimPixel);
    }

    // PID control
    
    m_steeringErrors.push_back(aimAngle);
    int N = m_steeringErrors.size();
    // auto lastFiveErrorAngles = slice(m_steeringErrors, std::max(0, N-5), N-1);

    double errorAngle = m_steeringErrors[N-1];
    // double errorIntegral = integrate(lastFiveErrorAngles, dt);
    double errorAngleRate = (m_steeringErrors[N-2] - m_steeringErrors[N-1]) / DT;

    double steering = m_kp * errorAngle /*+ m_di * errorIntegral */+ m_kd * errorAngleRate;
    groundSteeringAngle = static_cast<float>(steering);

    /////////////////////////////////////////////////////////////////////////
    // Calculate the speed:
    {
      float frontDistance = std::min(frontDistanceKiwi, frontDistanceSensor);
  
      if (width_kiwi == 0){ // speed based on angle
        pedalPosition = 0.01f * 1/abs(groundSteeringAngle);
        pedalPosition = (pedalPosition > constants::maxSpeed) ? constants::maxSpeed : pedalPosition;
        pedalPosition = (pedalPosition < 0.8f * constants::standardSpeed) ? 0.8f * constants::standardSpeed: pedalPosition;

      }
      else { // speed based on kiwi distance
        if (frontDistance > constants::maxBrakingDistance) {
          pedalPosition = constants::standardSpeed;
        } else if (frontDistance > constants::minBrakingDistance) {
          pedalPosition = static_cast<float>((frontDistance - constants::minBrakingDistance) * constants::proportionalBrake);
        } else {
          pedalPosition = 0;
        }
      }
      
    }  
  }

  // For opendlv-development:
  float distance = static_cast<float> (sqrt((float)(aimPixel.i * aimPixel.i + aimPixel.j * aimPixel.j)));
  // std::cout << "Aim: (" << aimPixel.i << "," << aimPixel.j << ")" << std::endl;
  {
    std::lock_guard<std::mutex> lock1(m_groundSteeringAngleRequestMutex);
    std::lock_guard<std::mutex> lock2(m_pedalPositionRequestMutex);
    std::lock_guard<std::mutex> lock3(m_aimPointMutex);

    opendlv::proxy::GroundSteeringRequest groundSteeringAngleRequest;
    groundSteeringAngleRequest.groundSteering(groundSteeringAngle);
    m_groundSteeringAngleRequest = groundSteeringAngleRequest;

    opendlv::proxy::PedalPositionRequest pedalPositionRequest;
    pedalPositionRequest.position(pedalPosition);
    m_pedalPositionRequest = pedalPositionRequest;

    opendlv::logic::action::AimPoint aimPointMessage;
    aimPointMessage.azimuthAngle(groundSteeringAngle);
    aimPointMessage.distance(distance);
    m_aimPoint = aimPointMessage;
  }
}

// From chalmers-revere/opendlv-logic-test-kiwi
double PathFollower::convertIrVoltageToDistance(float voltage) const noexcept
{
  double voltageDividerR1 = 1000.0;
  double voltageDividerR2 = 1000.0;

  double sensorVoltage = (voltageDividerR1 + voltageDividerR2) / voltageDividerR2 * voltage;
  double distance = (2.5 - sensorVoltage) / 0.07;
  return distance;
}