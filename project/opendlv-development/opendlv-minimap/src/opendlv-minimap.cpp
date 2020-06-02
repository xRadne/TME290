/*
 * Copyright (C) 2020 Ola Benderius
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"
#include <cmath>

// jason reading
// #include <json/value.h>
#include <fstream>

#define WIDTH 400

#define MIN_POSITION -4.0
#define MAX_POSITION 4.0

struct GridPoint {
  uint32_t i;
  uint32_t j;

  GridPoint(uint32_t a_i, uint32_t a_j): i(a_i), j(a_j) {}
};

struct Point {
  double x;
  double y;

  Point(double a_x, double a_y): x(a_x), y(a_y) {}
};

struct Line {
  double x0;
  double x1;
  double y0;
  double y1;

  Line(double a_x0, double a_y0, double a_x1, double a_y1):
    x0(a_x0), x1(a_x1), y0(a_y0), y1(a_y1) {}

  Point p0() {
    return Point(x0, y0);
  }

  Point p1() {
    return Point(x1, y1);
  }
};

// Functions
bool checkIntersection(Line a, Line b){
  double s0_x{a.x1 - a.x0};
  double s0_y{a.y1 - a.y0};
  double s1_x{b.x1 - b.x0};
  double s1_y{b.y1 - b.y0};

  double s{(-s0_y * (a.x0 - b.x0) + s0_x * (a.y0 - b.y0)) / 
    (-s1_x * s0_y + s0_x * s1_y)};
  double t{(s1_x * (a.y0 - b.y0) - s1_y * (a.x0 - b.x0)) / 
    (-s1_x * s0_y + s0_x * s1_y)};

  if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
    return true;
  }
  return false;
}

void drawRotatedRectangle(cv::Mat& image, cv::Point centerPoint, cv::Size rectangleSize, float rotationDegrees)
{
  cv::Scalar color = cv::Scalar(255.0, 255.0, 255.0); // white

  // Create the rotated rectangle
  cv::RotatedRect rotatedRectangle(centerPoint, rectangleSize, rotationDegrees);

  // We take the edges that OpenCV calculated for us
  cv::Point2f vertices2f[4];
  rotatedRectangle.points(vertices2f);

  // Convert them so we can use them in a fillConvexPoly
  cv::Point vertices[4];    
  for(int i = 0; i < 4; ++i){
    vertices[i] = vertices2f[i];
  }

  // Now we can fill the rotated rectangle with our specified color
  cv::fillConvexPoly(image,
                      vertices,
                      4,
                      color);
}

double distance(Point p1, Point p2) {
  return sqrt((p2.x-p1.x)*(p2.x-p1.x) + (p2.y-p1.y)*(p2.y-p1.y));
}

double aimAngle(Point source, double direction, Point target) {
  double dx = target.x - source.x;
  double dy = target.y - source.y;
  double angle = atan2(dy, dx);
  return angle - direction;
}

void ReplaceStringInPlace(std::string& subject, const std::string& search,
                          const std::string& replace) {
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos) {
         subject.replace(pos, search.length(), replace);
         pos += replace.length();
    }
}

std::vector<Point> arrayStringToPoints(std::string source) {
  std::vector<std::string> arrayStrings = stringtoolbox::split(source, '[');
  std::vector<Point> positions;
  for (unsigned i = 1; i < arrayStrings.size(); i++) {
    int length = arrayStrings[i].size();
    int end = length;
    if (arrayStrings[i][length-1] == ',' && arrayStrings[i][length-2] == ']') {
      end -= 2;
    } else if (arrayStrings[i][length-1] == ']') {
      end -= 1;
    }
    std::string insideArray = arrayStrings[i].substr(0, end);
    std::vector<std::string> coordinates = stringtoolbox::split(insideArray, ',');

    double x = std::stod(coordinates[0]);
    double y = std::stod(coordinates[1]);
    Point p = Point(x, y);
    positions.push_back(p);

  }
  return positions;
}

std::vector<Point> readMap(std::string source, std::string search) {
    std::string regexString = "\\\"" + search + "\\\",.*?\\\"instances\\\":\\[(.*?\\])\\]";
    std::regex rgx(regexString);
    std::smatch match;
    const std::string constSource = source;
    std::regex_search(constSource.begin(), constSource.end(), match, rgx);
  return arrayStringToPoints(match[1]);
}

// Main function
int32_t main(int32_t argc, char **argv) {
  int32_t retCode{0};
  auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
  if (0 == commandlineArguments.count("cid") 
      || 0 == commandlineArguments.count("map-file")
      || 0 == commandlineArguments.count("frame-id")
      || 0 == commandlineArguments.count("freq")) {
    std::cerr << argv[0] << " plots a top down view of a simulation map json file." << std::endl;
    std::cerr << "Example: " << argv[0] << " --cid=111 --freq=10 --frame-id=0"
      "--map-file=/opt/simulation-map.json --verbose" << std::endl;
    retCode = 1;
  } else {
    bool const verbose = (commandlineArguments.count("verbose") != 0);

    // Set up the OD4 session
    uint16_t const cid = std::stoi(commandlineArguments["cid"]);
    float const freq = std::stof(commandlineArguments["freq"]);
    uint32_t const frameId = static_cast<uint32_t>(
        std::stoi(commandlineArguments["frame-id"]));
    std::string const mapFile = commandlineArguments["map-file"];
    std::cout << "Reading from " << mapFile << std::endl;

    // Visualise the path
    cv::Mat background;
    {
      background = cv::Mat(WIDTH, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));

      std::ifstream inputJson(mapFile);
      std::string jsonStr((std::istreambuf_iterator<char>(inputJson)), std::istreambuf_iterator<char>());

      ReplaceStringInPlace(jsonStr, " ", "");
      ReplaceStringInPlace(jsonStr, "\n", "");

      std::vector<Point> redCones = readMap(jsonStr, "cone_red");
      for (Point p : redCones) {
        int x =  static_cast<uint32_t>((p.x - MIN_POSITION)/(MAX_POSITION-MIN_POSITION) * static_cast<double>(WIDTH));
        int y =  static_cast<uint32_t>((p.y - MIN_POSITION)/(MAX_POSITION-MIN_POSITION) * static_cast<double>(WIDTH));
        cv::circle(background, cv::Point(x, y), 2, cv::Scalar(0, 0, 255), -1, 8);
      }
      std::vector<Point> blueCone = readMap(jsonStr, "cone_blue");
      for (Point p : blueCone) {
        int x =  static_cast<uint32_t>((p.x - MIN_POSITION)/(MAX_POSITION-MIN_POSITION) * static_cast<double>(WIDTH));
        int y =  static_cast<uint32_t>((p.y - MIN_POSITION)/(MAX_POSITION-MIN_POSITION) * static_cast<double>(WIDTH));
        cv::circle(background, cv::Point(x, y), 2, cv::Scalar(255, 0, 0), -1, 8);
      }
      std::vector<Point> yellowCone = readMap(jsonStr, "cone_yellow");
      for (Point p : yellowCone) {
        int x =  static_cast<uint32_t>((p.x - MIN_POSITION)/(MAX_POSITION-MIN_POSITION) * static_cast<double>(WIDTH));
        int y =  static_cast<uint32_t>((p.y - MIN_POSITION)/(MAX_POSITION-MIN_POSITION) * static_cast<double>(WIDTH));
        cv::circle(background, cv::Point(x, y), 2, cv::Scalar(0, 255, 255), -1, 8);
      }
    }

    cluon::OD4Session od4(cid);

    opendlv::sim::Frame latestFrame;
    double distanceFront = 0.0;
    double distanceLeft = 0.0;
    double distanceRear = 0.0;
    double distanceRight = 0.0;
    opendlv::logic::action::AimPoint aimPoint;

    std::mutex frameMutex;
    std::mutex distanceMutex;
    std::mutex aimPointMutex;
    
    auto onAimPoint{[&frameId, &aimPoint, &aimPointMutex](
        cluon::data::Envelope &&envelope)
      {
        uint32_t const senderStamp = envelope.senderStamp();
        if (frameId == senderStamp) {
          std::lock_guard<std::mutex> const lock(aimPointMutex);
          aimPoint = cluon::extractMessage<opendlv::logic::action::AimPoint>(
              std::move(envelope));
        }
    }};

    auto onFrame{[&frameId, &latestFrame, &frameMutex, &verbose](
        cluon::data::Envelope &&envelope)
      {
        uint32_t const senderStamp = envelope.senderStamp();
        if (frameId == senderStamp) {
          std::lock_guard<std::mutex> const lock(frameMutex);
          latestFrame = cluon::extractMessage<opendlv::sim::Frame>(
              std::move(envelope));
        }
    }};

    auto onDistanceReading{[&distanceFront, &distanceRear, &distanceMutex](
        cluon::data::Envelope &&envelope)
      {
        uint32_t const senderStamp = envelope.senderStamp();
        auto distanceReading = 
          cluon::extractMessage<opendlv::proxy::DistanceReading>(
              std::move(envelope));
          
        std::lock_guard<std::mutex> const lock(distanceMutex);
        if (senderStamp == 0) {
          distanceFront = distanceReading.distance();
        } else {
          distanceRear = distanceReading.distance();
        }
      }};

    auto onVoltageReading{[&distanceLeft, &distanceRight, &distanceMutex](
        cluon::data::Envelope &&envelope)
      {
        uint32_t const senderStamp = envelope.senderStamp();
        auto voltageReading = 
          cluon::extractMessage<opendlv::proxy::VoltageReading>(
              std::move(envelope));

        double voltageDividerR1 = 1000.0;
        double voltageDividerR2 = 1000.0;

        double sensorVoltage = (voltageDividerR1 + voltageDividerR2) 
          / voltageDividerR2 * voltageReading.voltage();
        double distance = (2.5 - sensorVoltage) / 0.07;

        std::lock_guard<std::mutex> const lock(distanceMutex);
        if (senderStamp == 0) {
          distanceLeft = distance;
        } else {
          distanceRight = distance;
        }
      }};

    auto atFrequency{[background, &latestFrame, &frameMutex, &distanceFront, &distanceLeft, 
      &distanceRear, &distanceRight, &distanceMutex, &aimPoint, &aimPointMutex, &od4, &verbose]() -> bool
      {
        double posX;
        double posY;
        double posYaw;
        double distFront;
        double distLeft;
        double distRear;
        double distRight;
        double groundSteeringAngle;
        {
          std::lock_guard<std::mutex> const lock(frameMutex);
          posX = latestFrame.x();
          posY = latestFrame.y();
          posYaw = latestFrame.yaw();
        }
        {
          std::lock_guard<std::mutex> const lock(distanceMutex);
          distFront = distanceFront;
          distLeft = distanceLeft;
          distRear = distanceRear;
          distRight = distanceRight;
        }
        {
          std::lock_guard<std::mutex> const lock(aimPointMutex);
          groundSteeringAngle = static_cast<double>(aimPoint.azimuthAngle());
        }

        (void) posX;
        (void) posY;
        (void) posYaw;
        (void) distFront;
        (void) distLeft;
        (void) distRear;
        (void) distRight;
        
        
        // Image output
        if (verbose) {
          cv::Mat map = background.clone();

          // Draw Kiwi
          {
            int x = (int)((posX - MIN_POSITION)/(MAX_POSITION - MIN_POSITION) * (double) WIDTH);
            int y = (int)((posY - MIN_POSITION)/(MAX_POSITION - MIN_POSITION) * (double) WIDTH);
            float degrees = (float)(posYaw / 3.141592 * 180.0);

            int kiwiLength = (int)((0.36)/(MAX_POSITION - MIN_POSITION) * (double) WIDTH);
            int kiwiWidth = (int)((0.16)/(MAX_POSITION - MIN_POSITION) * (double) WIDTH);
            cv::Size kiwiSize(kiwiLength, kiwiWidth);
            drawRotatedRectangle(map, cv::Point(x, y), kiwiSize, degrees);
            
            int ax = (int)((posX + cos(posYaw + groundSteeringAngle) - MIN_POSITION)/(MAX_POSITION - MIN_POSITION) * (double) WIDTH);
            int ay = (int)((posY  + sin(posYaw + groundSteeringAngle) - MIN_POSITION)/(MAX_POSITION - MIN_POSITION) * (double) WIDTH);
            cv::line(map, cv::Point(x, y), cv::Point(ax, ay), cv::Scalar(255.0, 255.0, 255.0));
          }

          cv::flip(map, map, 0);
          cv::imshow("Global map", map);
          cv::waitKey(1);
        }

        return true;
      }};

    // Register the three data triggers, each spawning a thread
    od4.dataTrigger(opendlv::sim::Frame::ID(), onFrame);
    od4.dataTrigger(opendlv::proxy::DistanceReading::ID(), onDistanceReading);
    od4.dataTrigger(opendlv::proxy::VoltageReading::ID(), onVoltageReading);
    od4.dataTrigger(opendlv::logic::action::AimPoint::ID(), onAimPoint);
    
    // Register the time trigger, spawning a thread that blocks execution 
    // until CTRL-C is pressed
    od4.timeTrigger(freq, atFrequency);
  }
  return retCode;
}
