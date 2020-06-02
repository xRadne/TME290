#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string.h>
#include <math.h>  

///////////////////////////////////////////////////////////////////////////////////////
// Help functions:

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

std::vector<cv::Point> getConesPositions(opendlv::logic::perception::Cones const &conesReading) noexcept
{
    std::vector<int> iCones = csvStringToVector(conesReading.i());
    std::vector<int> jCones = csvStringToVector(conesReading.j());

    std::vector<cv::Point> cones;
    for (uint32_t k = 0; k < iCones.size(); k++) {
        cv::Point position = cv::Point(iCones[k]+640, 240-jCones[k]); // shift back to the coordinate system of the image
        cones.push_back(position);
    }

    return cones;
}

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ) {
        std::cerr << argv[0] << " collects information from all microservices for development/debugging." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=640 --height=480 --verbose" << std::endl;
    }
    else {
        const std::string NAME{commandlineArguments["name"]};
        const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
        const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
        uint16_t const CID = std::stoi(commandlineArguments["cid"]);
        const bool VERBOSE{commandlineArguments.count("verbose") != 0};

        // Attach to the shared memory.
        std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
        if (sharedMemory && sharedMemory->valid()) {
            std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

            cluon::OD4Session od4{CID};

            // Handler to receive cones messages:
            std::mutex blueConesMutex;
            std::mutex yellowConesMutex;
            std::mutex redConesMutex;
            std::vector<cv::Point> blueCones;
            std::vector<cv::Point> yellowCones;
            std::vector<cv::Point> redCones;
            auto onCones = [&blueConesMutex, &yellowConesMutex, &redConesMutex, &blueCones, &yellowCones, &redCones](cluon::data::Envelope &&env){
                opendlv::logic::perception::Cones conesMessage = cluon::extractMessage<opendlv::logic::perception::Cones>(std::move(env));

                // Store cone positions.
                if (conesMessage.color() == 0) {
                    std::lock_guard<std::mutex> lock(blueConesMutex);
                    blueCones = getConesPositions(conesMessage);   
                } else if (conesMessage.color() == 1) {
                    std::lock_guard<std::mutex> lock(yellowConesMutex);
                    yellowCones = getConesPositions(conesMessage);
                } else if (conesMessage.color() == 2) {
                    std::lock_guard<std::mutex> lock(redConesMutex);
                    redCones = getConesPositions(conesMessage);
                }
            };
            
            // Handler to receive aim point messages:
            std::mutex aimPointMutex;
            float azimuthAngle{0};
            float distance{0};
            auto onAimPoint = [&aimPointMutex, &azimuthAngle, &distance](cluon::data::Envelope &&env){
                opendlv::logic::action::AimPoint aimPointMessage = cluon::extractMessage<opendlv::logic::action::AimPoint>(std::move(env));

                std::lock_guard<std::mutex> lock(aimPointMutex);
                azimuthAngle = aimPointMessage.azimuthAngle();
                distance = aimPointMessage.distance();
            };

            // Handler to receive kiwi messages:
            // TODO How does this mutex now which variables to protect?
            std::mutex kiwiMutex;
            int16_t kiwi_center_x = 0;
            int16_t kiwi_center_y = 0;
            int16_t kiwi_width = 0;
            int16_t kiwi_height = 0;
            auto onKiwi = [&kiwiMutex, &kiwi_center_x, &kiwi_center_y, &kiwi_width, &kiwi_height](cluon::data::Envelope &&env){
                opendlv::logic::perception::Kiwi kiwiMessage = cluon::extractMessage<opendlv::logic::perception::Kiwi>(std::move(env));

                std::lock_guard<std::mutex> lock(kiwiMutex);
                kiwi_center_x = kiwiMessage.centreI();
                kiwi_center_y = kiwiMessage.centreJ();
                kiwi_width = kiwiMessage.width();
                kiwi_height = kiwiMessage.height();
            };

            od4.dataTrigger(opendlv::logic::action::AimPoint::ID(), onAimPoint);
            od4.dataTrigger(opendlv::logic::perception::Cones::ID(), onCones);
            od4.dataTrigger(opendlv::logic::perception::Kiwi::ID(), onKiwi);


            while (od4.isRunning()) {
                cv::Mat img;

                // Wait for a notification of a new frame.
                sharedMemory->wait();

                // Lock the shared memory.
                sharedMemory->lock();
                {
                    cv::Mat wrapped(HEIGHT, WIDTH, CV_8UC4, sharedMemory->data());
                    img = wrapped.clone();
                }
                sharedMemory->unlock();

                // Remove the upper part as well as the part of the own kiwi car by cropping:
                cv::Rect middlePartOfImage(0,370,WIDTH,240);
                img = img(middlePartOfImage);           

                ////////////////////////////////////////////////////////////////
                // Retrieve current cone positions:
                std::vector<cv::Point> blueConesCopy;
                std::vector<cv::Point> yellowConesCopy;
                std::vector<cv::Point> redConesCopy;
                {
                    std::lock_guard<std::mutex> lock(blueConesMutex);
                    std::lock_guard<std::mutex> lock2(yellowConesMutex);
                    std::lock_guard<std::mutex> lock3(redConesMutex);
                    blueConesCopy = blueCones;
                    yellowConesCopy = yellowCones;
                    redConesCopy = redCones;
                }

                // Retrieve current aim point:
                float azimuthAngleCopy{0};
                float distanceCopy{0};
                {
                    std::lock_guard<std::mutex> lock(aimPointMutex);
                    azimuthAngleCopy = azimuthAngle;
                    distanceCopy = distance;
                }

                cv::Point2d aimPoint = cv::Point2d(-distanceCopy*sin(azimuthAngleCopy)+640,-distanceCopy*cos(azimuthAngleCopy)+240);

                // Retrieve current kiwi bounding box:
                int16_t kiwi_center_x_copy = 0;
                int16_t kiwi_center_y_copy = 0;
                int16_t kiwi_width_copy = 0;
                int16_t kiwi_height_copy = 0;
                {
                    std::lock_guard<std::mutex> lock(kiwiMutex);
                    kiwi_center_x_copy = kiwi_center_x;
                    kiwi_center_y_copy = kiwi_center_y;
                    kiwi_width_copy = kiwi_width;
                    kiwi_height_copy = kiwi_height;
                }

                ////////////////////////////////////////////////////////////////
                // Display all information:
                if (VERBOSE) {
                    // Cone positions:
                    for (size_t i = 0; i < blueConesCopy.size(); i++) {
                        cv::Scalar color = cv::Scalar(255,0,0);
                        circle(img, blueConesCopy[i], 4, color, -1, 8, 0 );
                    }
                    for (size_t i = 0; i < yellowConesCopy.size(); i++) {
                        cv::Scalar color = cv::Scalar(0,255,255);
                        circle(img, yellowConesCopy[i], 4, color, -1, 8, 0 );
                    }
                    for (size_t i = 0; i < redConesCopy.size(); i++) {
                        cv::Scalar color = cv::Scalar(0,0,255);
                        circle(img, redConesCopy[i], 4, color, -1, 8, 0 );
                    }

                    // Aim point:
                    cv::Scalar color = cv::Scalar(255,255,255);
                    circle(img, aimPoint, 4, color, -1, 8, 0);
                    line(img, cv::Point2d(640,240), aimPoint, color, 2);

                    // Kiwi bounding box
                    // TODO add flag for toggle
                    cv::Point2d top_left = cv::Point2d(kiwi_center_x_copy - kiwi_width_copy/2.0,
                                                   (kiwi_center_y_copy + kiwi_height_copy/2.0) - 370.0);
                    cv::Point2d bottom_right = cv::Point2d(kiwi_center_x_copy + kiwi_width_copy/2.0,
                                                       (kiwi_center_y_copy - kiwi_height_copy/2.0) - 370.0);
                    rectangle(img, top_left, bottom_right, cv::Scalar(0, 0, 255));


                    imshow("Opendlv Development", img);
                    cv::waitKey(1);
                }
            }
        }
        retCode = 0;
    }
    return retCode;
}
