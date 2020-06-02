#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string.h>

namespace colorLimits {
    cv::Scalar const hsvBlueLow(115, 120, 30); // Note: H [0,180], S [0,255], V [0,255] (H usually defined as [0,360] otherwise)
    cv::Scalar const hsvBlueHigh(145, 255, 200);
    cv::Scalar const hsvYellowLow(14, 100, 120);
    cv::Scalar const hsvYellowHigh(20, 190, 220);
    //cv::Scalar const hsvRedLow1(0, 120, 120);
    //cv::Scalar const hsvRedHigh1(10, 160, 160);
    cv::Scalar const hsvRedLow2(170, 120, 100);
    cv::Scalar const hsvRedHigh2(180, 180, 160);
}

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ) {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
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
                // Detect cones by color limits:
                cv::Mat blueCones;
                cv::Mat yellowCones;
                cv::Mat redCones;
                {
                    cv::Mat hsv;
                    cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

                    cv::inRange(hsv, colorLimits::hsvBlueLow, colorLimits::hsvBlueHigh, blueCones);
                    cv::inRange(hsv, colorLimits::hsvYellowLow, colorLimits::hsvYellowHigh, yellowCones);
                    cv::inRange(hsv, colorLimits::hsvRedLow2, colorLimits::hsvRedHigh2, redCones);

                    // Open, close and dilate:
                    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,cv::Size(3,3),cv::Point(-1,1));
                    for (int i = 0; i < 3; i++) {
                        cv::morphologyEx(blueCones, blueCones, cv::MORPH_OPEN, element, cv::Point(-1,-1), 1, 1, 1); // removes noise in the background
                        cv::morphologyEx(blueCones, blueCones, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 3, 1, 1); // removes noise in the cones
                        cv::morphologyEx(blueCones, blueCones, cv::MORPH_DILATE, element, cv::Point(-1, -1), 2, 1, 1); // enlarges the segmented area with cones

                        cv::morphologyEx(yellowCones, yellowCones, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1, 1, 1);
                        cv::morphologyEx(yellowCones, yellowCones, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 3, 1, 1);
                        cv::morphologyEx(yellowCones, yellowCones, cv::MORPH_DILATE, element, cv::Point(-1, -1), 2, 1, 1);

                        cv::morphologyEx(redCones, redCones, cv::MORPH_OPEN, element, cv::Point(-1, -1), 1, 1, 1);
                        cv::morphologyEx(redCones, redCones, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 3, 1, 1);
                        cv::morphologyEx(redCones, redCones, cv::MORPH_DILATE, element, cv::Point(-1, -1), 2, 1, 1);
                    }
                }

                // For output:
                cv::Mat allCones = blueCones + yellowCones + redCones;
                cv::Mat output;
                bitwise_and(img,img,output,allCones);

                ////////////////////////////////////////////////////////////////
                // Bounding boxes for each cone:
                std::vector<cv::Rect> rectBlueCones;
                std::vector<cv::Rect> rectYellowCones;
                std::vector<cv::Rect> rectRedCones;

                // Bottom centre position of each bounding box, for plotting:
                std::vector<cv::Point> positionBlueCones;
                std::vector<cv::Point> positionYellowCones;
                std::vector<cv::Point> positionRedCones;

                // Positions to send as output, where the coordinates are shifted so that
                // (640,240) is the origin (positive x-direction to the right, positive y-direction up):
                std::string i_blueCones{};
                std::string j_blueCones{};
                std::string i_yellowCones{};
                std::string j_yellowCones{};
                std::string i_redCones{};
                std::string j_redCones{};

                {
                    // Find countors of the segmented cones:
                    std::vector<std::vector<cv::Point>> contoursBlue;
                    std::vector<std::vector<cv::Point>> contoursYellow;
                    std::vector<std::vector<cv::Point>> contoursRed;
                    std::vector<cv::Vec4i> hierarchy;
                    findContours(blueCones, contoursBlue, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
                    findContours(yellowCones, contoursYellow, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
                    findContours(redCones, contoursRed, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

                    // Find the bounding rectangles of the contours and remove the contours that are not cones:
                    for ( uint32_t i = 0; i < contoursBlue.size(); i++) {
                        cv::RotatedRect rectRotated = cv::minAreaRect(contoursBlue[i]);
                        cv::Rect rect = rectRotated.boundingRect();

                        if (rect.width < 0.9*rect.height && (rect.area() > 600 && rect.area() < 4000) ) { // a cone is higher than it is wide
                            rectBlueCones.push_back(rect);

                            cv::Point bottomCentre = cv::Point(static_cast<int>(rect.x + rect.width/2),static_cast<int>(rect.y + rect.height));
                            positionBlueCones.push_back(bottomCentre);
                            i_blueCones += std::to_string(bottomCentre.x - 640) + ",";
                            j_blueCones += std::to_string(bottomCentre.y*(-1) + 240) + ","; // center position of the base of the cone
                        } 
                    }
                    for ( uint32_t i = 0; i < contoursYellow.size(); i++) {
                        cv::RotatedRect rectRotated = cv::minAreaRect(contoursYellow[i]); 
                        cv::Rect rect = rectRotated.boundingRect();

                        if (rect.width < 0.9*rect.height && (rect.area() > 600 && rect.area() < 4000)) { // a cone is higher than it is wide
                            rectYellowCones.push_back(rect);

                            cv::Point bottomCentre = cv::Point(static_cast<int>(rect.x + rect.width/2),static_cast<int>(rect.y + rect.height));
                            positionYellowCones.push_back(bottomCentre);
                            i_yellowCones += std::to_string(bottomCentre.x - 640) + ",";
                            j_yellowCones += std::to_string(bottomCentre.y*(-1) + 240) + ",";
                        } 
                    }
                    for ( uint32_t i = 0; i < contoursRed.size(); i++) {
                        cv::RotatedRect rectRotated = cv::minAreaRect(contoursRed[i]); 
                        cv::Rect rect = rectRotated.boundingRect();

                        if (rect.width < 0.9*rect.height  && (rect.area() > 600 && rect.area() < 4000)) { // a cone is higher than it is wide
                            rectRedCones.push_back(rect);

                            cv::Point bottomCentre = cv::Point(static_cast<int>(rect.x + rect.width/2),static_cast<int>(rect.y + rect.height));
                            positionRedCones.push_back(bottomCentre);
                            i_redCones += std::to_string(bottomCentre.x - 640) + ",";
                            j_redCones += std::to_string(bottomCentre.y*(-1) + 240) + ",";
                        } 
                    }

                    if (VERBOSE) {
                        std::cout << "Blue cones: i = " << i_blueCones << " , j = " << j_blueCones << std::endl;
                        std::cout << "Yellow cones: i = " << i_yellowCones << " , j = " << j_yellowCones << std::endl;
                        std::cout << "Red cones: i = " << i_redCones << " , j = " << j_redCones << std::endl;
                    }
                }

                ////////////////////////////////////////////////////////////////
                // Send the messages with cone positions:
                opendlv::logic::perception::Cones blueConesMsg;
                blueConesMsg.color(0);
                blueConesMsg.i(i_blueCones);
                blueConesMsg.j(j_blueCones);

                opendlv::logic::perception::Cones yellowConesMsg;
                yellowConesMsg.color(1);
                yellowConesMsg.i(i_yellowCones);
                yellowConesMsg.j(j_yellowCones);

                opendlv::logic::perception::Cones redConesMsg;
                redConesMsg.color(2);
                redConesMsg.i(i_redCones);
                redConesMsg.j(j_redCones);

                od4.send(blueConesMsg);
                od4.send(yellowConesMsg);
                od4.send(redConesMsg);

                ////////////////////////////////////////////////////////////////
                // Display image.
                if (VERBOSE) {
                    // Original image:
                    cv::Mat dispImage = cv::Mat::zeros(cv::Size(WIDTH, 3*240), CV_8UC4);
                    cv::Rect ROI_1(0,0,WIDTH,240);
                    img.copyTo(dispImage(ROI_1));
                    // Segmented image:
                    cv::Rect ROI_2(0,240,WIDTH,240);
                    output.copyTo(dispImage(ROI_2));
                    // Cones with bounding boxes:
                    for (size_t i = 0; i < rectBlueCones.size(); i++) {
                        cv::Scalar color = cv::Scalar(255,0,0);
                        rectangle(img, rectBlueCones[i], color, 2);
                        circle(img, positionBlueCones[i], 4, color, -1, 8, 0 );
                    }
                    for (size_t i = 0; i < rectYellowCones.size(); i++) {
                        cv::Scalar color = cv::Scalar(0,255,255);
                        rectangle(img, rectYellowCones[i], color, 2);
                        circle(img, positionYellowCones[i], 4, color, -1, 8, 0 );
                    }
                    for (size_t i = 0; i < rectRedCones.size(); i++) {
                        cv::Scalar color = cv::Scalar(0,0,255);
                        rectangle(img, rectRedCones[i], color, 2);
                        circle(img, positionRedCones[i], 4, color, -1, 8, 0 );
                    }
                    cv::Rect ROI_3(0,480,WIDTH,240);
                    img.copyTo(dispImage(ROI_3));
                    imshow("Cone detection", dispImage);
                    cv::waitKey(1);
                }
            }
        }
        retCode = 0;
    }
    return retCode;
}
