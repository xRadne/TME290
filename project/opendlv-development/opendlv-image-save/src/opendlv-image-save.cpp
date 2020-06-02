#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string.h>


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
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=1280 --height=720 --verbose" << std::endl;
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
            int counter = 0;
            int saveEvery = 10;
            int nameCounter = 0;
            std::string name;
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

              counter ++;
              if (counter == saveEvery) {
                name = std::to_string(nameCounter) + ".png";
                cvtColor(img, img, COLOR_RGBA2RGB);
                cv::imwrite(name, img);
                counter = 0;
                nameCounter++;
                if (VERBOSE) {
                  std::cout << "Saved " << name << std::endl;
                }
              }
            }
        }
        retCode = 0;
    }
    return retCode;
}
