// Detection references: https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
// Kalman filter references: https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-using-kalman-filter/
//                           https://docs.opencv.org/master/de/d70/samples_2cpp_2kalman_8cpp-example.html

#include "cluon-complete.hpp"
#include "opendlv-standard-message-set.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/tracking.hpp"
#include <opencv2/dnn.hpp>

#include <cstdint>
#include <tuple>
#include <iostream>
#include <memory>
#include <mutex>
#include <string.h>

using namespace cv;
using namespace dnn;

// Constants
float confThreshold = 0.8f; // Confidence threshold
float nmsThreshold = 0.4f;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image

uint64_t lastTimeStepUs = 0;
uint64_t thisTimeStepUs;
int notFoundCount = 0;
bool found = false;

std::vector<std::string> classes;

// Remove the bounding boxes with low confidence using non-maxima suppression
std::tuple <uint16_t, uint16_t, uint16_t, uint16_t> postprocess(Mat& frame, const std::vector<Mat>& out);

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, int colour, Mat& frame);

// Get the names of the output layers
std::vector<std::string> getOutputsNames(const Net& net);

int32_t main(int32_t argc, char **argv) {
    int32_t retCode{1};
    auto commandlineArguments = cluon::getCommandlineArguments(argc, argv);
    if ( (0 == commandlineArguments.count("cid")) ||
         (0 == commandlineArguments.count("name")) ||
         (0 == commandlineArguments.count("width")) ||
         (0 == commandlineArguments.count("height")) ||
         (0 == commandlineArguments.count("weights")) ) {
        std::cerr << argv[0] << " attaches to a shared memory area containing an ARGB image." << std::endl;
        std::cerr << "Usage:   " << argv[0] << " --cid=<OD4 session> --name=<name of shared memory area> [--verbose]" << std::endl;
        std::cerr << "         --cid:    CID of the OD4Session to send and receive messages" << std::endl;
        std::cerr << "         --name:   name of the shared memory area to attach" << std::endl;
        std::cerr << "         --width:  width of the frame" << std::endl;
        std::cerr << "         --height: height of the frame" << std::endl;
        std::cerr << "         --weights: real or simulation" << std::endl;
        std::cerr << "Example: " << argv[0] << " --cid=112 --name=img.argb --width=640 --height=480 --verbose" << std::endl;
    }
    else {
      const std::string NAME{commandlineArguments["name"]};
      const uint32_t WIDTH{static_cast<uint32_t>(std::stoi(commandlineArguments["width"]))};
      const uint32_t HEIGHT{static_cast<uint32_t>(std::stoi(commandlineArguments["height"]))};
      uint16_t const CID = std::stoi(commandlineArguments["cid"]);
      const bool VERBOSE{commandlineArguments.count("verbose") != 0};
      
      int stateSize = 6;
      int measSize = 4;
      int contrSize = 0;
  
      unsigned int type = CV_32F;
      cv::KalmanFilter kf(stateSize, measSize, contrSize, type);
  
      cv::Mat state(stateSize, 1, type);
      cv::Mat measurement(measSize, 1, type);
      cv::setIdentity(kf.transitionMatrix);
      
      kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
      kf.measurementMatrix.at<float>(0) = 1.0f;
      kf.measurementMatrix.at<float>(7) = 1.0f;
      kf.measurementMatrix.at<float>(16) = 1.0f;
      kf.measurementMatrix.at<float>(23) = 1.0f;
      
      kf.processNoiseCov.at<float>(0) = 1e-2f;
      kf.processNoiseCov.at<float>(7) = 1e-2f;
      kf.processNoiseCov.at<float>(14) = 5.0f;
      kf.processNoiseCov.at<float>(21) = 5.0f;
      kf.processNoiseCov.at<float>(28) = 1e-2f;
      kf.processNoiseCov.at<float>(35) = 1e-2f;
  
      // Measures Noise Covariance Matrix R
      cv::setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1f));
      
      // Attach to the shared memory.
      std::unique_ptr<cluon::SharedMemory> sharedMemory{new cluon::SharedMemory{NAME}};
      if (sharedMemory && sharedMemory->valid()) {
        std::clog << argv[0] << ": Attached to shared memory '" << sharedMemory->name() << " (" << sharedMemory->size() << " bytes)." << std::endl;

        cluon::OD4Session od4{CID};
        
        // Load names of classes
        std::string classesFile = "/network_data/custom.names";
        std::ifstream ifs(classesFile.c_str());
        std::string line;
        while (getline(ifs, line)) classes.push_back(line);
        
        if (VERBOSE) {
            std::cout << "Class: " << classes[0] << std::endl;
        }
        // Give the configuration and weight files for the model
        std::string modelConfiguration = "/network_data/yolov3-tiny.cfg";
        std::string modelWeights = "/network_data/yolov3-tiny_last.weights";

        if (commandlineArguments["weights"] == "real") {
          modelWeights = "/network_data/yolov3-tiny_real.weights";
        }
        else if (commandlineArguments["weights"] == "simulation"){
          modelWeights = "/network_data/yolov3-tiny_simulation.weights";
        }
        else {
          std::cerr << " --weights: Wrong name: real or simulation" << std::endl;
          modelWeights = "none";
          return 1;
        }

        // Load the network
        Net net = readNetFromDarknet(modelConfiguration, modelWeights);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
        std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();

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

          cvtColor(img, img, COLOR_RGBA2RGB);
          
          ////////////////////////////////////////////////////////////////

          // Process frames

          // Create a 4D blob from a frame.
          Mat blob;
          blobFromImage(img, blob, 0.00392, Size(416, 416), true, false);

          //Sets the input to the network
          net.setInput(blob);

          // Runs the forward pass to get output of the output layers
          std::vector<Mat> outs;
          net.forward(outs, outNames);
          
          // Remove the bounding boxes with low confidence
          auto [centre_x_msg, centre_y_msg, width_msg, height_msg] = postprocess(img, outs);
  
          if (VERBOSE) {
            std::cout << "Before Filter" << std::endl;
            std::cout << centre_x_msg << " | " << centre_y_msg << " | " << width_msg << " | " << height_msg << std::endl;
          }
          
          float pred_x = 0;
          float pred_y = 0;
          float pred_w = 0;
          float pred_h = 0;
          
          // get time difference
          thisTimeStepUs = cluon::time::toMicroseconds(cluon::time::now());
          float dT = static_cast<float>((thisTimeStepUs - lastTimeStepUs) * 0.000001);
          lastTimeStepUs = thisTimeStepUs;
          
          // Kalman filter
          if (found)
          {
            // set A matrix
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;
            
            state = kf.predict();
    
            cv::Rect predRect;
            pred_x = state.at<float>(0);
            pred_y = state.at<float>(1);
            pred_w = state.at<float>(4);
            pred_h = state.at<float>(5);
          }
          
          // Update filter
          if (width_msg == 0)
          {
            notFoundCount++;
            found = notFoundCount < 100; // mark as lost after 100 frames
          }
          else
          {
            notFoundCount = 0;
  
            measurement.at<float>(0) = centre_x_msg;
            measurement.at<float>(1) = centre_y_msg;
            measurement.at<float>(2) = width_msg;
            measurement.at<float>(3) = height_msg;
    
            if (!found) // Detection previously lost
            {
              kf.errorCovPre.at<float>(0) = 1;
              kf.errorCovPre.at<float>(7) = 1;
              kf.errorCovPre.at<float>(14) = 1;
              kf.errorCovPre.at<float>(21) = 1;
              kf.errorCovPre.at<float>(28) = 1;
              kf.errorCovPre.at<float>(35) = 1;
      
              // state same as measurement
              state.at<float>(0) = measurement.at<float>(0);
              state.at<float>(1) = measurement.at<float>(1);
              state.at<float>(2) = 0;
              state.at<float>(3) = 0;
              state.at<float>(4) = measurement.at<float>(2);
              state.at<float>(5) = measurement.at<float>(3);
      
              kf.statePost = state;
      
              found = true;
            }
            else {
              kf.correct(measurement); // Kalman Correction
            }
          }
          
          centre_x_msg = static_cast<uint16_t>(pred_x);
          centre_y_msg = static_cast<uint16_t>(pred_y);
          width_msg = static_cast<uint16_t>(pred_w);
          height_msg = static_cast<uint16_t>(pred_h);
  
          if (VERBOSE) {
            std::cout << "After Filter" << std::endl;
            std::cout << centre_x_msg << " | " << centre_y_msg << " | " << width_msg << " | " << height_msg << std::endl;
            drawPred(0, 2.0f, centre_x_msg - width_msg/2, centre_y_msg - height_msg/2, centre_x_msg + width_msg/2, centre_y_msg + height_msg/2, 0, img);
          }
          
          // Put efficiency information. The function getPerfProfile returns the
          // overall time for inference(t) and the timings for each of the layers(in layersTimes)
          std::vector<double> layersTimes;
          double freq = getTickFrequency() / 1000;
          double t = net.getPerfProfile(layersTimes) / freq;
          std::string label = format("Inference time for a frame : %.2f ms", t);
          putText(img, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

          // Write the frame with the detection boxes
          Mat detectedFrame;
          img.convertTo(detectedFrame, CV_8U);


          ////////////////////////////////////////////////////////////////
          // Send the messages with cone positions:
          
          int32_t centre_x_msg_trans = static_cast<int32_t>(centre_x_msg) - WIDTH/2;
          centre_y_msg = HEIGHT - centre_y_msg;
          opendlv::logic::perception::Kiwi kiwiMsg;
          kiwiMsg.kiwiId(1);
          kiwiMsg.centreI(centre_x_msg_trans);
          kiwiMsg.centreJ(centre_y_msg);
          kiwiMsg.width(width_msg);
          kiwiMsg.height(height_msg);

          od4.send(kiwiMsg);

          if (VERBOSE) {
              std::cout << "Message send" << std::endl;
              std::cout << "Centre X: " << centre_x_msg_trans << std::endl;
              std::cout << "Centre Y: " << centre_y_msg << std::endl;
              std::cout << "Width: " << width_msg << std::endl;
              std::cout << "Height: " << height_msg << std::endl;
          }

          ////////////////////////////////////////////////////////////////
          // Display image.
          if (VERBOSE) {
              imshow("Kiwi detection", img);
              waitKey(1);
          }
        }
      }
      retCode = 0;
    }
    return retCode;
}

std::vector<std::string> getOutputsNames(const Net& net)
{
    // ----------------------------------
    // Get the names of the output layers
    // ----------------------------------
    static std::vector<std::string> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        std::vector<int> outLayers = net.getUnconnectedOutLayers();

        //get the names of all the layers in the network
        std::vector<std::string> layersNames = net.getLayerNames();

        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
            names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}


std::tuple <uint16_t, uint16_t, uint16_t, uint16_t> postprocess(Mat& frame, const std::vector<Mat>& outs)
{
    // ----------------------------------
    // Remove the bounding boxes with low confidence using non-maxima suppression
    // ----------------------------------
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<Rect> boxes;
    float max_confidence = 0.0f;
    uint16_t centre_x_fun = 0;
    uint16_t centre_y_fun = 0;
    uint16_t width_fun = 0;
    uint16_t height_fun = 0;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }

    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    std::vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, 255, frame);
        if (confidences[idx] > max_confidence) {
            max_confidence = confidences[idx];
            centre_x_fun = box.x + box.width/2;
            centre_y_fun = box.y + box.height/2;
            width_fun = box.width;
            height_fun = box.height;
        }
    }
    //static_cast<uint16_t>(centre_y)
    return {centre_x_fun, centre_y_fun, width_fun, height_fun};
}


void drawPred(int classId, float conf, int left, int top, int right, int bottom, int colour, Mat& frame)
{
    // ----------------------------------
    // Draw the predicted bounding box
    // Draw a rectangle displaying the bounding box
    // ----------------------------------
    rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, colour));

    //Get the label for the class name and its confidence
    std::string label = format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }

    //Display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);
    putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255,255,255));
}