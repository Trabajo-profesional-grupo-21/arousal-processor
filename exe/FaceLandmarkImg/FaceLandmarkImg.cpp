///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltru�aitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltru�aitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltru�aitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltru�aitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
// FaceLandmarkImg.cpp : Defines the entry point for the console application for detecting landmarks in images.

// dlib
#include <dlib/image_processing/frontal_face_detector.h>

#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>
#include <GazeEstimation.h>

#include <Visualizer.h>
#include <VisualizationUtils.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <RabbitCapture.h>
#include <chrono>
#include <ctime>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif
#define TOP_INTENSITIES_AMOUNT 5


std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}
double calculate_arousal(std::vector<std::pair<std::string, double>> &au_intensities){
	// AU45: blink
	//{"ActionUnit":[{"AUName":"AU01","Intensity":0.6393677665876463},{"AUName":"AU02","Intensity":0.0},{"AUName":"AU04","Intensity":0.0},{"AUName":"AU05","Intensity":0.06933280513717067},{"AUName":"AU06","Intensity":1.9017657678319653},{"AUName":"AU07","Intensity":2.137407390615198},{"AUName":"AU09","Intensity":0.0},{"AUName":"AU10","Intensity":2.88920207800761},{"AUName":"AU12","Intensity":2.608026953424237},{"AUName":"AU14","Intensity":2.271780912358205},{"AUName":"AU15","Intensity":0.49019558448372147},{"AUName":"AU17","Intensity":0.0},{"AUName":"AU20","Intensity":2.1307561516269664},{"AUName":"AU23","Intensity":0.0},{"AUName":"AU25","Intensity":2.2882124683240868},{"AUName":"AU26","Intensity":0.44641159993709495},{"AUName":"AU45","Intensity":0.0}]}
	std::vector<double> intensities;
	for (size_t i = 0; i < au_intensities.size(); ++i) {
		intensities.push_back(au_intensities[i].second);
	}
	sort(intensities.begin(), intensities.end(), std::greater<double>());

	std::vector<double> top_intensities(intensities.begin(),intensities.begin() + TOP_INTENSITIES_AMOUNT);
	double sum = std::accumulate(top_intensities.begin(), top_intensities.end(), 0.00000);
	double mean = sum / top_intensities.size();
	return mean;
}


int main(int argc, char **argv)
{

	//Convert arguments to more convenient vector form
	std::vector<std::string> arguments = get_arguments(argc, argv);

	// no arguments: output usage
	// if (arguments.size() == 1)
	// {
	// 	std::cout << "For command line arguments see:" << std::endl;
	// 	std::cout << " https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments";
	// 	return 0;
	// }

	// Prepare for image reading
	Utilities::RabbitCapture rabbit_reader;
	// The sequence reader chooses what to open based on command line arguments provided
	if (!rabbit_reader.Open(arguments))
	{
		std::cout << "Could not open RabbitReader" << std::endl;
		return 1;
	}

	// Load the models if images found
	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// The modules that are being used for tracking
	std::cout << "Loading the model" << std::endl;
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		return 1;
	}

	std::cout << "Model loaded" << std::endl;

	Utilities::FpsTracker fps_tracker;

	// Load facial feature extractor and AU analyser (make sure it is static)
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
	face_analysis_params.OptimizeForImages();
	FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

	// If bounding boxes not provided, use a face detector
	cv::CascadeClassifier classifier(det_parameters.haar_face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
	LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn(det_parameters.mtcnn_face_detector_location);

	// If can't find MTCNN face detector, default to HOG one
	if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::MTCNN_DETECTOR && face_detector_mtcnn.empty())
	{
		std::cout << "INFO: defaulting to HOG-SVM face detector" << std::endl;
		det_parameters.curr_face_detector = LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR;
	}

	// A utility for visualizing the results
	// Utilities::Visualizer visualizer(arguments);

	cv::Mat rgb_image;

	Utilities::Image img_info = rabbit_reader.GetNextImage();
	std::string frame_id = img_info.frame_id;
	rgb_image = img_info.image_data;


	if (!face_model.eye_model)
	{
		std::cout << "WARNING: no eye model found" << std::endl;
	}

	if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
	{
		std::cout << "WARNING: no Action Unit models found" << std::endl;
	}

	std::cout << "Starting tracking" << std::endl;
	while (!rgb_image.empty())
	{	

		fps_tracker.AddFrame();
		// auto now = std::chrono::system_clock::now();
		// std::time_t now_time = std::chrono::system_clock::to_time_t(now);
		// std::cout << "Timestamp actual: " << std::ctime(&now_time);


		Utilities::RecorderOpenFaceParameters recording_params(arguments, false, false,
			rabbit_reader.fx, rabbit_reader.fy, rabbit_reader.cx, rabbit_reader.cy);

		if (!face_model.eye_model)
		{
			recording_params.setOutputGaze(false);
		}

		// Utilities::RecorderOpenFace open_face_rec(rabbit_reader.name, recording_params, arguments);

		// visualizer.SetImage(rgb_image, rabbit_reader.fx, rabbit_reader.fy, rabbit_reader.cx, rabbit_reader.cy);

		// Making sure the image is in uchar grayscale (some face detectors use RGB, landmark detector uses grayscale)
		cv::Mat_<uchar> grayscale_image = rabbit_reader.GetGrayFrame();

		// Detect faces in an image
		std::vector<cv::Rect_<float> > face_detections;

		if (rabbit_reader.has_bounding_boxes)
		{
			// face_detections = rabbit_reader.GetBoundingBoxes();
			return 0;
		}
		else
		{
			if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HOG_SVM_DETECTOR)
			{
				std::vector<float> confidences;
				LandmarkDetector::DetectFacesHOG(face_detections, grayscale_image, face_detector_hog, confidences);
			}
			else if (det_parameters.curr_face_detector == LandmarkDetector::FaceModelParameters::HAAR_DETECTOR)
			{
				LandmarkDetector::DetectFaces(face_detections, grayscale_image, classifier);
			}
			else
			{
				std::vector<float> confidences;
				LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, face_detector_mtcnn, confidences);
			}
		}

		// Detect landmarks around detected faces
		int face_det = 0;
		// perform landmark detection for every face detected
		for (size_t face = 0; face < face_detections.size(); ++face)
		{

			// if there are multiple detections go through them
			bool success = LandmarkDetector::DetectLandmarksInImage(rgb_image, face_detections[face], face_model, det_parameters, grayscale_image);

			// Estimate head pose and eye gaze				
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, rabbit_reader.fx, rabbit_reader.fy, rabbit_reader.cx, rabbit_reader.cy);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gaze_direction0(0, 0, -1);
			cv::Point3f gaze_direction1(0, 0, -1);
			cv::Vec2f gaze_angle(0, 0);

			if (face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gaze_direction0, rabbit_reader.fx, rabbit_reader.fy, rabbit_reader.cx, rabbit_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gaze_direction1, rabbit_reader.fx, rabbit_reader.fy, rabbit_reader.cx, rabbit_reader.cy, false);
				gaze_angle = GazeAnalysis::GetGazeAngle(gaze_direction0, gaze_direction1);
			}

			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor; int num_hog_rows = 0, num_hog_cols = 0;

			// Perform AU detection and HOG feature extraction, as this can be expensive only compute it if needed by output or visualization
			// if (recording_params.outputAlignedFaces() || recording_params.outputHOG() || recording_params.outputAUs() || visualizer.vis_align || visualizer.vis_hog)
			if (recording_params.outputAlignedFaces() || recording_params.outputHOG() || recording_params.outputAUs())
			{
				face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);
				face_analyser.GetLatestAlignedFace(sim_warped_img);
				face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);
			}


			std::vector<std::pair<std::string, double>> au_intensities = face_analyser.GetCurrentAUsReg();
			std::vector<std::pair<std::string, double>> au_occurences = face_analyser.GetCurrentAUsClass();
			
			
			json output_json;
			json& action_units = output_json["ActionUnit"];

			for (size_t i = 0; i < au_intensities.size(); ++i) {
				json au_json;
				au_json["AUName"] = au_intensities[i].first;
				au_json["Intensity"] = au_intensities[i].second;
				action_units.push_back(au_json);
			}
			
			// calcular el arousal
			double arousal = calculate_arousal(au_intensities);
			// std::cout << "arousal: " << arousal << std::endl;
			output_json["arousal"] = arousal;
			
			
			// std::cout << output_json.dump() << std::endl;
				
			rabbit_reader.ProcessReply(frame_id, output_json);

			img_info = rabbit_reader.GetNextImage();
			frame_id = img_info.frame_id;
			rgb_image = img_info.image_data;
			std::cout << "FPS: " << fps_tracker.GetFPS() << std::endl;
		}
	}
	return 0;
}