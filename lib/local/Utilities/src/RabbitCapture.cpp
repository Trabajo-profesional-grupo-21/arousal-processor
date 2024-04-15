///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////
#include "stdafx_ut.h"

#include "RabbitCapture.h"
#include "ImageManipulationHelpers.h"

#include <stdexcept>
#include <typeinfo> 

// #include <thread> // Para sleep
// #include <chrono> // Para sleep

using namespace Utilities;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl




RabbitCapture::RabbitCapture(){
	std::string queue_name = "arousal_frames";
	std::string output_queue_name = "processed";
	std::string connection_name = "rabbitmq";
	std::string exchange_name = "frames";

	latest_frame = cv::Mat();
	latest_gray_frame = cv::Mat();

	try {
		// connection = AmqpClient::Channel::Create(connection_name);
		connection = AmqpClient::Channel::Create("moose.rmq.cloudamqp.com", 5672, "zacfsxvy", "zfCu8hS9snVGmySGhtvIVeMi6uvYssih", "zacfsxvy");

		connection->DeclareExchange(exchange_name, AmqpClient::Channel::EXCHANGE_TYPE_FANOUT);

		input_queue_id = connection->DeclareQueue(queue_name, false, true, false, false); // TODO: REVISAR PARAMS
		output_queue = connection->DeclareQueue(output_queue_name, false, true, false, false); // TODO: REVISAR PARAMS

		connection->BindQueue(queue_name, exchange_name, "");

		consumer_tag = connection->BasicConsume(queue_name, "", true, true, false, 1);
	} catch (const std::exception& e) {
        std::cerr << "Error: Failed to connect to RabbitMQ server. " << e.what() << std::endl;
		throw std::runtime_error("Failed to connect to RabbitMQ server: " + std::string(e.what()));
    }
};

bool RabbitCapture::Open(std::vector<std::string>& arguments)
{

	// Consuming the input arguments
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	// Some default values
	std::string input_root = "";
	fx = -1; fy = -1; cx = -1; cy = -1;

	std::string separator = std::string(1, fs::path::preferred_separator);

	// First check if there is a root argument (so that videos and input directories could be defined more easily)
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0)
		{
			input_root = arguments[i + 1] + separator;
			i++;
		}
		if (arguments[i].compare("-inroot") == 0)
		{
			input_root = arguments[i + 1] + separator;
			i++;
		}
	}

	std::string input_directory;
	std::string bbox_directory;

	bool directory_found = false;
	has_bounding_boxes = false;

	std::vector<std::string> input_image_files;

	return true;
}

void RabbitCapture::SetCameraIntrinsics(float fx, float fy, float cx, float cy)
{
	// If optical centers are not defined just use center of image
	if (cx == -1)
	{
		this->cx = this->image_width / 2.0f;
		this->cy = this->image_height / 2.0f;
	}
	else
	{
		this->cx = cx;
		this->cy = cy;
	}
	// Use a rough guess-timate of focal length
	if (fx == -1)
	{
		this->fx = 500.0f * (this->image_width / 640.0f);
		this->fy = 500.0f * (this->image_height / 480.0f);

		this->fx = (this->fx + this->fy) / 2.0f;
		this->fy = this->fx;
	}
	else
	{
		this->fx = fx;
		this->fy = fy;
	}
}

void RabbitCapture::ProcessReply(std::string frame_id, json img_processed){
	current_replies[frame_id] = img_processed;
	std::cout << "frame id " << frame_id << "info " << img_processed << std::endl;
	std::cout << "Tamanio del current batch " << current_batch.size() << std::endl;
	
	if (current_batch.empty()){
		json output_json;
		output_json["user_id"] = current_user_id;
		output_json["batch_id"] = current_batch_id;
		output_json["replies"] = current_replies;
		output_json["origin"] = "arousal";

		current_replies.clear();

		std::string message = output_json.dump();
		connection->BasicPublish("", output_queue, AmqpClient::BasicMessage::Create(message));
	}
}

// Returns a read image in 3 channel RGB format, also prepares a grayscale frame if needed
Image RabbitCapture::GetNextImage()
{	
	Image current_image;
	if (current_batch.empty()){
		AmqpClient::Envelope::ptr_t envelope;

		connection->BasicConsumeMessage(consumer_tag, envelope);

		std::string message_body = envelope->Message()->Body();

		json j = json::parse(message_body);
		current_user_id = j["user_id"];
		current_batch_id = j["batch_id"];
		current_batch = j["batch"];
	}
	std::cout << "Tamb dict antes de sacar un farme " << current_batch.size() << std::endl;
	auto it = current_batch.begin();
	std::string frame_id = it->first; 
	std::cout << "frame que itero " << frame_id << std::endl;
	std::vector<uchar> image_data = it->second;     
    current_batch.erase(it->first);
	std::cout << "Tamb dict despues de sacar uno frame " << current_batch.size() << std::endl;


	// std::vector<uchar> image_data = current_batch.front();
	// current_batch.pop_front();

	latest_frame = cv::imdecode(image_data, cv::IMREAD_COLOR);

	image_height = latest_frame.size().height;
	image_width = latest_frame.size().width;

	// Reset the intrinsics for every image if they are not set globally
	float _fx = -1;
	float _fy = -1;
	float _cx = -1;
	float _cy = -1;
	SetCameraIntrinsics(_fx, _fy, _cx, _cy);

	// Set the grayscale frame
	ConvertToGrayscale_8bit(latest_frame, latest_gray_frame);
	current_image.frame_id = frame_id;
	current_image.image_data = latest_frame;
	return current_image;
}

// std::vector<cv::Rect_<float> > RabbitCapture::GetBoundingBoxes()
// {
// 	if (!bounding_boxes.empty())
// 	{
// 		return bounding_boxes[frame_num - 1];
// 	}
// 	else
// 	{
// 		return std::vector<cv::Rect_<float> >();
// 	}
// }

double RabbitCapture::GetProgress()
{
	return 20;
	// return (double)frame_num / (double)image_files.size();
}

cv::Mat_<uchar> RabbitCapture::GetGrayFrame()
{
	return latest_gray_frame;
}
