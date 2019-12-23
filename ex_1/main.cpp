#include<stdio.h>

#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include  <opencv2\opencv.hpp>
#include "net.h"
#include "mat.h"
#include "benchmark.h"

#include "squeezenet_v1.1.id.h"
/*
	@brief 读取标签文件
	@param [input] strFileName 文件名
	@param [input] vecLabels 标签
*/
void read_labels(std::string strFileName,std::vector<std::string> &vecLabels)
{
	std::ifstream in(strFileName);

	if (in)
	{	
		std::string line;
		while (std::getline(in, line)) 
		{
			// std::cout << line << std::endl;
			vecLabels.push_back(line);
		}
	}
	else 
	{
		std::cout << "label file is not exit!!!" << std::endl;
	}
}
/*
	@brief squeezenet_v_1			预测单张图的类别
	@param [input] strImagePath		图片路径
*/
void forward_squeezenet_v_1(std::string strImagePath)
{
	// data
	std::string strLabelPath = "../model/synset_words.txt";
	std::vector<std::string> vecLabel;
	read_labels(strLabelPath, vecLabel);

	const float mean_vals[3] = { 104.f, 117.f, 123.f };
	cv::Mat matImage = cv::imread(strImagePath);
	// cv::resize(matImage, matImage, cv::Size(227, 227));
	if (matImage.empty()) 
	{
		printf("image is empty!!!\n");
	}
	
	const int nImageWidth = matImage.cols;
	const int nImageHeight = matImage.rows;

	// input and output
	ncnn::Mat matIn;
	ncnn::Mat matOut;
	// net
	ncnn::Net net;
	net.load_param_bin("../model/squeezenet_v1.1.param.bin");
	net.load_model("../model/squeezenet_v1.1.bin");
	
	const int nNetInputWidth = 227;
	const int nNetInputHeight = 227;

	// time
	double dStart = ncnn::get_current_time();

	// 判断图片大小是否和网络输入相同
	if (nNetInputWidth != nImageWidth || nNetInputHeight != nImageHeight)
	{
		matIn = ncnn::Mat::from_pixels_resize(matImage.data, ncnn::Mat::PIXEL_BGR, nImageWidth, nImageHeight, nNetInputWidth, nNetInputHeight);
	}
	else
	{
		matIn = ncnn::Mat::from_pixels(matImage.data, ncnn::Mat::PIXEL_BGR, nNetInputWidth, nNetInputHeight);
	}
	// 数据预处理
	matIn.substract_mean_normalize(mean_vals, 0);

	// forward
	ncnn::Extractor ex = net.create_extractor();
	ex.set_light_mode(true);
	ex.input(squeezenet_v1_1_param_id::BLOB_data, matIn);
	ex.extract(squeezenet_v1_1_param_id::BLOB_prob, matOut);

	printf("output_size: %d, %d, %d \n", matOut.w, matOut.h, matOut.c);
	
	// cls 1000 class
	std::vector<float> cls_scores;
	cls_scores.resize(matOut.w);
	for (int i = 0; i <matOut.w; i ++)
	{
		cls_scores[i] = matOut[i];
	}
	// return top class
	int top_class = 0;
	float max_score = 0.f;
	for (size_t i = 0; i<cls_scores.size(); i++)
	{
		float s = cls_scores[i];
		if (s > max_score)
		{
			top_class = i;
			max_score = s;
		}
	}
	double dEnd = ncnn::get_current_time();

	printf("%d  score: %f   spend time: %.2f ms\n", top_class, max_score, (dEnd - dStart));
	std::cout << vecLabel[top_class] << std::endl;
	cv::putText(matImage, vecLabel[top_class], cv::Point(5, 10), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " score:" + std::to_string(max_score), cv::Point(5, 20), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::putText(matImage, " time: " + std::to_string(dEnd - dStart) + "ms", cv::Point(5, 30), 1, 0.8, cv::Scalar(0, 0, 255), 1);
	cv::imwrite("..\\images\\result.jpg", matImage);
	//cv::imshow("result", matImage);
	//cv::waitKey(-1);
	net.clear();
	
	
}

int main()
{
	forward_squeezenet_v_1("..\\images\\grand.jpg");
	printf("hello ncnn");
	system("pause");
}