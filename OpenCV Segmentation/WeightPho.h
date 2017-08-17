#pragma once
#include <opencv2/opencv.hpp>

using namespace cv;

class WeightPho
{
public:
	WeightPho(Point pixPoint, float avgVal);

	Point GetPixPoint();
	//Point GetOthPic();
	float AvgVal();

private:
	Point pixPoint;
	//Point othPic;
	float avgVal;
};