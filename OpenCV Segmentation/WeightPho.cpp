#include "WeightPho.h"

WeightPho::WeightPho(Point pixPoint, float avgVal)
{
	this->pixPoint = pixPoint;
	this->avgVal = avgVal;
}

Point WeightPho::GetPixPoint()
{
	return this->pixPoint;
}

float WeightPho::AvgVal()
{
	return this->avgVal;
}
