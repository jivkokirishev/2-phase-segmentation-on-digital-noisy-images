// OpenCVTest.cpp : Defines the entry point for the console application.
//
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iomanip>
#include <algorithm>
#include <vector>
#include <numeric>
#include "WeightPho.h"
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <opencv2/core/eigen.hpp>

using namespace cv;

using namespace std;

auto labVector = vector<Point>{ Point(65, 53), Point(28, 22), Point(92, 46), Point(96, 12), Point(28, 30), Point(95, 81) };
//vector<Point>{ Point(65, 53), Point(28, 22), Point(92, 46), Point(96, 12), Point(28, 30), Point(95, 81) };
//vector<Point>{ Point(11, 36), Point(60, 15), Point(93, 38), Point(15, 14), Point(112, 11), Point(16, 58) };
//vector<Point>{ Point(25, 40), Point(78, 35), Point(60, 78), Point(60, 61), Point(109, 38), Point(46, 30) };
//vector<Point>{ Point(17, 17), Point(66, 41), Point(30, 78), Point(42, 3), Point(7, 66), Point(87, 77) };
//vector<Point>{ Point(7, 7), Point(34, 87), Point(87, 61), Point(33, 47), Point(86, 21), Point(47, 87) };
//vector<Point>{ Point(27, 29), Point(83, 24), Point(133, 24), Point(10, 10), Point(52, 29), Point(113, 44) };
//vector<Point>{ Point(34, 75), Point(75, 74), Point(57, 23), Point(105, 97), Point(7, 19), Point(91, 61) };

bool Compare(WeightPho i, WeightPho j)
{
	return i.AvgVal() > j.AvgVal();
}

float averaging(int x, int y, Mat *mat)
{
	int count = 4;
	float weight = ((int)mat->at<uchar>(Point(x, y)) / 255.0) * 4;
	if (x + 1 < mat->cols)
	{
		weight += ((int)mat->at<uchar>(Point(x + 1, y))) / 255.0;
		count++;
	}

	if (x - 1 >= 0)
	{
		weight += (int)mat->at<uchar>(Point(x - 1, y)) / 255.0;
		count++;
	}

	if (y + 1 < mat->rows)
	{
		auto newW = (int)mat->at<uchar>(Point(x, y + 1)) / 255.0;
		weight += newW;
		count++;
	}

	if (y - 1 >= 0)
	{
		weight += (int)mat->at<uchar>(Point(x, y - 1)) / 255.0;
		count++;
	}

	return weight / count;
}

SparseMat wPho(Mat *mat)
{
	int sizes[] = { mat->cols*mat->rows, mat->cols*mat->rows };
	SparseMat* wMatrix = new SparseMat(2, sizes, CV_32F);

	vector<WeightPho> picAvg = vector<WeightPho>();

	for (size_t y = 0; y < mat->rows; y++)
	{
		for (size_t x = 0; x < mat->cols; x++)
		{
			for (size_t i = 1; i <= 2; i++)
			{
				if ((int)(x + i) < mat->cols)
				{
					vector<Point> pts{ Point(x,y), Point((int)(x + i), y) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging((int)(x + i), y, mat), 2) );
					picAvg.push_back(WeightPho( Point((int)(x + i), y), avg));
				}

				if ((int)(x - i) >= 0)
				{
					vector<Point> pts{ Point(x,y), Point((int)(x - i), y) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging((int)(x - i), y, mat), 2) );
					picAvg.push_back(WeightPho(Point((int)(x - i), y), avg));
				}

				if ((int)(y + i) < mat->rows)
				{
					vector<Point> pts{ Point(x,y), Point(x,(int)(y + i)) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging(x, (int)(y + i), mat), 2) );
					picAvg.push_back(WeightPho(Point(x, (int)(y + i)), avg));
				}

				if ((int)(y - i) >= 0)
				{
					vector<Point> pts{ Point(x,y), Point(x,(int)(y - i)) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging(x, (int)(y - i), mat), 2) - pow(norm(pts, NORM_L2SQR), 2));
					picAvg.push_back(WeightPho(Point(x, (int)(y - i)), avg));
				}

				if ((int)(y + i) < mat->rows && (int)(x + i) < mat->cols)
				{
					vector<Point> pts{ Point(x,y), Point(x + i,(int)(y + i)) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging(x + i, (int)(y + i), mat), 2) - pow(norm(pts, NORM_L2SQR), 2));
					picAvg.push_back(WeightPho(Point(x + i, (int)(y + i)), avg));
				}

				if ((int)(y - i) >= 0 && (int)(x - i) >= 0)
				{
					vector<Point> pts{ Point(x,y), Point((int)(x - i), (int)(y - i)) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging((int)(x - i), (int)(y - i), mat), 2) - pow(norm(pts, NORM_L2SQR), 2));
					picAvg.push_back(WeightPho(Point((int)(x - i), (int)(y - i)), avg));
				}

				if ((int)(y - i) >= 0 && (int)(x + i) < mat->cols)
				{
					vector<Point> pts{ Point(x,y), Point((int)(x + i), (int)(y - i)) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging((int)(x + i), (int)(y - i), mat), 2) - pow(norm(pts, NORM_L2SQR), 2));
					picAvg.push_back(WeightPho(Point((int)(x + i), (int)(y - i)), avg));
				}

				if ((int)(y + i) < mat->rows && (int)(x - i) >= 0)
				{
					vector<Point> pts{ Point(x,y), Point((int)(x - i), (int)(y + i)) };
					float avg = exp(-pow(averaging(x, y, mat) - averaging((int)(x - i), (int)(y + i), mat), 2) - pow(norm(pts, NORM_L2SQR), 2));
					picAvg.push_back(WeightPho(Point((int)(x - i), (int)(y + i)), avg));
				}
			}

			sort(picAvg.begin(), picAvg.end(), Compare);
			float sum = 0.0f;
			for (size_t i = 0; i < 4; i++)
			{
				sum += picAvg[i].AvgVal();
			}

			for (size_t i = 0; i < 4; i++)
			{
				wMatrix->ref<float>(y * mat->cols + x, picAvg[i].GetPixPoint().y * mat->cols + picAvg[i].GetPixPoint().x) = picAvg[i].AvgVal() / sum;
			}
			picAvg.clear();
		}
	}

	return *wMatrix;
}

SparseMat wLab(vector<Point> lPoints, Mat *mat)
{
	int sizes[] = { mat->cols*mat->rows, mat->cols*mat->rows };
	SparseMat* wMatrix = new SparseMat(2, sizes, CV_32F);


	vector<WeightPho> labelWeights = vector<WeightPho>();
	for (vector<Point>::iterator it = lPoints.begin(); it != lPoints.end(); ++it)
	{
		labelWeights.push_back(WeightPho(*it, averaging(it->x, it->y, mat)));
	}

	vector<WeightPho> weights = vector<WeightPho>();
	for (size_t y = 0; y < mat->rows; y++)
	{
		for (size_t x = 0; x < mat->cols; x++)
		{
			float av2 = averaging(x, y, mat);
			int newY = y * mat->cols + x;

			for (vector<WeightPho>::iterator it = labelWeights.begin(); it != labelWeights.end(); ++it)
			{
				int newX = it->GetPixPoint().y * mat->cols + it->GetPixPoint().x;
				float e = exp(-pow(it->AvgVal() - av2, 2));
				weights.push_back(WeightPho(Point(newX, newY), e));
			}

			float sum = 0.0f;
			for (auto item : weights)
			{
				sum += item.AvgVal();
			}
			for (vector<WeightPho>::iterator it = weights.begin(); it != weights.end(); ++it)
			{
				wMatrix->ref<float>(it->GetPixPoint().y, it->GetPixPoint().x) = it->AvgVal() / sum;
			}
			weights.clear();
		}
	}



	// I am NOT sure if this has to be here 
	//for (vector<Point>::iterator it = lPoints.begin(); it != lPoints.end(); ++it)
	//{
	//	int newX = it->y * mat->cols + it->x;
	//	for (vector<Point>::iterator it2 = lPoints.begin(); it2 != lPoints.end(); ++it2)
	//	{
	//		int newY = it2->y * mat->cols + it2->x;
	//		if (newY != newX)
	//		{
	//			wMatrix->ref<float>(newX, newY) = 0.0f;
	//		}
	//		else
	//		{
	//			wMatrix->ref<float>(newY, newX) = 1.0f;
	//		}
	//	}
	//}

	return *wMatrix;
}

SparseMat wGeo(Mat *mat)
{
	int sizes[] = { mat->cols*mat->rows, mat->cols*mat->rows };
	SparseMat* wMatrix = new SparseMat(2, sizes, CV_32F);

	for (int y = 0; y < mat->rows; y++)
	{
		for (int x = 0; x < mat->cols; x++)
		{
			if (x + 1 < mat->cols)
			{
				wMatrix->ref<float>(y * mat->cols + x, y * mat->cols + x + 1) = 1.0f / 4.0f;
			}

			if (x - 1 >= 0)
			{
				wMatrix->ref<float>(y * mat->cols + x, y * mat->cols + x - 1) = 1.0f / 4.0f;
			}

			if (y + 1 < mat->rows)
			{
				wMatrix->ref<float>(y * mat->cols + x, (y + 1) * mat->cols + x) = 1.0f / 4.0f;
			}

			if (y - 1 >= 0)
			{
				wMatrix->ref<float>(y * mat->cols + x, (y - 1) * mat->cols + x) = 1.0f / 4.0f;
			}
		}
	}

	return *wMatrix;
}

SparseMat fullWeight(Mat *mat, float phoParam, float labParam)
{
	//TODO: to remove those hardcoded points and to automatically create labeled and unlabeled pixels
	/*mat->at<uchar>(Point(65, 54)) = 255;
	mat->at<uchar>(Point(24, 21)) = 255;
	mat->at<uchar>(Point(94, 50)) = 255;
	mat->at<uchar>(Point(30, 30)) = 0;
	mat->at<uchar>(Point(25, 63)) = 0;
	mat->at<uchar>(Point(96, 8)) = 0;*/
	//vector<Point> {Point(65, 54), Point(24, 21), Point(94, 50), Point(30, 30), Point(25, 63), Point(96, 8)}
	/*mat->at<uchar>(Point(34, 75)) = 0;
	mat->at<uchar>(Point(75, 74)) = 0;
	mat->at<uchar>(Point(57, 23)) = 0;
	mat->at<uchar>(Point(105, 97)) = 255;
	mat->at<uchar>(Point(7, 19)) = 255;
	mat->at<uchar>(Point(91, 61)) = 255;*/
	/*mat->at<uchar>(Point(25, 40)) = 255;
	mat->at<uchar>(Point(78, 35)) = 255;
	mat->at<uchar>(Point(60, 78)) = 255;
	mat->at<uchar>(Point(60, 61)) = 0;
	mat->at<uchar>(Point(109, 38)) = 0;
	mat->at<uchar>(Point(46, 30)) = 0;*/
	/*mat->at<uchar>(Point(17, 17)) = 255;
	mat->at<uchar>(Point(66, 41)) = 255;
	mat->at<uchar>(Point(30, 78)) = 255;
	mat->at<uchar>(Point(42, 3)) = 0;
	mat->at<uchar>(Point(7, 66)) = 0;
	mat->at<uchar>(Point(87, 77)) = 0;*/
	/*mat->at<uchar>(Point(11, 36)) = 255;
	mat->at<uchar>(Point(60, 15)) = 255;
	mat->at<uchar>(Point(93, 38)) = 255;
	mat->at<uchar>(Point(15, 14)) = 0;
	mat->at<uchar>(Point(112, 11)) = 0;
	mat->at<uchar>(Point(16, 58)) = 0;*/
	mat->at<uchar>(Point(65, 53)) = 255;
	mat->at<uchar>(Point(28, 22)) = 255;
	mat->at<uchar>(Point(92, 46)) = 255;
	mat->at<uchar>(Point(96, 12)) = 0;
	mat->at<uchar>(Point(28, 30)) = 0;
	mat->at<uchar>(Point(95, 81)) = 0;


	namedWindow("4x4", WINDOW_FREERATIO);
	imshow("4x4", *mat);

	SparseMat geoWeight = wGeo(mat);
	SparseMat phoWeight = wPho(mat);
	SparseMat labWeight = wLab(labVector, mat);

	Mat pho;
	phoWeight.convertTo(pho, -1);

	for (size_t y = 0; y < pho.rows; y++)
	{
		for (size_t x = 0; x < pho.cols; x++)
		{
			if (x != y)
			{
				auto val1 = pho.at<float>(x, y);
				auto val2 = pho.at<float>(y, x);
				if (val1 > val2)
				{
					pho.at<float>(x, y) = val2;
					pho.at<float>(y, x) = val2;
				}
				else if (val1 < val2)
				{
					pho.at<float>(x, y) = val1;
					pho.at<float>(y, x) = val1;
				}
			}
		}
	}

	Mat geo, lab;
	geoWeight.convertTo(geo, -1);
	labWeight.convertTo(lab, -1);

	Mat wPrim = (1.0 / (1 + phoParam))*geo + (phoParam / (1 + phoParam))*pho;
	Mat matrix = (1 / (1 + labParam))*wPrim + (labParam / (1 + labParam))*lab;

	SparseMat* wMatrix = new SparseMat(matrix);

	return *wMatrix;
}

SparseMat diagonal(Mat* mat)
{
	*mat = -1 * *mat;

	for (size_t i = 0; i < mat->rows; i++)
	{
		vector<float> row = vector<float>();
		mat->row(i).copyTo(row);
		float val = fabs(std::accumulate(row.begin(), row.end(), 0.0f));
		mat->at<float>(i, i) = val;
	}

	SparseMat* dMatrix = new SparseMat(*mat);

	return *dMatrix;
}

void wLLAndwLU(SparseMat* wMatrix, Mat* img, vector<Point> lPoints)
{
	for (vector<Point>::iterator it = lPoints.begin(); it != lPoints.end(); ++it)
	{
		int newX = it->y * img->cols + it->x;
		for (vector<Point>::iterator it2 = lPoints.begin(); it2 != lPoints.end(); ++it2)
		{
			int newY = it2->y * img->cols + it2->x;
			if (newY != newX)
			{
				wMatrix->ref<float>(newY, newX) = 0.0f;
			}
			else
			{
				wMatrix->ref<float>(newY, newX) = 1.0f;
			}
		}
	}

	for (size_t y = 0; y < img->rows; y++)
	{
		for (size_t x = 0; x < img->cols; x++)
		{
			int newY = y * img->cols + x;

			for (vector<Point>::iterator it = lPoints.begin(); it != lPoints.end(); ++it)
			{
				int newX = it->y * img->cols + it->x;
				if (newX != newY)
				{
					wMatrix->ref<float>(newX, newY) = 0.0f;
				}
			}
		}
	}
}

vector<float> computeQ(SparseMat* wMatrix, Mat* img, vector<Point> lPoints)
{
	vector<float> q = vector<float>();

	for (size_t y = 0; y < img->rows * img->cols; y++)
	{
		float sum = 0;
		for (vector<Point>::iterator it = lPoints.begin(); it != lPoints.end(); ++it)
		{
			int newX = it->y * img->cols + it->x;
			sum += wMatrix->ref<float>(y, newX) * ((int)img->at<uchar>(*it) / 255.0f);
		}
		q.push_back(sum);
		sum = 0;
	}

	return q;
}

void wUL(SparseMat* wMatrix, Mat* img, vector<Point> lPoints)
{
	for (size_t y = 0; y < img->rows; y++)
	{
		for (size_t x = 0; x < img->cols; x++)
		{
			int newY = y * img->cols + x;

			for (vector<Point>::iterator it = lPoints.begin(); it != lPoints.end(); ++it)
			{
				int newX = it->y * img->cols + it->x;
				if (newX != newY)
				{
					wMatrix->ref<float>(newY, newX) = 0.0f;
				}
			}
		}
	}
}

int main()
{
	namedWindow("Display", WINDOW_FREERATIO);
	namedWindow("DisplayGray", WINDOW_FREERATIO);
	namedWindow("DisplayGraussian", WINDOW_FREERATIO);
	Mat image = imread("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCVTest\\x64\\Debug\\cables.jpg");
	//Mat image = Mat(96, 96, CV_8UC3, Scalar(255,255,255));
	/*for (size_t y = 0; y < image.rows; y++)
	{
		for (size_t x = 0; x < image.cols * 3; x+=3)
		{
			if (((x > 24 * 3 && y > 24) || (x <= 24 * 3 && y <= 24))||((x > 48 * 3 && y > 48) || (x <= 72 * 3 && y <= 72))|| ((x > 72 * 3 && y > 72) || (x <= 72 * 3 && y <= 72)))
			{
				image.at<uchar>(y, x) = 255;
				image.at<uchar>(y, x + 1) = 255;
				image.at<uchar>(y, x + 2) = 255;
			}
			else
			{
				image.at<uchar>(y, x) = 0;
				image.at<uchar>(y, x + 1) = 0;
				image.at<uchar>(y, x + 2) = 0;
			}
		}
	}*/
	imshow("Display", image);
	Mat gray = Mat();
	cvtColor(image, gray, CV_BGR2GRAY);

	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\cables_original.jpg", gray);

	Mat gaussian_noise = gray.clone();
	randn(gaussian_noise, 128, 100);

	vector<Point> pts{ Point(0,0), Point(1,1) };

	//NORM_INF
	cout << norm(pts, NORM_L2SQR) << endl;

	Mat gr_noise = gaussian_noise + gray - 128;  //  gray;

	for (size_t i = 0; i < gr_noise.cols * gr_noise.rows; i++)
	{
		if ((int)gr_noise.at<uchar>(i) > 255)
		{
			gr_noise.at<uchar>(i) = 255;
		}
		else if ((int)gr_noise.at<uchar>(i) < 0)
		{
			gr_noise.at<uchar>(i) = 0;
		}
	}

	imshow("DisplayGray", gray);
	imshow("DisplayGraussian", gr_noise);
	//imshow("DisplayGray", gray);
	//imwrite("c:\\opencv\\fdsa.png", gray);

	int histSize = 257;

	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;

	Mat hist;

	/// Compute the histograms:
	calcHist(&gr_noise, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);


	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\chessboard noise hist.jpg", histImage);

	//gr_noise = gr_noise.rowRange(Range(0, 12)).colRange(Range(0, 14));
	//gr_noise = gr_noise.rowRange(Range(0, 60)).colRange(Range(0, 60));
	//gr_noise = gray;
	const int n = gr_noise.cols * gr_noise.rows;

	cout << n;
	int sizes[] = { n, n };
	SparseMat* wMatrix = new SparseMat(2, sizes, CV_32F);
	cout << endl << endl << endl;
	//cout << endl << wMatrix->ref<float>(1, 2) << endl;
	//wMatrix->ref<float>(1, 2) = 1.0f / 4.0f;
	cout << endl << endl << endl;



	//Mat *blabla = new Mat();
	//wMatrix->convertTo(*blabla, CV_32F);
	cout << endl << endl << endl << endl;
	/*gr_noise.at<uchar>(Point(0, 0)) = 0;
	gr_noise.at<uchar>(Point(1, 1)) = 255;
	gr_noise.at<uchar>(Point(2, 2)) = 0;
	gr_noise.at<uchar>(Point(3, 2)) = 0;
	gr_noise.at<uchar>(Point(1, 5)) = 255;
	gr_noise.at<uchar>(Point(2, 6)) = 255;*/
	SparseMat secMat; // = wLab(vector<Point> {Point(4, 7), Point(9, 5), Point(6, 4), Point(9, 2), Point(7, 5), Point(0, 2)}, &gr_noise);
	secMat = fullWeight(&gr_noise, 5, 1.0f / 12.0f);

	Mat converted;
	secMat.convertTo(converted, -1);

	wLLAndwLU(&secMat, &gr_noise, labVector);
	auto q = computeQ(&secMat, &gr_noise, labVector);
	//for (auto item : q)
	//{
	//	cout << item<<endl;
	//}
	//
	Mat converted2;

	secMat.convertTo(converted2, -1);

	namedWindow("convert123", WINDOW_FREERATIO);
	imshow("convert123", converted2);

	SparseMat diag = diagonal(&converted2);
	wUL(&diag, &gr_noise, labVector);

	diag.convertTo(converted2, -1);

	cout << converted2.rows << "  " << converted2.cols<<endl;
	cout << q.size()<<endl;

	/*for (size_t x = 0; x < 45; x++)
	{
		for (size_t y = 0; y < 45; y++)
		{
			cout << diag.ref<float>(x, y) << " ";
		}
		cout << endl;
	}*/

	vector<float> v;
	Mat m = Mat(q.size(), 1, CV_32F);
	Eigen::VectorXf x(q.size());
	for (size_t i = 0; i < q.size(); i++)
	{
		x[i] = q[i];
	}

	Mat m2 = Mat(gr_noise.rows, gr_noise.cols, CV_32F);

	//Mat m2;

	//solve(converted2, m, m2, DECOMP_LU);

	const int ncr = converted2.cols;
	Eigen::MatrixXf matr(ncr, ncr);
	cv2eigen(converted2, matr);
	Eigen::SparseMatrix<float> sMatrix = matr.sparseView();

	Eigen::VectorXf b(q.size());
	Eigen::SparseMatrix<double> A(n, n);
	// fill A and b
	Eigen::ConjugateGradient<Eigen::SparseMatrix<float>, Eigen::Lower | Eigen::Upper> cg;
	cg.compute(sMatrix);
	b = cg.solve(x);


	for (size_t x = 0; x < 45; x++)
	{
		for (size_t y = 0; y < 45; y++)
		{
			cout << matr.col(x).row(y) << " ";
		}
		cout << endl;
	}

	//cout << m2 << endl;

	//m2.copyTo(v);

	/*for (auto item: v)
	{
		cout << item << endl;
	}*/

	Mat m3 = Mat(gr_noise.rows, gr_noise.cols, CV_32F, float(0));

	for (size_t i = 0; i < b.size(); i++)
	{
		auto p = Point(i % gr_noise.cols, i / gr_noise.cols);
 		m3.at<float>(p) = b[i];
	}

	Mat tres, tres2;
	Mat m4;
	m3.convertTo(m4, CV_8UC1);

	double  minVal, maxVal;
	minMaxLoc(m3, &minVal, &maxVal);  //find  minimum  and  maximum  intensities
	Mat  draw;
	m3.convertTo(m4, CV_8U, 255.0 / (maxVal - minVal), -minVal);



	MatND histogr;
	calcHist(&m4, 1, 0, Mat(), histogr, 1, &histSize, &histRange, true, false);

	// Show the calculated histogram in command window
	double total;
	total = m4.rows * m4.cols;
	for (int h = 0; h < histSize; h++)
	{
		float binVal = histogr.at<float>(h);
		cout << " " << binVal;
	}

	// Plot the histogram
	//int hist_w2 = 512; int hist_h2 = 400;
	//int bin_w2 = cvRound((double)hist_w2 / histSize);

	Mat histImage2(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));
	normalize(histogr, histogr, 0, histImage2.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage2, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(histogr.at<float>(i))),
			Scalar(255, 255, 255), 2, 8, 0);
	}

	namedWindow("Result", 1);
	imshow("Result", histImage2);
	Mat tres23;
	threshold(m4, tres, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	threshold(gr_noise, tres23, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\chessboard noise.jpg", gr_noise);
	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\chessboard treshold.jpg", tres);
	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\chessboard weight hist.jpg", histImage2);
	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\chessboard weight.jpg", m4);
	imwrite("C:\\Users\\Jivko\\Desktop\\Programi\\OpenCV Segmentation\\result images\\chessboard noise otsu.jpg", tres23);

	namedWindow("convert", WINDOW_FREERATIO);
	imshow("convert", m3);
	namedWindow("treshold", WINDOW_FREERATIO);
	imshow("treshold", tres);

	cvWaitKey(0);
	return 0;
}




/*int main()
{
	int size[] = { 2,2 };

	Mat m = Mat(2,2,CV_32F, float(0));
	m.at<float>(0, 1) = 57;
	SparseMat mat = SparseMat(m);
	mat.ref<float>(1, 6) = 43;
	mat.ref<float>(1, 9) = 43;
	mat.ref<float>(5, 3) = 43;
	mat.ref<float>(0, 0) = 43;
	mat.ref<float>(5, 6) = 43;
	mat.ref<float>(8, 6) = 43;
	mat.ref<float>(1, 4) = 43;

	SparseMatIterator_<float>
		it = mat.begin<float>(),
		itEnd = mat.end<float>();

	for (; it != itEnd; it++)
	{
		for (size_t i = 0; i < mat.dims(); i++)
		{
			cout << it.node()->idx[i]<< " ";
		}
		cout << *it << endl;
	}

	int a;
	cin >> a;
	cvWaitKey(0);
	return 0;
}*/

//for (int y = 0; y < gr_noise.rows; y++)
//{
//	for (int x = 0; x < gr_noise.cols; x++)
//	{
//		if (x + 1 < gr_noise.cols)
//		{
//			cout << "p1" << endl;
//			cout << "x = " << x << endl;
//			cout << "y = " << y << endl;
//			cout << "p2" << endl;
//			cout << "x = " << x + 1<< endl;
//			cout << "y = " << y << endl;
//			//cout << "before: " << wMatrix->value<float>(x * gr_noise.cols + y, (x + 1) * gr_noise.cols + y) << endl;
//			wMatrix->ref<float>(x * gr_noise.cols + y, (x + 1) * gr_noise.cols + y) = 1.0f / 4.0f;
//			//cout << "after: " << wMatrix->value<float>(x * gr_noise.cols + y, (x + 1) * gr_noise.cols + y) << endl;
//		}
//
//		if (x - 1 >= 0)
//		{
//			cout << "p1" << endl;
//			cout << "x = " << x << endl;
//			cout << "y = " << y << endl;
//			cout << "p2" << endl;
//			cout << "x = " << x - 1 << endl;
//			cout << "y = " << y << endl;
//			//cout << "before: " << wMatrix->ref<float>(x * gr_noise.cols + y, (x - 1) * gr_noise.cols + y) << endl;
//			wMatrix->ref<float>(x * gr_noise.cols + y, (x - 1) * gr_noise.cols + y) = 1.0f / 4.0f;
//			//cout << "after: " << wMatrix->ref<float>(x * gr_noise.cols + y, (x - 1) * gr_noise.cols + y) << endl;
//		}
//
//		if (y + 1 < gr_noise.rows)
//		{
//			cout << "p1" << endl;
//			cout << "x = " << x << endl;
//			cout << "y = " << y << endl;
//			cout << "p2" << endl;
//			cout << "x = " << x << endl;
//			cout << "y + 1 = " << y + 1 << endl;
//			//cout << "before: " << wMatrix->ref<float>(x * gr_noise.cols + y, x * gr_noise.cols + y + 1) << endl;
//			wMatrix->ref<float>(x * gr_noise.cols + y, x * gr_noise.cols + y + 1) = 1.0f / 4.0f;
//			//cout << "after: " << wMatrix->ref<float>(x * gr_noise.cols + y, x * gr_noise.cols + y + 1) << endl;
//		}
//
//		if (y - 1 >= 0)
//		{
//			cout << "p1" << endl;
//			cout << "x = " << x << endl;
//			cout << "y = " << y << endl;
//			cout << "p2" << endl;
//			cout << "x = " << x << endl;
//			cout << "y - 1 = " << y - 1 << endl;
//			//cout << "before: " << wMatrix->ref<float>(x * gr_noise.cols + y, x * gr_noise.cols + y - 1) << endl;
//			wMatrix->ref<float>(x * gr_noise.cols + y, x * gr_noise.cols + y - 1) = 1.0f / 4.0f;
//			//cout << "after: " << wMatrix->ref<float>(x * gr_noise.cols + y, x * gr_noise.cols + y - 1) << endl;
//		}
//	}
//}