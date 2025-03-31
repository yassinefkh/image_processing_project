#ifndef IMAGEUTILS_HPP
#define IMAGEUTILS_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class ImageUtils {
public:
    static cv::Mat preprocessImage(const cv::Mat& image);
    static cv::Mat detectEdges(const cv::Mat& image);
    static std::vector<cv::Point> extractContourPoints(const cv::Mat& edges);
    static std::pair<cv::Point2d, cv::Point2d> computePCA(const std::vector<cv::Point>& points);
    static std::pair<cv::Point, cv::Point> computeLineEndpoints(const cv::Point2d& mean, const cv::Point2d& dir, int width, int height);
    static std::vector<double> extractDepthProfile(const cv::Mat& depthMap, const cv::Point2d& mean, const cv::Point2d& dir, std::vector<cv::Point>& profilePoints);
    static void exportProfile(const std::vector<double>& depthValues, const std::string& filename);
    static std::vector<int> detectTransitions(const std::vector<double>& signal);
    
    static std::vector<double> extractVerticalProfile(const cv::Mat& depthMap);
    static std::vector<double> extractRotatedProfile(const cv::Mat& depthMap, double angle);
    
};

#endif
