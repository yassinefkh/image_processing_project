#ifndef IMAGEUTILS_HPP
#define IMAGEUTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>

class ImageUtils {
public:
    // charger une image depuis un fichier
    static cv::Mat loadImage(const std::string& path);
    // afficher une image dans une fenêtre
    static void displayImage(const std::string& windowName, const cv::Mat& image);
    // convertir une image en niveaux de gris
    static cv::Mat convertToGrayscale(const cv::Mat& image);
    // appliquer flou gaussien
    static cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize);
};

#endif 