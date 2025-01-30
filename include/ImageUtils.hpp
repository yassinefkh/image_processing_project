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
    // appliquer un flou gaussien
    static cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize);
    // appliquer le seuillage d'Otsu
    static cv::Mat applyOtsuThreshold(const cv::Mat& image);
    // appliquer la détection des contours avec Canny
    static cv::Mat applyCanny(const cv::Mat& image, double threshold1, double threshold2);
    // appliquer la transformée de Hough pour détecter des lignes
    static cv::Mat applyHoughTransform(const cv::Mat& edges, std::vector<cv::Vec4i>& detectedLines);
    // calcul des points de fuite
    static cv::Mat computeVanishingPoints(const std::vector<cv::Vec4i>& lines, cv::Mat& image);
    // dilatation
    static cv::Mat applyDilation(const cv::Mat& image, int kernelSize);

};

#endif
