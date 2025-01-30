#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>

cv::Mat ImageUtils::loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Erreur : Impossible de charger l'image à partir de " + path);
    }
    return image;
}

void ImageUtils::displayImage(const std::string& windowName, const cv::Mat& image) {
    cv::imshow(windowName, image);
}

cv::Mat ImageUtils::convertToGrayscale(const cv::Mat& image) {
    cv::Mat grayscaleImage;
    cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
    return grayscaleImage;
}

cv::Mat ImageUtils::applyGaussianBlur(const cv::Mat& image, int kernelSize) {
    if (kernelSize <= 0) {
        throw std::invalid_argument("Erreur : kernelSize doit être un entier strictement positif.");
    }
    if (kernelSize % 2 == 0) {
        kernelSize += 1; 
        std::cout << "Attention : kernelSize était pair, il a été corrigé à " << kernelSize << std::endl;
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), 0);
    return blurredImage;
}
cv::Mat ImageUtils::applyOtsuThreshold(const cv::Mat& image) {
    if (image.channels() > 1) {
        throw std::invalid_argument("Erreur : L'image d'entrée pour le seuillage d'Otsu doit être en niveaux de gris.");
    }

    cv::Mat thresholdedImage;
    cv::threshold(image, thresholdedImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    return thresholdedImage;
}