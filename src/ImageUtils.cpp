#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

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
    //cv::threshold(image, thresholdedImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(image, thresholdedImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    return thresholdedImage;
}


cv::Mat ImageUtils::applyCanny(const cv::Mat& image, double threshold1, double threshold2) {
    cv::Mat edges;
    cv::Canny(image, edges, threshold1, threshold2);
    return edges;
}

cv::Mat ImageUtils::applyDilation(const cv::Mat& image, int kernelSize) {
    cv::Mat dilatedImage;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
    
    cv::dilate(image, dilatedImage, kernel);
    
    return dilatedImage;
}


cv::Mat ImageUtils::applyHoughTransform(const cv::Mat& edges, std::vector<cv::Vec4i>& detectedLines) {
    cv::Mat houghImage;
    cv::cvtColor(edges, houghImage, cv::COLOR_GRAY2BGR); 

    cv::HoughLinesP(edges, detectedLines, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);
    
    std::cout << "Lignes détectées (x1, y1, x2, y2) : " << std::endl;
    for (size_t i = 0; i < detectedLines.size(); i++) {
        cv::Vec4i l = detectedLines[i];
        cv::line(houghImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        std::cout << "Ligne " << i + 1 << ": (" << l[0] << ", " << l[1] << ") -> (" << l[2] << ", " << l[3] << ")" << std::endl;
    }

    return houghImage;
}


cv::Point2f computeIntersection(cv::Vec4i line1, cv::Vec4i line2) {
    float x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
    float x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
    // calcul des coefficients des droites (y = ax + b)
    float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (denom == 0) return cv::Point2f(-1, -1); // lignes parallèles

    float px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
    float py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
    
    return cv::Point2f(px, py);
}

cv::Mat ImageUtils::computeVanishingPoints(const std::vector<cv::Vec4i>& lines, cv::Mat& image) {
    cv::Mat outputImage = image.clone();
    std::vector<cv::Point2f> vanishingPoints;

    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            cv::Point2f intersection = computeIntersection(lines[i], lines[j]);

            if (intersection.x >= 0 && intersection.y >= 0 && intersection.x < image.cols && intersection.y < image.rows) {
                vanishingPoints.push_back(intersection);
                cv::circle(outputImage, intersection, 5, cv::Scalar(0, 255, 0), -1); 
                cv::line(outputImage, cv::Point(lines[i][0], lines[i][1]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(outputImage, cv::Point(lines[i][2], lines[i][3]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                
                cv::line(outputImage, cv::Point(lines[j][0], lines[j][1]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(outputImage, cv::Point(lines[j][2], lines[j][3]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }
        }
    }

    std::cout << "Points de fuite détectés : " << std::endl;
    for (const auto& vp : vanishingPoints) {
        std::cout << "Point : (" << vp.x << ", " << vp.y << ")" << std::endl;
    }

    return outputImage;
}

