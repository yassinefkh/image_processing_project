#include <iostream>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"

int main() {
    try {
        cv::Mat image = ImageUtils::loadImage(INPUT_IMAGE_PATH);
        ImageUtils::displayImage("Image originale", image);

        cv::Mat grayscaleImage = ImageUtils::convertToGrayscale(image);
        ImageUtils::displayImage("Image en niveaux de gris", grayscaleImage);

        cv::Mat blurredImage = ImageUtils::applyGaussianBlur(grayscaleImage, GAUSSIAN_BLUR_KERNEL_SIZE);
        ImageUtils::displayImage("Image avec flou gaussien", blurredImage);

        cv::Mat otsuThresholdedImage = ImageUtils::applyOtsuThreshold(blurredImage);
        ImageUtils::displayImage("Image seuillée avec Otsu", otsuThresholdedImage);

        cv::Mat edges = ImageUtils::applyCanny(otsuThresholdedImage, 50, 150);
        ImageUtils::displayImage("Contours détectés avec Canny", edges);

        cv::Mat dilatedEdges = ImageUtils::applyDilation(edges, 10);
        ImageUtils::displayImage("Contours épaissis avec dilatation", dilatedEdges);
        
        std::vector<cv::Vec4i> detectedLines;
        cv::Mat houghImage = ImageUtils::applyHoughTransform(dilatedEdges, detectedLines);
        ImageUtils::displayImage("Lignes détectées avec Hough", houghImage);

        cv::Mat vanishingPointsImage = ImageUtils::computeVanishingPoints(detectedLines, houghImage);
        ImageUtils::displayImage("Points de fuite détectés", vanishingPointsImage);

        cv::waitKey(0); 
        cv::destroyAllWindows(); 

        std::cout << "Traitement terminé avec succès." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
