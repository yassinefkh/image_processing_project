#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"



int main() {
    try {
        // Chargement des images
        std::string imagePath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/img/t3i23.jpg";
        std::string depthPath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/depth/t3i23_depth.png";
        cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_GRAYSCALE);

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Erreur : impossible de charger l'image !" << std::endl;
            return 1;
        }

        // Prétraitement de l'image
        cv::Mat processedImage = ImageUtils::preprocessImage(image);
        cv::imshow("Image Prétraitée", processedImage);

        // Détection des contours
        cv::Mat edges = ImageUtils::detectEdges(processedImage);
        cv::imshow("Contours", edges);

        std::vector<cv::Vec4i> detectedLines;
        cv::Mat houghImage = ImageUtils::applyHoughTransform(edges, depthMap, detectedLines);
        ImageUtils::displayImage("Lignes détectées avec Hough", houghImage);


        cv::waitKey(0); // Attendre une touche pour fermer les fenêtres

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}