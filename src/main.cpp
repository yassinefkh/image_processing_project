#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
      
        cv::Mat image = ImageUtils::loadImage(INPUT_IMAGE_PATH);
        ImageUtils::displayImage("Image originale", image);

        // Conversion en niveaux de gris 
        cv::Mat grayscaleImage = ImageUtils::convertToGrayscale(image);
        ImageUtils::displayImage("Image en niveaux de gris", grayscaleImage);

        // Floutage Gaussien 
        cv::Mat blurredImage = ImageUtils::applyGaussianBlur(grayscaleImage, 5);
        ImageUtils::displayImage("Image avec flou gaussien", blurredImage);

        // Détection des contours avec Canny 
        cv::Mat edges = ImageUtils::applyCanny(blurredImage, 50, 150);
        ImageUtils::displayImage("Contours détectés avec Canny", edges);

        // Dilatation pour renforcer les contours 
        cv::Mat dilatedEdges = ImageUtils::applyDilation(edges, 2);
        ImageUtils::displayImage("Contours épaissis avec dilatation", dilatedEdges);

        //  Détection des lignes avec la Transformée de Hough 
        std::vector<cv::Vec4i> detectedLines;
        cv::Mat houghImage = ImageUtils::applyHoughTransform(dilatedEdges, detectedLines);
        ImageUtils::displayImage("Lignes détectées avec Hough", houghImage);

        // Filtrage des lignes horizontales 
        std::vector<cv::Vec4i> horizontalLines = ImageUtils::filterHorizontalLines(detectedLines);
        cv::Mat horizontalImage = ImageUtils::drawLabeledLines(dilatedEdges, horizontalLines);
        ImageUtils::displayImage("Lignes horizontales détectées", horizontalImage);

        // Fusion des lignes proches d'une même marche 
        double seuilFusion = 5.0;  
        std::vector<cv::Vec4i> mergedLines = ImageUtils::mergeCloseLines(horizontalLines, seuilFusion);
        cv::Mat mergedImage = ImageUtils::drawLabeledLines(dilatedEdges, mergedLines);
        ImageUtils::displayImage("Lignes fusionnées (bords des marches)", mergedImage);

        // Sélection des marches par espacement régulier 
        double avgSpacing = 0.0;
        std::vector<cv::Vec4i> stairLines = ImageUtils::filterRegularlySpacedLines(mergedLines, avgSpacing);
        cv::Mat finalImage = ImageUtils::drawLabeledLines(dilatedEdges, stairLines);
        ImageUtils::displayImage("Marches détectées", finalImage);

        std::cout << "Nombre de marches détectées : " << stairLines.size() << std::endl;
        std::cout << "Espacement moyen entre les marches : " << avgSpacing << " pixels" << std::endl;

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
