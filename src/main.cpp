#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    try {
        cv::Mat image = ImageUtils::loadImage(INPUT_IMAGE_PATH);
        ImageUtils::displayImage("Image originale", image);

        cv::Mat grayscaleImage = ImageUtils::convertToGrayscale(image);
        ImageUtils::displayImage("Image en niveaux de gris", grayscaleImage);

        cv::Mat equalizedImage = ImageUtils::equalizeHistogram(grayscaleImage);
        ImageUtils::displayImage("Image égalisée", equalizedImage);

        cv::Mat blurredImage = ImageUtils::applyGaussianBlur(equalizedImage, 3);
        ImageUtils::displayImage("Image avec flou gaussien", blurredImage);

        cv::Mat edges = ImageUtils::applyCanny(blurredImage, 150, 300);
        ImageUtils::displayImage("Contours détectés avec Canny", edges);

        cv::Mat openedEdges = ImageUtils::applyOpening(edges, 1);
        ImageUtils::displayImage("Contours après ouverture", openedEdges);

        std::vector<cv::Vec4i> detectedLines;
        cv::Mat houghImage = ImageUtils::applyHoughTransform(openedEdges, detectedLines);
        ImageUtils::displayImage("Lignes détectées avec Hough", houghImage);

        std::vector<cv::Vec4i> horizontalLines = ImageUtils::filterHorizontalLines(detectedLines);
        cv::Mat horizontalImage = ImageUtils::drawLabeledLines(openedEdges, horizontalLines);
        ImageUtils::displayImage("Lignes horizontales détectées", horizontalImage);

        std::vector<cv::Vec4i> filteredLines = ImageUtils::filterShortLines(horizontalLines, 50);
        std::vector<cv::Vec4i> mergedLines = ImageUtils::mergeOverlappingLines(filteredLines, 10);
        std::vector<cv::Vec4i> cleanedLines = ImageUtils::filterIrregularlySpacedLines(mergedLines, 20);

        cv::Mat finalImage = ImageUtils::drawLabeledLines(openedEdges, cleanedLines);
        ImageUtils::displayImage("Marches détectées", finalImage);



        cv::Mat blurredImageBis = ImageUtils::applyGaussianBlur(grayscaleImage, 3);
        ImageUtils::displayImage("Image avec flou gaussien", blurredImageBis);


        cv::Mat quantizedImage = ImageUtils::quantize(blurredImageBis, 2);
        ImageUtils::displayImage("Image quantifiée", quantizedImage);

        int stride = 10;  // On analyse tous les 10 pixels en largeur
        auto [scanImage, stepPatterns] = ImageUtils::scanImageForStepPatterns(quantizedImage, stride);

        cv::imshow("Analyse complète des motifs de marches", scanImage);
        std::cout << "Nombre de paires détectées par colonne : " << std::endl;
        for (size_t i = 0; i < stepPatterns.size(); ++i) {
            std::cout << "Colonne " << i * stride << " : " << stepPatterns[i] << " paires" << std::endl;
        }


        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
