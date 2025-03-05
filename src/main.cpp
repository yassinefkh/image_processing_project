#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"

std::vector<cv::Vec4i> getHorizontalHoughLines(const cv::Mat& edgeImage, 
                                               double angleThreshold = 15.0, 
                                               double minEccentricity = 5.0, 
                                               double minLength = 100.0) {
    std::vector<cv::Vec4i> lines, filteredLines;

    cv::HoughLinesP(edgeImage, lines, 1, CV_PI / 180, 50, 30, 10);

    for (const auto& line : lines) {
        double dx = line[2] - line[0];  
        double dy = line[3] - line[1]; 
        double angle = std::atan2(dy, dx) * 180.0 / CV_PI; 
        double length = std::sqrt(dx * dx + dy * dy);  
        double eccentricity = std::abs(dx) / (std::abs(dy) + 1e-5);

        if (std::abs(angle) < angleThreshold && eccentricity > minEccentricity && length > minLength) {
            filteredLines.push_back(line);
        }
    }

    return filteredLines; 
}



int main() {
    try {
        std::string inputImagePath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/t3i25.jpg";
        cv::Mat image = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Erreur : Impossible de charger l'image !" << std::endl;
            return 1;
        }

        cv::Mat claheImage = ImageUtils::applyCLAHE(image);
        cv::Mat gaborImage = ImageUtils::applyGaborFilter(claheImage);
        cv::normalize(gaborImage, gaborImage, 0, 255, cv::NORM_MINMAX);
        gaborImage.convertTo(gaborImage, CV_8U);
        cv::Mat edges = ImageUtils::detectEdges(gaborImage);

        int blockSize = 100; 
        cv::Mat edgesWithBlocks;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::Mat dilatedEdges;
        cv::dilate(edges, dilatedEdges, kernel);
        cv::Mat closedEdges;
        cv::morphologyEx(dilatedEdges, closedEdges, cv::MORPH_CLOSE, kernel);
        
        cv::Mat stairMask = ImageUtils::extractROIUsingBlocks(closedEdges, edgesWithBlocks, blockSize);
        if (stairMask.empty()) {
            std::cerr << "Erreur : le masque des escaliers est vide !" << std::endl;
            return 1;
        }

        cv::Mat finalROI;
        image.copyTo(finalROI, stairMask);
        if (finalROI.empty()) {
            std::cerr << "Erreur : l'isolement de la région des escaliers a échoué !" << std::endl;
            return 1;
        }

        cv::Mat blackBlockMask = ImageUtils::detectBlackBlocks(finalROI, blockSize);
        if (blackBlockMask.empty()) {
            std::cerr << "Erreur : aucun bloc noir détecté !" << std::endl;
            return 1;
        }

        cv::imshow("Contours détectés", edges);
        cv::imshow("Contours avec blocs retenus", edgesWithBlocks);
        cv::imshow("Masque des escaliers", stairMask);
        cv::imshow("Région des escaliers isolée", finalROI);
        cv::imshow("Masque des blocs noirs", blackBlockMask);

        cv::Mat cleanedMask = ImageUtils::removeIsolatedBlackBlocks(blackBlockMask, blockSize);
        if (cleanedMask.empty()) {
            std::cerr << "Erreur : aucun bloc noir isolé détecté !" << std::endl;
            return 1;
        }

        cv::imshow("Masque des blocs noirs nettoyé", cleanedMask);

        cv::Mat finalMaskedImage = ImageUtils::applyMaskToImage(image, 1-cleanedMask);
        cv::imshow("Région finale après application du masque finalmaskedimage", finalMaskedImage);

        //cv::Mat quantizedImage = ImageUtils::quantize(finalMaskedImage, 4);
        //cv::imshow("Image quantifiée", quantizedImage);

  
        //cv::Mat cannyEdges;
        //scv::Canny(finalMaskedImage, cannyEdges, 50, 150);
        //cv::imshow("Contours de l'image finale", cannyEdges);

/*         cv::Mat gaborFinal = ImageUtils::applyGaborFilter(finalMaskedImage);
        cv::normalize(gaborFinal, gaborFinal, 0, 255, cv::NORM_MINMAX);
        gaborFinal.convertTo(gaborFinal, CV_8U);
        cv::Mat finalEdges = ImageUtils::detectEdges(gaborFinal);

        cv::imshow("Filtrage Gabor sur image finale", gaborFinal);
        cv::imshow("Contours Canny sur image finale", finalEdges); */

        cv::Mat sobelX, sobelY, sobelFinal;

        cv::Sobel(finalMaskedImage, sobelY, CV_32F, 0, 1, 3); 

        cv::convertScaleAbs(sobelY, sobelFinal);

        cv::threshold(sobelFinal, sobelFinal, 50, 255, cv::THRESH_BINARY);

        cv::imshow("Contours horizontaux (Sobel Y)", sobelFinal);



        double stairAngle;
        cv::Mat axisImage = ImageUtils::computePrincipalAxis(sobelFinal, stairAngle);
        cv::imshow("Axe principal de l'escalier", axisImage);
        std::cout << "Angle principal de l'escalier : " << stairAngle << " degrés" << std::endl;


        cv::Mat medianFiltered;
        cv::medianBlur(sobelFinal, medianFiltered, 5);
        cv::imshow("Contours filtrés par médiane", medianFiltered);
    
        cv::Mat blurredFinal, thresholded;
        cv::GaussianBlur(medianFiltered, blurredFinal, cv::Size(9, 9), 20);
        cv::imshow("Contours floutés", blurredFinal);
        cv::threshold(blurredFinal, thresholded, 50, 255, cv::THRESH_BINARY);
        cv::imshow("Contours floutés et seuillés", thresholded);
        cv::Mat dilatedBlurred;
        cv::Mat kernelBis = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
        cv::dilate(thresholded, dilatedBlurred, kernelBis);
        cv::imshow("Contours dilatés", dilatedBlurred);


    std::vector<cv::Vec4i> horizontalLines = getHorizontalHoughLines(blurredFinal);


    cv::Mat houghImage;
    cv::cvtColor(blurredFinal, houghImage, cv::COLOR_GRAY2BGR);

    for (const auto& line : horizontalLines) {
        cv::line(houghImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Lignes horizontales détectées", houghImage);
    cv::waitKey(0);




        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
