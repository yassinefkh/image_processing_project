#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

std::vector<float> removeOutliersIQR(std::vector<float>& values) {
    if (values.size() < 4) return values; // Pas assez de données pour calculer l'IQR

    std::sort(values.begin(), values.end());

    float Q1 = values[values.size() / 4];
    float Q3 = values[3 * values.size() / 4];
    float IQR = Q3 - Q1;

    float lowerBound = Q1 - 1.5 * IQR;
    float upperBound = Q3 + 1.5 * IQR;

    std::vector<float> filteredValues;
    for (float v : values) {
        if (v >= lowerBound && v <= upperBound) {
            filteredValues.push_back(v);
        }
    }
    return filteredValues;
}


int main() {
    try {
        std::string inputImagePath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/input/img5.png";
        cv::Mat image = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Erreur : Impossible de charger l'image !" << std::endl;
            return 1;
        }

        // Prétraitement de l'image
        cv::Mat blurredImage;
        cv::GaussianBlur(image, blurredImage, cv::Size(5, 5), 1.5);

        cv::Mat sharpenedImage;
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<  
                           0, -1,  0,  
                          -1,  5, -1,  
                           0, -1,  0);
        cv::filter2D(blurredImage, sharpenedImage, -1, kernel);

        cv::Mat thresholdedImage;
        cv::adaptiveThreshold(sharpenedImage, thresholdedImage, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv::THRESH_BINARY_INV, 15, 5);

        // Extraction des contours horizontaux
        cv::Mat horizontalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 1));
        cv::Mat dilatedImage;
        cv::dilate(thresholdedImage, dilatedImage, horizontalKernel);

        cv::Mat erodedImage;
        cv::erode(dilatedImage, erodedImage, horizontalKernel);

        cv::Mat verticalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 25));
        cv::Mat verticalFiltered;
        cv::morphologyEx(thresholdedImage, verticalFiltered, cv::MORPH_OPEN, verticalKernel);
        cv::bitwise_not(verticalFiltered, verticalFiltered);

        cv::Mat horizontalLinesOnly;
        cv::bitwise_and(erodedImage, verticalFiltered, horizontalLinesOnly);

        cv::Mat cleanImage;
        cv::morphologyEx(horizontalLinesOnly, cleanImage, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 1)));

        // Filtrage par excentricité
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(cleanImage, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat eccentricityFiltered = cv::Mat::zeros(cleanImage.size(), CV_8UC1);
        std::vector<float> stepYPositions;

        for (const auto& contour : contours) {
            if (contour.size() < 5) continue;
            cv::RotatedRect ellipse = cv::fitEllipse(contour);
            double minorAxis = std::min(ellipse.size.width, ellipse.size.height);
            double majorAxis = std::max(ellipse.size.width, ellipse.size.height);
            double eccentricity = std::sqrt(1 - (minorAxis * minorAxis) / (majorAxis * majorAxis));
            if (eccentricity > 0.99) {
                cv::drawContours(eccentricityFiltered, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
            }
        }

        // Filtrage par longueur d'axe majeur
        std::vector<std::vector<cv::Point>> contoursFiltered;
        std::vector<cv::Vec4i> hierarchyFiltered;
        cv::findContours(eccentricityFiltered, contoursFiltered, hierarchyFiltered, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::Mat majorAxisFiltered = cv::Mat::zeros(cleanImage.size(), CV_8UC1);

        double minMajorAxisLength = 200;
        double maxMajorAxisLength = 1000;

        for (const auto& contour : contoursFiltered) {
            if (contour.size() < 5) continue;
            cv::RotatedRect ellipse = cv::fitEllipse(contour);
            double majorAxis = std::max(ellipse.size.width, ellipse.size.height);
            if (majorAxis >= minMajorAxisLength && majorAxis <= maxMajorAxisLength) {
                stepYPositions.push_back(ellipse.center.y);
                cv::drawContours(majorAxisFiltered, std::vector<std::vector<cv::Point>>{contour}, -1, 255, cv::FILLED);
            }
        }


        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(majorAxisFiltered, lines, 1, CV_PI / 180, 100, 200, 10);

        cv::Mat houghVisualization = majorAxisFiltered.clone();
        cv::cvtColor(houghVisualization, houghVisualization, cv::COLOR_GRAY2BGR);

        for (const auto& line : lines) {
            float y1 = static_cast<float>(line[1]);
            float y2 = static_cast<float>(line[3]);
            float avgY = (y1 + y2) / 2; // Prendre la moyenne de la ligne
            stepYPositions.push_back(avgY);

            cv::line(houghVisualization, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Lignes détectées (Hough)", houghVisualization);

        // Visualisation des marches détectées
        cv::Mat stepVisualization = majorAxisFiltered.clone();
        cv::cvtColor(stepVisualization, stepVisualization, cv::COLOR_GRAY2BGR);



        cv::imshow("Positions des marches détectées", stepVisualization);
        stepYPositions = removeOutliersIQR(stepYPositions);
        cv::imshow("Positions des marches détectées", stepVisualization);

                for (float y : stepYPositions) {
            cv::line(stepVisualization, cv::Point(0, y), cv::Point(stepVisualization.cols, y), cv::Scalar(0, 255, 0), 2);
        }
        // K-Means pour regrouper les marches
        if (!stepYPositions.empty()) {
            cv::Mat stepData(stepYPositions.size(), 1, CV_32F);
            for (size_t i = 0; i < stepYPositions.size(); i++) {
                stepData.at<float>(i, 0) = stepYPositions[i];
            }

            // Normalisation
            double minVal, maxVal;
            cv::minMaxLoc(stepData, &minVal, &maxVal);
            stepData = (stepData - minVal) / (maxVal - minVal);

            // Déterminer un nombre raisonnable de clusters
            int numClusters = std::max(2, std::min(15, static_cast<int>(stepYPositions.size() / 2)));

            cv::Mat labels, centers;
            cv::kmeans(stepData, numClusters, labels, 
                       cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 10, 1.0),
                       3, cv::KMEANS_PP_CENTERS, centers);

            std::cout << "Nombre estimé de marches (K-Means) : " << centers.rows << std::endl;

            // Visualisation du clustering
            cv::Mat kmeansVisualization = image.clone();
            cv::cvtColor(kmeansVisualization, kmeansVisualization, cv::COLOR_GRAY2BGR);

            for (int i = 0; i < centers.rows; i++) {
                float y = centers.at<float>(i, 0) * (maxVal - minVal) + minVal;
                cv::line(kmeansVisualization, cv::Point(0, y), cv::Point(kmeansVisualization.cols, y), cv::Scalar(0, 0, 255), 2);
            }

            cv::imshow("Clusters K-Means des marches", kmeansVisualization);
        }

        // Affichage des étapes
        cv::imshow("Image originale", image);
        cv::imshow("Image floutée", blurredImage);
        cv::imshow("Image après netteté", sharpenedImage);
        cv::imshow("Image seuillée", thresholdedImage);
        cv::imshow("Après dilatation horizontale", dilatedImage);
        cv::imshow("Après érosion", erodedImage);
        cv::imshow("Filtrage des structures verticales", verticalFiltered);
        cv::imshow("Lignes horizontales uniquement", horizontalLinesOnly);
        cv::imshow("Après nettoyage final", cleanImage);
        cv::imshow("Filtrage par excentricité", eccentricityFiltered);
        cv::imshow("Filtrage par Major Axis Length", majorAxisFiltered);

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
