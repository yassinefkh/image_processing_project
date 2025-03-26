#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"

int main() {
    try {
        // chargement des images
        std::string imagePath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/img/t3i18.png";
        std::string depthPath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/depth/t3i18_depth.png";

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_GRAYSCALE);
        if (image.empty() || depthMap.empty()) {
            std::cerr << "Erreur : Impossible de charger l'image ou la depth map !" << std::endl;
            return 1;
        }

        // prétraitement de l'image
        cv::Mat processedImage = ImageUtils::preprocessImage(image);
        cv::imshow("Image Prétraitée", processedImage);

        // détection des contours
        cv::Mat edges = ImageUtils::detectEdges(processedImage);
        cv::imshow("Contours", edges);

        // extraction des points de contour et PCA
        std::vector<cv::Point> points = ImageUtils::extractContourPoints(edges);
        if (points.empty()) {
            std::cerr << "Erreur : Aucune donnée détectée pour le PCA !" << std::endl;
            return 1;
        }

        auto [mean, principalVector] = ImageUtils::computePCA(points);
        auto [pt1, pt2] = ImageUtils::computeLineEndpoints(mean, principalVector, depthMap.cols, depthMap.rows);

        // affichage de l'axe principal sur l'image des contours
        cv::Mat axisImage;
        cv::cvtColor(edges, axisImage, cv::COLOR_GRAY2BGR);
        cv::line(axisImage, pt1, pt2, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Axe Principal", axisImage);

        // affichage de la depth map avec la ligne tracée
        cv::Mat depthWithLine;
        cv::cvtColor(depthMap, depthWithLine, cv::COLOR_GRAY2BGR);
        cv::line(depthWithLine, pt1, pt2, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Depth Map avec Ligne PCA", depthWithLine);

        // extraction du profil de profondeur
        std::vector<cv::Point> profilePoints;
        std::vector<double> depthValues = ImageUtils::extractDepthProfile(depthMap, mean, principalVector, profilePoints);
        ImageUtils::exportProfile(depthValues, "profil.csv");

        // détection des transitions (marches)
        std::vector<int> transitionIndices = ImageUtils::detectTransitions(depthValues);
        std::cout << "Nombre de marches détectées : " << transitionIndices.size() << std::endl;

        // affichage des transitions sur le profil
        cv::Mat plot;
        if (!depthValues.empty()) {
            cv::Ptr<cv::plot::Plot2d> plotProfile = cv::plot::Plot2d::create(cv::Mat(depthValues));
            plotProfile->render(plot);
            for (int idx : transitionIndices) {
                cv::circle(plot, cv::Point(idx, depthValues[idx]), 5, cv::Scalar(0, 0, 255), -1);
            }
            cv::imshow("Profil de Profondeur avec Transitions", plot);
        }

        // affichage des marches détectées sur l'image
        cv::Mat imageWithSteps;
        cv::cvtColor(image, imageWithSteps, cv::COLOR_GRAY2BGR);
        for (int idx : transitionIndices) {
            cv::Point pt = profilePoints[idx];
            cv::circle(imageWithSteps, pt, 5, cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("Image avec Marches Détectées", imageWithSteps);

        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
