#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <cmath>

/**
 * @brief Exporte les valeurs du profil de profondeur dans un fichier texte.
 * 
 * @param depthValues Vecteur contenant les valeurs de profondeur.
 * @param filename Nom du fichier de sortie.
 */
void ImageUtils::exportProfile(const std::vector<double>& depthValues, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }
    for (double val : depthValues) {
        file << val << "\n";
    }
    file.close();
}

/**
 * @brief Calcule les points d'intersection d'une ligne avec les bords de l'image.
 * 
 * @param mean Point moyen (centre) de la ligne.
 * @param dir Vecteur directionnel de la ligne.
 * @param width Largeur de l'image.
 * @param height Hauteur de l'image.
 * @return Paire de points représentant les extrémités de la ligne.
 */
std::pair<cv::Point, cv::Point> ImageUtils::computeLineEndpoints(const cv::Point2d& mean, const cv::Point2d& dir, int width, int height) {
    cv::Point pt1, pt2;

    if (std::abs(dir.x) < 1e-6) {
        pt1 = cv::Point(mean.x, 0);
        pt2 = cv::Point(mean.x, height - 1);
        return {pt1, pt2};
    }

    double slope = dir.y / dir.x;
    pt1 = cv::Point(0, mean.y - slope * mean.x);
    pt2 = cv::Point(width - 1, mean.y + slope * (width - 1 - mean.x));

    if (pt1.y < 0 || pt1.y >= height) {
        pt1.y = (pt1.y < 0) ? 0 : height - 1;
        pt1.x = mean.x + (pt1.y - mean.y) / slope;
    }
    if (pt2.y < 0 || pt2.y >= height) {
        pt2.y = (pt2.y < 0) ? 0 : height - 1;
        pt2.x = mean.x + (pt2.y - mean.y) / slope;
    }

    return {pt1, pt2};
}

/**
 * @brief Applique un flou gaussien suivi d'une égalisation adaptative (CLAHE) sur une image.
 * 
 * @param image Image d'entrée en niveaux de gris.
 * @return Image prétraitée.
 */
cv::Mat ImageUtils::preprocessImage(const cv::Mat& image) {
    cv::Mat blurred, claheImage;
    cv::GaussianBlur(image, blurred, cv::Size(15, 15), 0);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(blurred, claheImage);
    return claheImage;
}

/**
 * @brief Détecte les contours d'une image en utilisant les filtres Sobel.
 * 
 * @param image Image en niveaux de gris.
 * @return Image binaire des contours détectés.
 */
cv::Mat ImageUtils::detectEdges(const cv::Mat& image) {
    cv::Mat edges, sobelX, sobelY;
    cv::Sobel(image, sobelX, CV_16S, 1, 0);
    cv::Sobel(image, sobelY, CV_16S, 0, 1);
    cv::convertScaleAbs(sobelX, sobelX);
    cv::convertScaleAbs(sobelY, sobelY);
    cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, edges);
    cv::threshold(edges, edges, 50, 255, cv::THRESH_BINARY);
    return edges;
}

/**
 * @brief Extrait les coordonnées des points de contour à partir d'une image binaire.
 * 
 * @param edges Image binaire des contours.
 * @return Vecteur de points appartenant aux contours.
 */
std::vector<cv::Point> ImageUtils::extractContourPoints(const cv::Mat& edges) {
    std::vector<cv::Point> points;
    for (int y = 0; y < edges.rows; y++) {
        for (int x = 0; x < edges.cols; x++) {
            if (edges.at<uchar>(y, x) > 0) {
                points.push_back(cv::Point(x, y));
            }
        }
    }
    return points;
}

/**
 * @brief Calcule l'axe principal (via PCA) des points d'un contour.
 * 
 * @param points Vecteur de points du contour.
 * @return Paire contenant le point moyen et le vecteur directionnel principal.
 */
std::pair<cv::Point2d, cv::Point2d> ImageUtils::computePCA(const std::vector<cv::Point>& points) {
    cv::Mat data(points.size(), 2, CV_64F);
    for (size_t i = 0; i < points.size(); i++) {
        data.at<double>(i, 0) = points[i].x;
        data.at<double>(i, 1) = points[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    return {cv::Point2d(pca.mean), cv::Point2d(pca.eigenvectors.row(0))};
}

/**
 * @brief Extrait le profil de profondeur le long de l'axe principal.
 * 
 * @param depthMap Image de profondeur correspondante.
 * @param mean Point moyen de l'axe principal.
 * @param dir Vecteur directionnel principal.
 * @param profilePoints Vecteur de points où les valeurs de profondeur sont extraites.
 * @return Vecteur contenant les valeurs de profondeur extraites.
 */
std::vector<double> ImageUtils::extractDepthProfile(const cv::Mat& depthMap, const cv::Point2d& mean, const cv::Point2d& dir, std::vector<cv::Point>& profilePoints) {
    std::vector<double> depthValues;
    double step = std::sqrt(dir.x * dir.x + dir.y * dir.y);
    for (double t = -depthMap.cols; t <= depthMap.cols; t += step) {
        int x = mean.x + t * dir.x;
        int y = mean.y + t * dir.y;
        if (x >= 0 && x < depthMap.cols && y >= 0 && y < depthMap.rows) {
            profilePoints.push_back(cv::Point(x, y));
            depthValues.push_back(depthMap.at<uchar>(y, x));
        }
    }
    return depthValues;
}
