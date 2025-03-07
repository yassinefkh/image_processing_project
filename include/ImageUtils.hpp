#ifndef IMAGEUTILS_HPP
#define IMAGEUTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ImageUtils {
public:
    // === Chargement et affichage ===
    static cv::Mat loadImage(const std::string& path);
    static void displayImage(const std::string& windowName, const cv::Mat& image);

    // === Prétraitement ===
    static cv::Mat convertToGrayscale(const cv::Mat& image);
    static cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize);
    static cv::Mat applyOtsuThreshold(const cv::Mat& image);
    static cv::Mat equalizeHistogram(const cv::Mat& image);
    static cv::Mat adjustContrastGamma(const cv::Mat& image, double gamma);

    // === Détection des contours ===
    static cv::Mat applyCanny(const cv::Mat& image, double threshold1, double threshold2);
    static cv::Mat applyDilation(const cv::Mat& image, int kernelSize);
    static cv::Mat applyErosion(const cv::Mat& image, int kernelSize);
    static cv::Mat applyOpening(const cv::Mat& image, int kernelSize);
    static void computeHorizontalProjectionHistogram(const cv::Mat& binaryImage);
    static std::pair<cv::Mat, int> detectTransitionsAndCountPairs(const cv::Mat& image, int xCoord);
    static std::pair<cv::Mat, std::vector<int>> scanImageForStepPatterns(const cv::Mat& image, int stride);
    static std::pair<int, int> detectStaircaseRegion(const cv::Mat& image, int threshold);
    
    



    static int estimateStepCount(const std::vector<int>& transitions);



    // === Détection et traitement des lignes ===
    static cv::Mat applyHoughTransform(const cv::Mat& edges, std::vector<cv::Vec4i>& detectedLines);
    static std::vector<cv::Vec4i> filterHorizontalLines(const std::vector<cv::Vec4i>& lines);
    static std::vector<cv::Vec4i> filterHorizontalLinesByLength(const std::vector<cv::Vec4i>& lines, double minLength, double maxAngleDeviation);
    static std::vector<cv::Vec4i> filterRegularlySpacedLines(const std::vector<cv::Vec4i>& lines, double& avgSpacing);
    static std::vector<cv::Vec4i> filterOutliersBasedOnSpacing(const std::vector<cv::Vec4i>& lines, double threshold);
    static std::vector<cv::Vec4i> mergeCloseLines(const std::vector<cv::Vec4i>& lines, double mergeThreshold);
    static std::vector<cv::Vec4i> fitStairModel(const std::vector<cv::Vec4i>& lines, double avgSpacing);
    static cv::Mat computeVanishingPoints(const std::vector<cv::Vec4i>& lines, cv::Mat& image);
    static std::vector<cv::Vec4i> mergeCloseParallelLines(const std::vector<cv::Vec4i>& lines, double maxYDistance);
    static std::vector<cv::Vec4i> sortLinesByLength(const std::vector<cv::Vec4i>& lines);
    static std::vector<cv::Vec4i> mergeOverlappingLines(const std::vector<cv::Vec4i>& lines, double maxYDistance);
    static std::vector<cv::Vec4i> filterIrregularlySpacedLines(const std::vector<cv::Vec4i>& lines, double expectedSpacing);
    static std::vector<cv::Vec4i> filterShortLines(const std::vector<cv::Vec4i>& lines, double minLength);
    static int getMostFrequentValue(const std::vector<int>& values);


    
    // === Visualisation ===
    static cv::Mat drawLabeledLines(const cv::Mat& inputImage, const std::vector<cv::Vec4i>& lines);

    // === Détection des composantes connexes ===
    static cv::Mat detectConnectedComponents(const cv::Mat& binaryImage, std::vector<cv::Rect>& components);

    // === Quantification de l'image ===
    static cv::Mat quantize(const cv::Mat& image, int numberOfLevels);


    static cv::Mat applyCLAHE(const cv::Mat& image);
    static cv::Mat applyGaborFilter(const cv::Mat& image);
    static cv::Mat extractROIUsingBlocks(const cv::Mat& edges, cv::Mat& edgesWithBlocks, int blockSize);
    static cv::Mat detectBlackBlocks(const cv::Mat& image, int blockSize);
    static cv::Mat removeIsolatedBlackBlocks(const cv::Mat& mask, int blockSize);
    static cv::Mat applyMaskToImage(const cv::Mat& image, const cv::Mat& mask);
    static cv::Mat computePrincipalAxis(const cv::Mat& edges, double& angle);
    static cv::Mat reduceMinimumValueOfHistogram(const cv::Mat& image, int minValue);
    static void exportProfile(const std::vector<double>& depthValues, const std::string& filename);
    static std::pair<cv::Point, cv::Point> computeLineEndpoints(const cv::Point2d& mean, const cv::Point2d& dir, int width, int height);
    static double calculateTransitionThreshold(const std::vector<double>& signal);
    static std::vector<int> detectTransitions(const std::vector<double>& signal);
       // Prétraitement de l'image (flou gaussien + CLAHE)
    static cv::Mat preprocessImage(const cv::Mat& image);

    // Détection des contours avec un filtre spécifique
    static cv::Mat detectEdges(const cv::Mat& image);

    // Extraction des points des contours
    static std::vector<cv::Point> extractContourPoints(const cv::Mat& edges);

    // Calcul de la PCA
    static std::pair<cv::Point2d, cv::Point2d> computePCA(const std::vector<cv::Point>& points);

    // Extraction du profil de profondeur
    static std::vector<double> extractDepthProfile(const cv::Mat& depthMap, 
                                                   const cv::Point2d& mean, 
                                                   const cv::Point2d& dir, 
                                                   std::vector<cv::Point>& profilePoints);
    
};

#endif // IMAGEUTILS_HPP
