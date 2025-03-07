#include <opencv2/opencv.hpp>
#include <iostream>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"

int main() {
    try {
        // Charger l'image originale et la depth map
        std::string imagePath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/Groupe1_Image6.jpg";
        std::string depthPath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/tmpjl07t406.png"; 

        cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
        cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_GRAYSCALE);

        if (image.empty() || depthMap.empty()) {
            std::cerr << "Erreur : Impossible de charger l'image ou la depth map !" << std::endl;
            return 1;
        }

        cv::Mat mask;
        double threshold_value = 30;
        cv::threshold(depthMap, mask, threshold_value, 255, cv::THRESH_BINARY_INV);

        cv::Mat result;
        cv::bitwise_not(mask, mask);
        image.copyTo(result, mask);


        cv::Canny(result, result, 100, 200);
        cv::imshow("Canny", result);

        
        cv::Mat binaryImageA;
        cv::adaptiveThreshold(result, binaryImageA, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);
        cv::imshow("Adaptative Threshold", binaryImageA);


        cv::medianBlur(binaryImageA, result, 15);
        cv::imshow("Median Blur", result);


        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        clahe->apply(result, result);
        cv::imshow("CLAHE", result);

         // blur
        cv::GaussianBlur(result, result, cv::Size(15, 15), 0);
        //cv::imshow("Gaussian Blur", result);
        


        cv::Mat quantizedImage = ImageUtils::quantize(result, 2);

        // adaptative threshold
        cv::Mat binaryImageB;
        cv::adaptiveThreshold(quantizedImage, binaryImageB, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);
        //cv::imshow("Adaptative Threshold", binaryImageB);

        // Érosion morphologique avec un kernel horizontal pour réduire le bruit
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 1));
        cv::Mat erodedImage;
        cv::erode(quantizedImage, erodedImage, kernel);

        // Fermeture morphologique pour reconnecter les composantes
        cv::Mat closedImage;
        cv::morphologyEx(erodedImage, closedImage, cv::MORPH_CLOSE, kernel);


        cv::Mat binaryImage;
        cv::threshold(closedImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

       
        cv::Mat labels, stats, centroids;
        int numLabels = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids, 8, CV_32S);

        int minSize = 50;
        cv::Mat smallComponents = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
        cv::Mat filteredImage = binaryImage.clone();

        // Identifier et afficher les petites composantes connexes
        for (int i = 1; i < numLabels; i++) {  // i=0 correspond au fond
            if (stats.at<int>(i, cv::CC_STAT_AREA) < minSize) {
                smallComponents.setTo(255, labels == i);
                filteredImage.setTo(0, labels == i);
            }
        }

        //cv::imshow("Petites composantes à supprimer", smallComponents);

        cv::imshow("Image originale", image);
        //cv::imshow("Depth Map", depthMap);
        //cv::imshow("Image après masquage", result);
        cv::imshow("Image quantifiée", quantizedImage);
        cv::imshow("Image érodée", erodedImage);
        //cv::imshow("Composantes connexes détectées", binaryImage);
        cv::imshow("Image après suppression des petites composantes", filteredImage);

       
        cv::waitKey(0);
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
