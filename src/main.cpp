#include <iostream>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"

int main() {
    try {
        cv::Mat image = ImageUtils::loadImage(INPUT_IMAGE_PATH);

        ImageUtils::displayImage("Image originale", image);

        cv::Mat grayscaleImage = ImageUtils::convertToGrayscale(image);

        ImageUtils::displayImage("Image en niveaux de gris", grayscaleImage);

        std::cout << "Traitement terminé avec succès." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }

    return 0;
}