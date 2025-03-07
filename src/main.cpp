#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>  
#include <vector>
#include <cmath>
#include <sstream> 
#include <cstdlib>
#include <thread>
#include <chrono>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"

namespace fs = std::filesystem;

std::string removeExtension(const std::string& filename) {
    size_t lastDot = filename.find_last_of(".");
    return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}

int executePythonScript() {
    std::string command = "python peak.py";  
    int ret = std::system(command.c_str());

    if (ret != 0) {
        std::cerr << "Erreur lors de l'exécution du script Python !" << std::endl;
        return -1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::ifstream resultFile("result.txt");
    int numSteps = -1;
    if (resultFile.is_open()) {
        resultFile >> numSteps;
        resultFile.close();
    } else {
        std::cerr << "Erreur : Impossible de lire le fichier de sortie Python." << std::endl;
    }

    return numSteps;
}

std::map<std::string, int> loadGroundTruth(const std::string& csvPath) {
    std::map<std::string, int> groundTruth;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible de lire le fichier CSV !" << std::endl;
        return groundTruth;
    }

    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue; 
        }

        std::stringstream ss(line);
        std::string imageName, stepsStr;
        int trueSteps;

        if (std::getline(ss, imageName, ',') && std::getline(ss, stepsStr)) {
            try {
                trueSteps = std::stoi(stepsStr);
                std::string nameWithoutExt = removeExtension(imageName);
                groundTruth[nameWithoutExt] = trueSteps;
            } catch (...) {
                std::cerr << "Erreur : Impossible de convertir '" << stepsStr << "' en nombre entier." << std::endl;
            }
        }
    }

    file.close();
    return groundTruth;
}

int main() {
    try {
        std::string folderPath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data";  
        std::string csvPath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/annotations.csv"; 

        std::map<std::string, int> groundTruth = loadGroundTruth(csvPath);
        if (groundTruth.empty()) {
            std::cerr << "Aucune donnée trouvée dans le fichier CSV !" << std::endl;
            return 1;
        }

        std::vector<int> detectedSteps;
        std::vector<int> trueSteps;

        for (const auto& entry : fs::directory_iterator(folderPath)) {
    
            if (entry.path().extension() != ".jpg" && entry.path().extension() != ".png" && entry.path().extension() != ".jpeg") 
                continue;

            std::string imagePath = entry.path().string();
            std::string imageName = entry.path().filename().string();
            std::string nameWithoutExt = removeExtension(imageName);
            std::string depthPath = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/depth/" + nameWithoutExt + ".png";  // On suppose que les depth maps sont en PNG

            if (groundTruth.find(nameWithoutExt) == groundTruth.end()) {
                std::cerr << "!!! Image " << imageName << " non trouvée dans le CSV, ignorée." << std::endl;
                continue;
            }

            cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_GRAYSCALE);
            if (image.empty() || depthMap.empty()) {
                std::cerr << "Erreur : Impossible de charger " << imageName << " ou sa depth map !" << std::endl;
                continue;
            }

            cv::Mat processedImage = ImageUtils::preprocessImage(image);

            cv::Mat edges = ImageUtils::detectEdges(processedImage);

            std::vector<cv::Point> points = ImageUtils::extractContourPoints(edges);
            if (points.empty()) {
                std::cerr << "!!! Aucune donnée détectée pour le PCA sur " << imageName << ", ignorée." << std::endl;
                continue;
            }

            auto [mean, principalVector] = ImageUtils::computePCA(points);

            std::vector<cv::Point> profilePoints;
            std::vector<double> depthValues = ImageUtils::extractDepthProfile(depthMap, mean, principalVector, profilePoints);
            ImageUtils::exportProfile(depthValues, "profil.csv");

            int numDetectedSteps = executePythonScript();
            if (numDetectedSteps < 0) {
                std::cerr << "!!! Échec de la détection pour " << imageName << ", ignorée." << std::endl;
                continue;
            }

            detectedSteps.push_back(numDetectedSteps);
            trueSteps.push_back(groundTruth[nameWithoutExt]);

            std::cout << ">> Image: " << imageName
                      << " | Détecté: " << numDetectedSteps
                      << " | Réel: " << groundTruth[nameWithoutExt] << std::endl;
        }

        if (detectedSteps.empty()) {
            std::cerr << "Aucune image n'a été traitée correctement !" << std::endl;
            return 1;
        }

        double mse = 0.0;
        for (size_t i = 0; i < detectedSteps.size(); i++) {
            mse += std::pow(detectedSteps[i] - trueSteps[i], 2);
        }
        mse /= detectedSteps.size();

        std::cout << "\n ---> MSE (Erreur Quadratique Moyenne) : " << mse << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
