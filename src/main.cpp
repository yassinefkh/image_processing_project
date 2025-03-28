#include <opencv2/opencv.hpp>
#include <opencv2/plot.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <chrono>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"

namespace fs = std::filesystem;

std::string removeExtension(const std::string& filename) {
    size_t lastDot = filename.find_last_of('.');
    return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}

/**
 * @brief Exécute le script Python pour détecter les marches à partir du profil de profondeur.
 * 
 * @return Nombre de marches détectées ou -1 en cas d'erreur.
 */
int executePythonScript() {
    int ret = std::system("python3 peak.py");
    if (ret != 0) {
        std::cerr << "Erreur : Échec de l'exécution du script Python." << std::endl;
        return -1;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    std::ifstream resultFile("result.txt");
    int numSteps = -1;
    if (resultFile.is_open()) {
        resultFile >> numSteps;
        resultFile.close();
    } else {
        std::cerr << "Erreur : Impossible de lire result.txt." << std::endl;
    }

    return numSteps;
}

/**
 * @brief Charge les annotations (vérité terrain) à partir d'un fichier CSV.
 * 
 * @param csvPath Chemin vers le fichier CSV.
 * @return Dictionnaire associant chaque image à son nombre réel de marches.
 */
std::map<std::string, int> loadGroundTruth(const std::string& csvPath) {
    std::map<std::string, int> groundTruth;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir " << csvPath << std::endl;
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
        if (std::getline(ss, imageName, ',') && std::getline(ss, stepsStr)) {
            try {
                int steps = std::stoi(stepsStr);
                groundTruth[removeExtension(imageName)] = steps;
            } catch (...) {
                std::cerr << "Erreur : Conversion invalide dans le CSV (" << stepsStr << ")." << std::endl;
            }
        }
    }
    file.close();
    return groundTruth;
}

/**
 * @brief Programme principal.
 * 
 * Chargement des images et des cartes de profondeur, traitement par PCA et profil de profondeur,
 * exécution du script Python pour détection, et calcul des métriques d'erreur.
 */
int main() {
    try {
        std::string imgFolder = "data/img";
        std::string depthFolder = "data/depth";
        std::string csvPath = "data/annotations.csv";

        auto groundTruth = loadGroundTruth(csvPath);
        if (groundTruth.empty()) {
            std::cerr << "Erreur : Ground truth vide." << std::endl;
            return 1;
        }

        std::vector<int> detectedSteps;
        std::vector<int> trueSteps;

        for (const auto& entry : fs::directory_iterator(imgFolder)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                if (ext != ".jpg" && ext != ".png" && ext != ".jpeg") continue;

                std::string imagePath = entry.path().string();
                std::string imageName = entry.path().filename().string();
                std::string nameWithoutExt = removeExtension(imageName);

                if (groundTruth.find(nameWithoutExt) == groundTruth.end()) {
                    std::cerr << "Image " << imageName << " ignorée (pas dans le CSV)." << std::endl;
                    continue;
                }

                std::string depthPath = depthFolder + "/" + nameWithoutExt + "_depth";
                if (fs::exists(depthPath + ".png")) {
                    depthPath += ".png";
                } else if (fs::exists(depthPath + ".jpg")) {
                    depthPath += ".jpg";
                } else {
                    std::cerr << "Depth map introuvable pour " << imageName << "." << std::endl;
                    continue;
                }

                cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
                cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_GRAYSCALE);
                if (image.empty() || depthMap.empty()) {
                    std::cerr << "Erreur : Chargement de " << imageName << " ou de sa depth map impossible." << std::endl;
                    continue;
                }

                cv::Mat processed = ImageUtils::preprocessImage(image);
                cv::Mat edges = ImageUtils::detectEdges(processed);
                auto points = ImageUtils::extractContourPoints(edges);
                if (points.empty()) {
                    std::cerr << "Aucun point détecté pour " << imageName << "." << std::endl;
                    continue;
                }

                auto [mean, principalVector] = ImageUtils::computePCA(points);
                std::vector<cv::Point> profilePoints;
                auto depthValues = ImageUtils::extractDepthProfile(depthMap, mean, principalVector, profilePoints);
                ImageUtils::exportProfile(depthValues, "profil.csv");

                int numDetected = executePythonScript();
                if (numDetected < 0) {
                    std::cerr << "Échec détection pour " << imageName << "." << std::endl;
                    continue;
                }

                detectedSteps.push_back(numDetected);
                trueSteps.push_back(groundTruth[nameWithoutExt]);

                std::cout << "Image : " << imageName
                          << " | Détecté : " << numDetected
                          << " | Vérité terrain : " << groundTruth[nameWithoutExt]
                          << std::endl;
            }
        }

        if (detectedSteps.empty()) {
            std::cerr << "Aucune détection valide." << std::endl;
            return 1;
        }

        double mse = 0.0, mae = 0.0;
        for (size_t i = 0; i < detectedSteps.size(); i++) {
            mse += std::pow(detectedSteps[i] - trueSteps[i], 2);
            mae += std::abs(detectedSteps[i] - trueSteps[i]);
        }
        mse /= detectedSteps.size();
        mae /= detectedSteps.size();

        std::cout << "\n=== Résultats ===\n";
        std::cout << "MSE : " << mse << std::endl;
        std::cout << "MAE : " << mae << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
