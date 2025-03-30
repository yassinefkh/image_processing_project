#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <map>
#include <cmath>
#include <thread>
#include <chrono>
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"

namespace fs = std::filesystem;

/**
 * @brief Supprime l'extension d'un nom de fichier.
 */
std::string removeExtension(const std::string& filename) {
    size_t lastDot = filename.find_last_of('.');
    return (lastDot == std::string::npos) ? filename : filename.substr(0, lastDot);
}

/**
 * @brief Exécute le script Python de détection de marches.
 */
int executePythonScript() {
    int ret = std::system("python3 peak.py");
    if (ret != 0) {
        std::cerr << "Erreur : Échec du script Python." << std::endl;
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
 * @brief Charge le test set avec labels à partir du fichier CSV.
 */
std::vector<std::tuple<std::string, int, std::string>> loadTestSet(const std::string& csvPath) {
    std::vector<std::tuple<std::string, int, std::string>> data;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir " << csvPath << std::endl;
        return data;
    }

    std::string line;
    bool firstLine = true;
    while (std::getline(file, line)) {
        if (firstLine) {
            firstLine = false;
            continue;
        }
        std::stringstream ss(line);
        std::string imageName, stepsStr, difficulty;
        if (std::getline(ss, imageName, ',') && std::getline(ss, stepsStr, ',') && std::getline(ss, difficulty)) {
            try {
                int steps = std::stoi(stepsStr);
                data.emplace_back(imageName, steps, difficulty);
            } catch (...) {
                std::cerr << "Erreur : Problème de parsing sur " << imageName << std::endl;
            }
        }
    }
    file.close();
    return data;
}

/**
 * @brief Programme principal d'évaluation.
 */
int main() {
    try {
        std::string imgFolder = "data/img";
        std::string depthFolder = "data/depth";
        std::string testCsvPath = "data/test_set.csv";

        auto testSet = loadTestSet(testCsvPath);
        if (testSet.empty()) {
            std::cerr << "Erreur : Test set vide." << std::endl;
            return 1;
        }

        std::vector<int> detectedSteps;
        std::vector<int> trueSteps;
        std::vector<double> maes;
        std::vector<double> mses;
        std::vector<std::tuple<std::string, int, std::string>> validSamples;

        for (const auto& [imageName, trueCount, difficulty] : testSet) {
            std::string nameWithoutExt = removeExtension(imageName);

            std::string imagePath = imgFolder + "/" + imageName;
            std::string depthPath = depthFolder + "/" + nameWithoutExt + "_depth.png";

            if (!fs::exists(imagePath) || !fs::exists(depthPath)) {
                std::cerr << "Fichiers manquants pour " << imageName << "." << std::endl;
                continue;
            }

            cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            cv::Mat depthMap = cv::imread(depthPath, cv::IMREAD_GRAYSCALE);
            if (image.empty() || depthMap.empty()) {
                std::cerr << "Erreur de chargement pour " << imageName << "." << std::endl;
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

            int detected = executePythonScript();
            if (detected < 0) {
                std::cerr << "Détection échouée pour " << imageName << "." << std::endl;
                continue;
            }

            double mae = std::abs(detected - trueCount);
            double mse = std::pow(detected - trueCount, 2);
            detectedSteps.push_back(detected);
            trueSteps.push_back(trueCount);
            maes.push_back(mae);
            mses.push_back(mse);
            validSamples.emplace_back(imageName, trueCount, difficulty);

            std::cout << "Image : " << imageName
                      << " | Détecté : " << detected
                      << " | Vérité terrain : " << trueCount
                      << " | MAE : " << mae
                      << " | MSE : " << mse << std::endl;
        }

        if (detectedSteps.empty()) {
            std::cerr << "Aucune détection valide." << std::endl;
            return 1;
        }

        double maeTotal = 0.0;
        double mseTotal = 0.0;
        for (size_t i = 0; i < maes.size(); ++i) {
            maeTotal += maes[i];
            mseTotal += mses[i];
        }
        maeTotal /= maes.size();
        mseTotal /= mses.size();

        std::cout << "\n=== Évaluation finale ===\n";
        std::cout << "Nombre d'images évaluées : " << maes.size() << std::endl;
        std::cout << "MAE global : " << maeTotal << std::endl;
        std::cout << "MSE global : " << mseTotal << std::endl;

        // Sauvegarde CSV
        std::ofstream out("data/test_set_evaluated.csv");
        out << "image,steps,difficulty,mae,mse\n";
        for (size_t i = 0; i < validSamples.size(); ++i) {
            auto [name, trueCount, diff] = validSamples[i];
            out << name << "," << trueCount << "," << diff << "," << maes[i] << "," << mses[i] << "\n";
        }
        out.close();
        std::cout << "Résultats sauvegardés dans data/test_set_evaluated.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
