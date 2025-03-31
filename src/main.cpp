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
#include "/home/augustepl/Desktop/MASTER/S2/ANALYSE_IMAGE/PROJET/image_processing_project/include/ImageUtils.hpp"

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
 * Utilise des techniques alternatives si moins de 3 marches sont détectées.
 */
int main() {
    try {
        std::string imgFolder = "data/img";
        std::string depthFolder = "data/depth";
        std::string csvPath = "data/annotations.csv";
        std::string outputDir = "results";

        if (!fs::exists(outputDir)) {
            fs::create_directory(outputDir);
        }

        auto groundTruth = loadGroundTruth(csvPath);
        if (groundTruth.empty()) {
            std::cerr << "Erreur : Ground truth vide." << std::endl;
            return 1;
        }

        std::vector<int> detectedSteps;
        std::vector<int> trueSteps;

    
        std::ofstream resultsFile(outputDir + "/results.csv");
        if (resultsFile.is_open()) {
            resultsFile << "Image,TrueSteps,DetectedSteps,Method,Error\n";
        }

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
                std::cout << "Profil de profondeur exporté dans profil.csv" << std::endl;

                int numDetected = executePythonScript();
                std::string methodUsed = "PCA";

                // If less than 3 steps detected, try alternative profiles
                if (numDetected < 3) {
                    std::cout << "Moins de 3 marches détectées, essai de techniques alternatives..." << std::endl;
                    
                    // Try vertical profile
                    auto verticalProfile = ImageUtils::extractVerticalProfile(depthMap);
                    ImageUtils::exportProfile(verticalProfile, "profil.csv");
                    int verticalSteps = executePythonScript();
                    std::cout << "Profil vertical: " << verticalSteps << " marches" << std::endl;
                    
                    // Try rotated profiles at different angles
                    int bestRotatedSteps = 0;
                    double bestAngle = 0;
                    
                    for (int angle = -90; angle <= 90; angle += 20) {
                        auto rotatedProfile = ImageUtils::extractRotatedProfile(depthMap, angle);
                        ImageUtils::exportProfile(rotatedProfile, "profil.csv");
                        int rotatedSteps = executePythonScript();
                        std::cout << "Rotation " << angle << "°: " << rotatedSteps << " marches" << std::endl;
                        
                        if (rotatedSteps > bestRotatedSteps) {
                            bestRotatedSteps = rotatedSteps;
                            bestAngle = angle;
                        }
                    }
                    
                    // Select the best result (highest step count)
                    if (verticalSteps > numDetected && verticalSteps >= bestRotatedSteps) {
                        numDetected = verticalSteps;
                        methodUsed = "Vertical";
                        std::cout << "Selection: profil vertical (" << numDetected << " marches)" << std::endl;
                    }
                    else if (bestRotatedSteps > numDetected) {
                        numDetected = bestRotatedSteps;
                        methodUsed = "Rotation " + std::to_string(static_cast<int>(bestAngle)) + "°";
                        std::cout << "Selection: profil rotation " << bestAngle << "° (" << numDetected << " marches)" << std::endl;
                    }
                }

                if (numDetected < 0) {
                    std::cerr << "Échec détection pour " << imageName << "." << std::endl;
                    continue;
                }

                // Save visualization
                cv::Mat visualization;
                cv::cvtColor(depthMap, visualization, cv::COLOR_GRAY2BGR);
                
                // Draw line and profile points on visualization
                cv::Point pt1, pt2;
                std::tie(pt1, pt2) = ImageUtils::computeLineEndpoints(mean, principalVector, depthMap.cols, depthMap.rows);
                cv::line(visualization, pt1, pt2, cv::Scalar(0, 255, 0), 2);
                
                for (const auto& point : profilePoints) {
                    cv::circle(visualization, point, 2, cv::Scalar(0, 0, 255), -1);
                }
                
                // Add text with results
                std::string resultText = "Détecté: " + std::to_string(numDetected) + 
                                      " | Réel: " + std::to_string(groundTruth[nameWithoutExt]) +
                                      " | Méthode: " + methodUsed;
                
                cv::putText(visualization, resultText, cv::Point(10, 30), 
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                
                // Save visualization image
                cv::imwrite(outputDir + "/" + nameWithoutExt + "_result.jpg", visualization);

                detectedSteps.push_back(numDetected);
                trueSteps.push_back(groundTruth[nameWithoutExt]);

                int error = std::abs(numDetected - groundTruth[nameWithoutExt]);
                
                // Write to results CSV
                if (resultsFile.is_open()) {
                    resultsFile << imageName << ","
                              << groundTruth[nameWithoutExt] << ","
                              << numDetected << ","
                              << methodUsed << ","
                              << error << "\n";
                }

                std::cout << ">> Image: " << imageName
                          << " | Détecté: " << numDetected
                          << " | Vérité terrain: " << groundTruth[nameWithoutExt]
                          << " | Méthode: " << methodUsed
                          << std::endl;
            }
        }

        if (resultsFile.is_open()) {
            resultsFile.close();
        }

        if (detectedSteps.empty()) {
            std::cerr << "Aucune détection valide." << std::endl;
            return 1;
        }

        double mse = 0.0, mae = 0.0;
        int exactMatches = 0;
        
        for (size_t i = 0; i < detectedSteps.size(); i++) {
            mse += std::pow(detectedSteps[i] - trueSteps[i], 2);
            mae += std::abs(detectedSteps[i] - trueSteps[i]);
            if (detectedSteps[i] == trueSteps[i]) {
                exactMatches++;
            }
        }
        
        mse /= detectedSteps.size();
        mae /= detectedSteps.size();
        double accuracy = 100.0 * exactMatches / detectedSteps.size();

        std::cout << "\n=== Résultats ===\n";
        std::cout << "Images traitées: " << detectedSteps.size() << std::endl;
        std::cout << "Prédictions exactes: " << exactMatches << " (" << accuracy << "%)" << std::endl;
        std::cout << "MSE: " << mse << std::endl;
        std::cout << "MAE: " << mae << std::endl;
        
        // Save metrics to file
        std::ofstream metricsFile(outputDir + "/metrics.txt");
        if (metricsFile.is_open()) {
            metricsFile << "Images traitées: " << detectedSteps.size() << std::endl;
            metricsFile << "Prédictions exactes: " << exactMatches << " (" << accuracy << "%)" << std::endl;
            metricsFile << "MSE: " << mse << std::endl;
            metricsFile << "MAE: " << mae << std::endl;
            metricsFile.close();
        }

    } catch (const std::exception& e) {
        std::cerr << "Erreur: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}