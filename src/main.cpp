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
#include "/home/augustepl/Desktop/MASTER/S2/ANALYSE_IMAGE/PROJET/image_processing_project/include/ImageUtils.hpp"

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
std::vector<std::tuple<std::string, int, std::string, std::string>> loadTestSet(const std::string& csvPath) {
    std::vector<std::tuple<std::string, int, std::string, std::string>> data;
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
        std::string imageName, stepsStr, difficulty, category;
        if (std::getline(ss, imageName, ',') && 
            std::getline(ss, stepsStr, ',') && 
            std::getline(ss, difficulty, ',') &&
            std::getline(ss, category)) {
            try {
                int steps = std::stoi(stepsStr);
                data.emplace_back(imageName, steps, difficulty, category);
            } catch (...) {
                std::cerr << "Erreur : Problème de parsing sur " << imageName << std::endl;
            }
        }
    }
    file.close();
    return data;
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
                std::cout << "Added ground truth: " << imageName << " -> " << steps << std::endl;
            } catch (...) {
                std::cerr << "Erreur : Conversion invalide dans le CSV (" << stepsStr << ")." << std::endl;
            }
        }
    }
    file.close();
    std::cout << "Loaded " << groundTruth.size() << " ground truth entries" << std::endl;
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
        std::string testCsvPath = "data/test_set.csv";
        std::string outputDir = "results";

        if (!fs::exists(outputDir)) {
            fs::create_directory(outputDir);
        }

    
        auto groundTruth = loadGroundTruth(csvPath);
        auto testSet = loadTestSet(testCsvPath);
        
        if (groundTruth.empty() && testSet.empty()) {
            std::cerr << "Erreur : Ground truth et test set vides." << std::endl;
            return 1;
        }

        // Create results CSV file
        std::ofstream resultsFile(outputDir + "/results.csv");
        if (resultsFile.is_open()) {
            resultsFile << "Image,TrueSteps,DetectedSteps,Method,Error,Difficulty,Category\n";
        }

        std::vector<int> detectedSteps;
        std::vector<int> trueSteps;
        std::vector<double> maes;
        std::vector<double> mses;
        std::vector<std::tuple<std::string, int, std::string, std::string>> validSamples;

        if (!testSet.empty()) {
            std::cout << "Using test set with " << testSet.size() << " images" << std::endl;
            
            for (const auto& entry : testSet) {
                std::string imageName = std::get<0>(entry);
                int trueCount = std::get<1>(entry);
                std::string difficulty = std::get<2>(entry);
                std::string category = std::get<3>(entry);
                
                std::string nameWithoutExt = removeExtension(imageName);
                std::string imagePath;
                std::string depthPath;

                // Try different extensions for the input image
                const std::vector<std::string> extensions = {".jpg", ".jpeg", ".png", ".bmp"};
                bool imageFound = false;

                for (const auto& ext : extensions) {
                    if (fs::exists(imgFolder + "/" + nameWithoutExt + ext)) {
                        imagePath = imgFolder + "/" + nameWithoutExt + ext;
                        imageFound = true;
                        break;
                    }
                }
                
    
                if (!imageFound && fs::exists(imgFolder + "/" + imageName)) {
                    imagePath = imgFolder + "/" + imageName;
                    imageFound = true;
                }

                if (!imageFound) {
                    std::cerr << "Image introuvable pour " << imageName << std::endl;
                    continue;
                }

                
                bool depthFound = false;
                for (const auto& ext : extensions) {
                    if (fs::exists(depthFolder + "/" + nameWithoutExt + "_depth" + ext)) {
                        depthPath = depthFolder + "/" + nameWithoutExt + "_depth" + ext;
                        depthFound = true;
                        break;
                    }
                }

                if (!depthFound) {
                    std::cerr << "Depth map introuvable pour " << imageName << "." << std::endl;
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
                std::cout << "Profil de profondeur exporté dans profil.csv" << std::endl;

                int detected = executePythonScript();
                std::string methodUsed = "PCA";

            
                if (detected < 3) {
                    std::cout << "Moins de 3 marches détectées, essai de techniques alternatives : " << std::endl;
                    
                    // vertical profile
                    auto verticalProfile = ImageUtils::extractVerticalProfile(depthMap);
                    ImageUtils::exportProfile(verticalProfile, "profil.csv");
                    int verticalSteps = executePythonScript();
                    std::cout << "Profil vertical: " << verticalSteps << " marches" << std::endl;
                    
                    // rotated profiles 
                    int bestRotatedSteps = 0;
                    double bestAngle = 0;
                    
                    for (int angle = -90; angle <= 90; angle += 10) {
                        if (angle == 0) continue; // Skip 0 same as vertical
                        
                        auto rotatedProfile = ImageUtils::extractRotatedProfile(depthMap, angle);
                        ImageUtils::exportProfile(rotatedProfile, "profil.csv");
                        int rotatedSteps = executePythonScript();
                        std::cout << "Rotation " << angle << "°: " << rotatedSteps << " marches" << std::endl;
                        
                        if (rotatedSteps > bestRotatedSteps) {
                            bestRotatedSteps = rotatedSteps;
                            bestAngle = angle;
                        }
                    }
                    
                    
                    
                    if (verticalSteps > detected && verticalSteps >= bestRotatedSteps) {
                        detected = verticalSteps;
                        methodUsed = "Vertical";
                        std::cout << "Selection: profil vertical (" << detected << " marches)" << std::endl;
                    }
                    else if (bestRotatedSteps > detected) {
                        detected = bestRotatedSteps;
                        methodUsed = "Rotation " + std::to_string(static_cast<int>(bestAngle)) + "°";
                        std::cout << "Selection: profil rotation " << bestAngle << "° (" << detected << " marches)" << std::endl;
                    }
                }

                if (detected < 0) {
                    std::cerr << "Détection échouée pour " << imageName << "." << std::endl;
                    continue;
                }
/*
                // Create visualization
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
                std::string resultText = "Detected: " + std::to_string(detected) + 
                                        " | Truth: " + std::to_string(trueCount) +
                                        " | Method: " + methodUsed;
                
                cv::putText(visualization, resultText, cv::Point(10, 30), 
                          cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
                
                // Save visualization
                std::string visPath = outputDir + "/" + nameWithoutExt + "_detection.jpg";
                cv::imwrite(visPath, visualization);
*/

                double mae = std::abs(detected - trueCount);
                double mse = std::pow(detected - trueCount, 2);
                detectedSteps.push_back(detected);
                trueSteps.push_back(trueCount);
                maes.push_back(mae);
                mses.push_back(mse);
                validSamples.emplace_back(imageName, trueCount, difficulty, category);

                // Write to results CSV
                if (resultsFile.is_open()) {
                    resultsFile << imageName << ","
                              << trueCount << ","
                              << detected << ","
                              << methodUsed << ","
                              << mae << ","
                              << difficulty << ","
                              << category << "\n";
                }

                std::cout << ">> Image: " << imageName
                          << " | Détecté: " << detected
                          << " | Vérité terrain: " << trueCount
                          << " | MAE: " << mae
                          << " | MSE: " << mse 
                          << " | Méthode: " << methodUsed
                          << std::endl;
            }
        } else {
            
            std::cout << "No test set found. Using all images from ground truth." << std::endl;
        }

        if (resultsFile.is_open()) {
            resultsFile.close();
        }

        if (detectedSteps.empty()) {
            std::cerr << "Aucune détection valide." << std::endl;
            return 1;
        }

        double maeTotal = 0.0;
        double mseTotal = 0.0;
        int exactMatches = 0;
        
        for (size_t i = 0; i < detectedSteps.size(); i++) {
            maeTotal += maes[i];
            mseTotal += mses[i];
            if (detectedSteps[i] == trueSteps[i]) {
                exactMatches++;
            }
        }
        
        maeTotal /= detectedSteps.size();
        mseTotal /= detectedSteps.size();
        double accuracy = 100.0 * exactMatches / detectedSteps.size();

        // Calculate MAE per difficulty level
        std::map<std::string, std::vector<double>> difficultyMaes;
        // Calculate MAE per category
        std::map<std::string, std::vector<double>> categoryMaes;

        for (size_t i = 0; i < validSamples.size(); ++i) {
            const auto& sample = validSamples[i];
            std::string diff = std::get<2>(sample);
            std::string cat = std::get<3>(sample);
            difficultyMaes[diff].push_back(maes[i]);
            categoryMaes[cat].push_back(maes[i]);
        }

        std::cout << "\n=== Évaluation finale ===\n";
        std::cout << "Images traitées: " << detectedSteps.size() << std::endl;
        std::cout << "Prédictions exactes: " << exactMatches << " (" << accuracy << "%)" << std::endl;
        std::cout << "MAE global: " << maeTotal << std::endl;
        std::cout << "MSE global: " << mseTotal << std::endl;
        
        std::cout << "\n=== MAE par niveau de difficulté ===\n";
        for (const auto& [difficulty, errors] : difficultyMaes) {
            double avgMae = 0.0;
            for (double mae : errors) {
                avgMae += mae;
            }
            avgMae /= errors.size();
            std::cout << "Difficulté " << difficulty << " (" << errors.size() << " images): MAE = " << avgMae << std::endl;
        }

        std::cout << "\n=== MAE par catégorie ===\n";
        for (const auto& [category, errors] : categoryMaes) {
            double avgMae = 0.0;
            for (double mae : errors) {
                avgMae += mae;
            }
            avgMae /= errors.size();
            std::cout << "Catégorie " << category << " (" << errors.size() << " images): MAE = " << avgMae << std::endl;
        }

        // Save metrics to file
        std::ofstream metricsFile(outputDir + "/metrics.txt");
        if (metricsFile.is_open()) {
            metricsFile << "Images traitées: " << detectedSteps.size() << std::endl;
            metricsFile << "Prédictions exactes: " << exactMatches << " (" << accuracy << "%)" << std::endl;
            metricsFile << "MAE global: " << maeTotal << std::endl;
            metricsFile << "MSE global: " << mseTotal << std::endl;
            
            metricsFile << "\n=== MAE par niveau de difficulté ===\n";
            for (const auto& [difficulty, errors] : difficultyMaes) {
                double avgMae = 0.0;
                for (double mae : errors) {
                    avgMae += mae;
                }
                avgMae /= errors.size();
                metricsFile << "Difficulté " << difficulty << " (" << errors.size() << " images): MAE = " << avgMae << std::endl;
            }
            
            metricsFile << "\n=== MAE par catégorie ===\n";
            for (const auto& [category, errors] : categoryMaes) {
                double avgMae = 0.0;
                for (double mae : errors) {
                    avgMae += mae;
                }
                avgMae /= errors.size();
                metricsFile << "Catégorie " << category << " (" << errors.size() << " images): MAE = " << avgMae << std::endl;
            }
            
        
            std::map<std::string, std::vector<double>> difficultyMses;
            std::map<std::string, std::vector<double>> categoryMses;
            
            for (size_t i = 0; i < validSamples.size(); ++i) {
                const auto& sample = validSamples[i];
                std::string diff = std::get<2>(sample);
                std::string cat = std::get<3>(sample);
                difficultyMses[diff].push_back(mses[i]);
                categoryMses[cat].push_back(mses[i]);
            }
            
            metricsFile << "\n=== MSE par niveau de difficulté ===\n";
            for (const auto& [difficulty, errors] : difficultyMses) {
                double avgMse = 0.0;
                for (double mse : errors) {
                    avgMse += mse;
                }
                avgMse /= errors.size();
                metricsFile << "Difficulté " << difficulty << " (" << errors.size() << " images): MSE = " << avgMse << std::endl;
            }
            
            metricsFile << "\n=== MSE par catégorie ===\n";
            for (const auto& [category, errors] : categoryMses) {
                double avgMse = 0.0;
                for (double mse : errors) {
                    avgMse += mse;
                }
                avgMse /= errors.size();
                metricsFile << "Catégorie " << category << " (" << errors.size() << " images): MSE = " << avgMse << std::endl;
            }
            
            metricsFile.close();
        }

        // Sauvegarde CSV détaillé
        std::ofstream out("data/test_set_evaluated.csv");
        out << "image,steps,difficulty,category,mae,mse\n";
        for (size_t i = 0; i < validSamples.size(); ++i) {
            const auto& sample = validSamples[i];
            std::string name = std::get<0>(sample);
            int trueCount = std::get<1>(sample);
            std::string diff = std::get<2>(sample);
            std::string cat = std::get<3>(sample);
            out << name << "," << trueCount << "," << diff << "," << cat << "," << maes[i] << "," << mses[i] << "\n";
        }
        out.close();
        std::cout << "Résultats sauvegardés dans data/test_set_evaluated.csv\n";

    } catch (const std::exception& e) {
        std::cerr << "Erreur: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}