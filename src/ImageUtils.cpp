#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/ImageUtils.hpp"
#include "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/include/constants.hpp"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <numeric>


cv::Mat ImageUtils::loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Erreur : Impossible de charger l'image à partir de " + path);
    }
    return image;
}

void ImageUtils::displayImage(const std::string& windowName, const cv::Mat& image) {
    cv::imshow(windowName, image);
}

cv::Mat ImageUtils::convertToGrayscale(const cv::Mat& image) {
    cv::Mat grayscaleImage;
    cv::cvtColor(image, grayscaleImage, cv::COLOR_BGR2GRAY);
    return grayscaleImage;
}

cv::Mat ImageUtils::applyGaussianBlur(const cv::Mat& image, int kernelSize) {
    if (kernelSize <= 0) {
        throw std::invalid_argument("Erreur : kernelSize doit être un entier strictement positif.");
    }
    if (kernelSize % 2 == 0) {
        kernelSize += 1; 
        std::cout << "Attention : kernelSize était pair, il a été corrigé à " << kernelSize << std::endl;
    }

    cv::Mat blurredImage;
    cv::GaussianBlur(image, blurredImage, cv::Size(kernelSize, kernelSize), 0);
    return blurredImage;
}
cv::Mat ImageUtils::applyOtsuThreshold(const cv::Mat& image) {
    if (image.channels() > 1) {
        throw std::invalid_argument("Erreur : L'image d'entrée pour le seuillage d'Otsu doit être en niveaux de gris.");
    }

    cv::Mat thresholdedImage;
    //cv::threshold(image, thresholdedImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::threshold(image, thresholdedImage, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    return thresholdedImage;
}


cv::Mat ImageUtils::applyCanny(const cv::Mat& image, double threshold1, double threshold2) {
    cv::Mat edges;
    cv::Canny(image, edges, threshold1, threshold2);
    return edges;
}

cv::Mat ImageUtils::applyDilation(const cv::Mat& image, int kernelSize) {
    cv::Mat dilatedImage;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(kernelSize, kernelSize));
    
    cv::dilate(image, dilatedImage, kernel);
    
    return dilatedImage;
}

cv::Mat ImageUtils::applyOpening(const cv::Mat& image, int kernelSize) {
    cv::Mat openedImage;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

    cv::morphologyEx(image, openedImage, cv::MORPH_OPEN, kernel);

    return openedImage;
}

cv::Mat ImageUtils::applyErosion(const cv::Mat& image, int kernelSize) {
    cv::Mat erodedImage;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));

    cv::erode(image, erodedImage, kernel);

    return erodedImage;
}



cv::Mat ImageUtils::applyHoughTransform(const cv::Mat& edges, std::vector<cv::Vec4i>& detectedLines) {
    cv::Mat houghImage;
    cv::cvtColor(edges, houghImage, cv::COLOR_GRAY2BGR); 

    cv::HoughLinesP(edges, detectedLines, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);
    
    std::cout << "Lignes détectées (x1, y1, x2, y2) : " << std::endl;
    for (size_t i = 0; i < detectedLines.size(); i++) {
        cv::Vec4i l = detectedLines[i];
        cv::line(houghImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        std::cout << "Ligne " << i + 1 << ": (" << l[0] << ", " << l[1] << ") -> (" << l[2] << ", " << l[3] << ")" << std::endl;
    }

    return houghImage;
}

std::vector<cv::Vec4i> ImageUtils::mergeCloseLines(const std::vector<cv::Vec4i>& lines, double mergeThreshold) {
    if (lines.empty()) return {};

    std::vector<cv::Vec4i> mergedLines;
    std::vector<cv::Vec4i> sortedLines = lines;

    // Trier les lignes par leur position verticale (y moyen)
    std::sort(sortedLines.begin(), sortedLines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return ((a[1] + a[3]) / 2.0) < ((b[1] + b[3]) / 2.0);
    });

    for (size_t i = 0; i < sortedLines.size(); i++) {
        cv::Vec4i currentLine = sortedLines[i];
        double yCurrent = (currentLine[1] + currentLine[3]) / 2.0;
        bool merged = false;

        // Vérifier s'il existe déjà une ligne très proche en Y
        for (auto& mergedLine : mergedLines) {
            double yMerged = (mergedLine[1] + mergedLine[3]) / 2.0;
            if (std::abs(yCurrent - yMerged) < mergeThreshold) {
                // Fusionner : prendre la ligne la plus longue
                mergedLine[0] = std::min(mergedLine[0], currentLine[0]);
                mergedLine[1] = std::min(mergedLine[1], currentLine[1]);
                mergedLine[2] = std::max(mergedLine[2], currentLine[2]);
                mergedLine[3] = std::max(mergedLine[3], currentLine[3]);
                merged = true;
                break;
            }
        }

        // Ajouter la ligne si elle ne peut pas être fusionnée
        if (!merged) {
            mergedLines.push_back(currentLine);
        }
    }

    return mergedLines;
}


cv::Point2f computeIntersection(cv::Vec4i line1, cv::Vec4i line2) {
    float x1 = line1[0], y1 = line1[1], x2 = line1[2], y2 = line1[3];
    float x3 = line2[0], y3 = line2[1], x4 = line2[2], y4 = line2[3];
    // calcul des coefficients des droites (y = ax + b)
    float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (denom == 0) return cv::Point2f(-1, -1); // lignes parallèles

    float px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
    float py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
    
    return cv::Point2f(px, py);
}

cv::Mat ImageUtils::computeVanishingPoints(const std::vector<cv::Vec4i>& lines, cv::Mat& image) {
    cv::Mat outputImage = image.clone();
    std::vector<cv::Point2f> vanishingPoints;

    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            cv::Point2f intersection = computeIntersection(lines[i], lines[j]);

            if (intersection.x >= 0 && intersection.y >= 0 && intersection.x < image.cols && intersection.y < image.rows) {
                vanishingPoints.push_back(intersection);
                cv::circle(outputImage, intersection, 5, cv::Scalar(0, 255, 0), -1); 
                cv::line(outputImage, cv::Point(lines[i][0], lines[i][1]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(outputImage, cv::Point(lines[i][2], lines[i][3]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                
                cv::line(outputImage, cv::Point(lines[j][0], lines[j][1]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
                cv::line(outputImage, cv::Point(lines[j][2], lines[j][3]), intersection, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
            }
        }
    }

    std::cout << "Points de fuite détectés : " << std::endl;
    for (const auto& vp : vanishingPoints) {
        std::cout << "Point : (" << vp.x << ", " << vp.y << ")" << std::endl;
    }

    return outputImage;
}

std::vector<cv::Vec4i> ImageUtils::filterHorizontalLines(const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Vec4i> horizontalLines;
    for (const auto& line : lines) {
        double dx = line[2] - line[0];
        double dy = line[3] - line[1];

        if (dx == 0) continue;
        double angle = std::atan2(std::abs(dy), std::abs(dx)) * 180.0 / CV_PI;

        if (angle < 10.0) {  // Seulement les lignes proches de l'horizontal
            horizontalLines.push_back(line);
        }
    }
    return horizontalLines;
}

cv::Mat ImageUtils::drawLabeledLines(const cv::Mat& inputImage, const std::vector<cv::Vec4i>& lines) {
    cv::Mat outputImage = inputImage.clone();
    if (outputImage.channels() == 1) { 
        cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = 0; i < lines.size(); i++) {
        cv::Point p1(lines[i][0], lines[i][1]);
        cv::Point p2(lines[i][2], lines[i][3]);

        cv::line(outputImage, p1, p2, cv::Scalar(0, 255, 0), 2);
        std::string label = "L" + std::to_string(i);
        cv::Point textPos = p1 + cv::Point(5, -5); 
        cv::putText(outputImage, label, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    return outputImage;
}

std::vector<cv::Vec4i> ImageUtils::filterRegularlySpacedLines(const std::vector<cv::Vec4i>& lines, double& avgSpacing) {
    if (lines.empty()) return {};

    std::vector<cv::Vec4i> sortedLines = lines;
    std::sort(sortedLines.begin(), sortedLines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return ((a[1] + a[3]) / 2.0) < ((b[1] + b[3]) / 2.0);
    });

    std::vector<cv::Vec4i> filteredLines;
    std::vector<double> spacings;

    for (size_t i = 1; i < sortedLines.size(); i++) {
        double prevY = (sortedLines[i - 1][1] + sortedLines[i - 1][3]) / 2.0;
        double currentY = (sortedLines[i][1] + sortedLines[i][3]) / 2.0;
        double spacing = std::abs(currentY - prevY);
        spacings.push_back(spacing);
    }

    if (!spacings.empty()) {
        avgSpacing = std::accumulate(spacings.begin(), spacings.end(), 0.0) / spacings.size();
    } else {
        avgSpacing = 0.0;
    }

    for (size_t i = 0; i < sortedLines.size(); i++) {
        double currentY = (sortedLines[i][1] + sortedLines[i][3]) / 2.0;
        double prevY = (i > 0) ? (sortedLines[i - 1][1] + sortedLines[i - 1][3]) / 2.0 : currentY;

        if (std::abs(currentY - prevY) < (avgSpacing * 1.5)) { 
            filteredLines.push_back(sortedLines[i]);
        }
    }
    return filteredLines;
}


std::vector<cv::Vec4i> ImageUtils::filterOutliersBasedOnSpacing(const std::vector<cv::Vec4i>& lines, double threshold) {
    std::vector<cv::Vec4i> filteredLines;

    std::vector<cv::Vec4i> sortedLines = lines;
    std::sort(sortedLines.begin(), sortedLines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return ((a[1] + a[3]) / 2.0) < ((b[1] + b[3]) / 2.0);
    });

    std::vector<double> spacings;
    for (size_t i = 1; i < sortedLines.size(); i++) {
        double prevY = (sortedLines[i - 1][1] + sortedLines[i - 1][3]) / 2.0;
        double currentY = (sortedLines[i][1] + sortedLines[i][3]) / 2.0;
        spacings.push_back(std::abs(currentY - prevY));
    }
    
    double avgSpacing = std::accumulate(spacings.begin(), spacings.end(), 0.0) / spacings.size();

    for (size_t i = 0; i < sortedLines.size(); i++) {
        double currentY = (sortedLines[i][1] + sortedLines[i][3]) / 2.0;
        double prevY = (i > 0) ? (sortedLines[i - 1][1] + sortedLines[i - 1][3]) / 2.0 : currentY;

        if (std::abs(currentY - prevY) < (avgSpacing * threshold)) {
            filteredLines.push_back(sortedLines[i]);
        }
    }

    return filteredLines;
}


cv::Mat ImageUtils::quantize(const cv::Mat& image, int numberOfLevels) {
    assert(numberOfLevels > 0);

    cv::Mat res;
    image.convertTo(res, CV_32F);  

    float step = 255.0f / numberOfLevels;

    res.forEach<float>([&](float &pixel, const int *position) {
        pixel = floor(pixel / step) * step + step / 2; 
    });

    res.convertTo(res, CV_8U);

    return res;
}

cv::Mat ImageUtils::detectConnectedComponents(const cv::Mat& binaryImage, std::vector<cv::Rect>& components) {
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(binaryImage, labels, stats, centroids, 8, CV_32S);

    cv::Mat output = binaryImage.clone();
    cv::cvtColor(output, output, cv::COLOR_GRAY2BGR);

    int minArea = 500; 
    int maxArea = 100000;
    int minHeight = 10;  
    int maxHeight = 200;  

    for (int i = 1; i < numComponents; i++) { 
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Ajustement des critères de filtrage
        if (area > minArea && area < maxArea && height > minHeight && height < maxHeight) {
            cv::rectangle(output, cv::Point(x, y), cv::Point(x + width, y + height), cv::Scalar(0, 255, 0), 2);
            components.push_back(cv::Rect(x, y, width, height));
        }
    }

    return output;
}


std::vector<cv::Vec4i> ImageUtils::filterHorizontalLinesByLength(const std::vector<cv::Vec4i>& lines, double minLength, double maxAngleDeviation) {
    std::vector<cv::Vec4i> filteredLines;
    
    for (const auto& line : lines) {
        double dx = line[2] - line[0];
        double dy = line[3] - line[1];
        double length = std::sqrt(dx * dx + dy * dy);
        double angle = std::atan2(std::abs(dy), std::abs(dx)) * 180.0 / CV_PI;

        if (length > minLength && angle < maxAngleDeviation) { 
            filteredLines.push_back(line);
        }
    }
    return filteredLines;
}

std::vector<cv::Vec4i> ImageUtils::fitStairModel(const std::vector<cv::Vec4i>& lines, double avgSpacing) {
    std::vector<cv::Vec4i> stairLines;
    
    for (size_t i = 0; i < lines.size(); i++) {
        double y = (lines[i][1] + lines[i][3]) / 2.0;
        double expectedY = i * avgSpacing;

        if (std::abs(y - expectedY) < avgSpacing / 2) { 
            stairLines.push_back(lines[i]);
        }
    }
    
    return stairLines;
}

std::vector<cv::Vec4i> ImageUtils::mergeCloseParallelLines(const std::vector<cv::Vec4i>& lines, double maxYDistance) {
    if (lines.empty()) return {};

    std::vector<cv::Vec4i> mergedLines;
    std::vector<bool> merged(lines.size(), false);

    for (size_t i = 0; i < lines.size(); i++) {
        if (merged[i]) continue;
        
        cv::Vec4i currentLine = lines[i];
        double yCurrent = (currentLine[1] + currentLine[3]) / 2.0;

        for (size_t j = i + 1; j < lines.size(); j++) {
            if (merged[j]) continue;

            cv::Vec4i candidateLine = lines[j];
            double yCandidate = (candidateLine[1] + candidateLine[3]) / 2.0;

            if (std::abs(yCurrent - yCandidate) < maxYDistance) {
                // Fusionner les lignes : prendre la plus longue
                currentLine[0] = std::min(currentLine[0], candidateLine[0]);
                currentLine[1] = std::min(currentLine[1], candidateLine[1]);
                currentLine[2] = std::max(currentLine[2], candidateLine[2]);
                currentLine[3] = std::max(currentLine[3], candidateLine[3]);

                merged[j] = true;
            }
        }

        mergedLines.push_back(currentLine);
    }

    return mergedLines;
}

std::vector<cv::Vec4i> ImageUtils::sortLinesByLength(const std::vector<cv::Vec4i>& lines) {
    std::vector<std::pair<cv::Vec4i, double>> lineLengths;

    // Calcul de la longueur de chaque ligne
    for (const auto& line : lines) {
        double length = std::sqrt(std::pow(line[2] - line[0], 2) + std::pow(line[3] - line[1], 2));
        lineLengths.push_back({line, length});
    }

    // Tri des lignes par longueur décroissante
    std::sort(lineLengths.begin(), lineLengths.end(), 
        [](const std::pair<cv::Vec4i, double>& a, const std::pair<cv::Vec4i, double>& b) {
            return a.second > b.second;  // Tri décroissant
        });

    // Extraction des lignes triées
    std::vector<cv::Vec4i> sortedLines;
    for (const auto& pair : lineLengths) {
        sortedLines.push_back(pair.first);
    }

    return sortedLines;
}

std::vector<cv::Vec4i> ImageUtils::filterShortLines(const std::vector<cv::Vec4i>& lines, double minLength) {
    std::vector<cv::Vec4i> filteredLines;
    for (const auto& line : lines) {
        double length = std::sqrt(std::pow(line[2] - line[0], 2) + std::pow(line[3] - line[1], 2));
        if (length >= minLength) {
            filteredLines.push_back(line);
        }
    }
    return filteredLines;
}

std::vector<cv::Vec4i> ImageUtils::mergeOverlappingLines(const std::vector<cv::Vec4i>& lines, double maxYDistance) {
    std::vector<cv::Vec4i> mergedLines;
    std::vector<bool> merged(lines.size(), false);

    for (size_t i = 0; i < lines.size(); i++) {
        if (merged[i]) continue;
        cv::Vec4i currentLine = lines[i];
        double yCurrent = (currentLine[1] + currentLine[3]) / 2.0;

        for (size_t j = i + 1; j < lines.size(); j++) {
            if (merged[j]) continue;
            cv::Vec4i candidateLine = lines[j];
            double yCandidate = (candidateLine[1] + candidateLine[3]) / 2.0;

            if (std::abs(yCurrent - yCandidate) < maxYDistance) {
                currentLine[0] = std::min(currentLine[0], candidateLine[0]);
                currentLine[1] = std::min(currentLine[1], candidateLine[1]);
                currentLine[2] = std::max(currentLine[2], candidateLine[2]);
                currentLine[3] = std::max(currentLine[3], candidateLine[3]);
                merged[j] = true;
            }
        }

        mergedLines.push_back(currentLine);
    }

    return mergedLines;
}


std::vector<cv::Vec4i> ImageUtils::filterIrregularlySpacedLines(const std::vector<cv::Vec4i>& lines, double expectedSpacing) {
    if (lines.empty()) return {};

    std::vector<cv::Vec4i> sortedLines = lines;
    std::sort(sortedLines.begin(), sortedLines.end(), [](const cv::Vec4i& a, const cv::Vec4i& b) {
        return ((a[1] + a[3]) / 2.0) < ((b[1] + b[3]) / 2.0);
    });

    std::vector<cv::Vec4i> filteredLines;
    filteredLines.push_back(sortedLines[0]);

    for (size_t i = 1; i < sortedLines.size(); i++) {
        double prevY = (sortedLines[i - 1][1] + sortedLines[i - 1][3]) / 2.0;
        double currentY = (sortedLines[i][1] + sortedLines[i][3]) / 2.0;
        double spacing = std::abs(currentY - prevY);

        if (spacing > expectedSpacing * 0.5 && spacing < expectedSpacing * 1.5) {
            filteredLines.push_back(sortedLines[i]);
        }
    }

    return filteredLines;
}


cv::Mat ImageUtils::equalizeHistogram(const cv::Mat& image) {
    if (image.channels() != 1) {
        throw std::invalid_argument("L'image doit être en niveaux de gris.");
    }

    cv::Mat equalizedImage;
    cv::equalizeHist(image, equalizedImage);

    return equalizedImage;
}

void ImageUtils::computeHorizontalProjectionHistogram(const cv::Mat& binaryImage) {
    if (binaryImage.empty() || binaryImage.channels() != 1) {
        throw std::invalid_argument("L'image doit être binaire (niveaux de gris).");
    }

    std::vector<int> projection(binaryImage.rows, 0);

    for (int y = 0; y < binaryImage.rows; ++y) {
        projection[y] = cv::countNonZero(binaryImage.row(y));
    }

    int width = 400;  
    int height = binaryImage.rows;
    cv::Mat histogram(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    double maxVal = *std::max_element(projection.begin(), projection.end());

    for (int y = 0; y < height; ++y) {
        int barLength = static_cast<int>((projection[y] / maxVal) * (width - 10));
        cv::line(histogram, cv::Point(0, y), cv::Point(barLength, y), cv::Scalar(255, 255, 255), 1);
    }

    cv::flip(histogram, histogram, 0);
    cv::imshow("Histogramme de Projection Horizontale", histogram);
    cv::waitKey(0);
}

cv::Mat ImageUtils::adjustContrastGamma(const cv::Mat& image, double gamma) {
    cv::Mat result;
    cv::Mat imgFloat;
    image.convertTo(imgFloat, CV_32F, 1.0 / 255.0);
    cv::pow(imgFloat, gamma, imgFloat);
    imgFloat.convertTo(result, CV_8U, 255.0);
    return result;
}

std::pair<cv::Mat, int> ImageUtils::detectTransitionsAndCountPairs(const cv::Mat& image, int xCoord) {
    cv::Mat outputImage = image.clone();
    if (outputImage.channels() == 1) {
        cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
    }

    int rows = image.rows;
    cv::line(outputImage, cv::Point(xCoord, 0), cv::Point(xCoord, rows), cv::Scalar(0, 255, 0), 1);

    int numPairs = 0;
    bool lastWasRed = false; 

    for (int y = rows - 2; y >= 0; --y) {
        uchar currentPixel = image.at<uchar>(y, xCoord);
        uchar nextPixel = image.at<uchar>(y + 1, xCoord);

        if (currentPixel != nextPixel) {
            cv::Scalar color;
            bool isRed = false;

            if (currentPixel < nextPixel) { 
                color = cv::Scalar(0, 0, 255);
                isRed = true;
            } else { 
                color = cv::Scalar(255, 0, 0);
                isRed = false;
            }

            cv::drawMarker(outputImage, cv::Point(xCoord, y), color, cv::MARKER_CROSS, 15, 1);

            if (!isRed && lastWasRed) {  
                numPairs++;
            }

            lastWasRed = isRed; 
        }
    }

    return {outputImage, numPairs};
}



int ImageUtils::estimateStepCount(const std::vector<int>& transitions) {
    std::map<int, int> freq;
    for (int t : transitions) {
        freq[t]++;
    }

    int maxCount = 0;
    int probableSteps = -1;
    for (const auto& [value, count] : freq) {
        if (count > maxCount) {
            maxCount = count;
            probableSteps = value;
        }
    }

    return probableSteps;
}



std::pair<cv::Mat, std::vector<int>> ImageUtils::scanImageForStepPatterns(const cv::Mat& image, int stride) {
    cv::Mat outputImage = image.clone();
    
    if (outputImage.channels() == 1) {
        cv::cvtColor(outputImage, outputImage, cv::COLOR_GRAY2BGR);
    }

    std::vector<int> transitionCounts;

    for (int x = 0; x < image.cols; x += stride) {  
        auto [annotatedColumn, numPairs] = ImageUtils::detectTransitionsAndCountPairs(image, x);
        
        transitionCounts.push_back(numPairs);

        cv::line(outputImage, cv::Point(x, 0), cv::Point(x, image.rows), cv::Scalar(0, 255, 0), 1);

        for (int y = 0; y < annotatedColumn.rows; ++y) {
            cv::Vec3b pixel = annotatedColumn.at<cv::Vec3b>(y, x);
            if (pixel == cv::Vec3b(0, 0, 255) || pixel == cv::Vec3b(255, 0, 0)) {
                cv::drawMarker(outputImage, cv::Point(x, y), pixel, cv::MARKER_CROSS, 10, 1);
            }
        }
    }

    return {outputImage, transitionCounts};
}
