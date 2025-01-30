#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include <string>

// chemin de l'image d'entrée
const std::string INPUT_IMAGE_PATH = "/Volumes/SSD/M1VMI/S2/image_processing/env/projet/data/input/img2.jpg";

// taille du noyau pour le flou gaussien (doit être impair)
const int GAUSSIAN_BLUR_KERNEL_SIZE = 15;

// seuils pour Canny
const int CANNY_THRESHOLD1 = 50;
const int CANNY_THRESHOLD2 = 150;

// paramètres pour la transformée de Hough
const double HOUGH_RHO = 0.001;             // Résolution en pixels
const double HOUGH_THETA = CV_PI / 180; // Résolution angulaire en radians
const int HOUGH_THRESHOLD = 50;         // Nombre minimum d'intersections pour détecter une ligne
const int HOUGH_MIN_LINE_LENGTH = 50;   // Longueur minimale pour une ligne
const int HOUGH_MAX_LINE_GAP = 300;      // Espace maximal entre segments pour fusionner en une ligne

const double ANGLE_THRESHOLD = 10.0;  // seuil en degrés pour filtrer les droites

#endif
