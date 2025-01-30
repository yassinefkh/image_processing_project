# Compilateur
CXX = g++

# Options de compilation
CXXFLAGS = -std=c++11 -Wall

# Vérifier si Conda est activé, sinon utiliser `pkg-config`
ifdef CONDA_PREFIX
    INCLUDES = -I$(CONDA_PREFIX)/include/opencv4
    LIBS = -L$(CONDA_PREFIX)/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc
else
    # Utilisation de `pkg-config` pour récupérer les bons chemins
    INCLUDES = $(shell pkg-config --cflags opencv4)
    LIBS = $(shell pkg-config --libs opencv4)
endif

# Nom de l'exécutable
TARGET = test

# Répertoires
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

# Fichiers source et objets
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Règle par défaut
all: $(TARGET)

# Création de l'exécutable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

# Compilation des fichiers sources
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Création du dossier obj s'il n'existe pas
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Nettoyage des fichiers générés
clean:
	rm -rf $(TARGET) $(OBJ_DIR)
