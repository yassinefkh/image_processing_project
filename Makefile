CXX = g++


CXXFLAGS = -std=c++17 -Wall

ifdef CONDA_PREFIX
    INCLUDES = -I$(CONDA_PREFIX)/include/opencv4
    LIBS = -L$(CONDA_PREFIX)/lib -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_ximgproc -lopencv_photo -lopencv_plot
else
    INCLUDES = $(shell pkg-config --cflags opencv4)
    LIBS = $(shell pkg-config --libs opencv4)
endif

TARGET = stair_detector

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

clean:
	rm -rf $(TARGET) $(OBJ_DIR)
