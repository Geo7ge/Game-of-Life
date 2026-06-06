# Compilator și flag-uri
CXX = mpicxx
CXXFLAGS = -O3 -Wall -std=c++17

# Target principal
TARGET = game_of_life

# Surse și obiecte
SRCS = main.cpp simulator.cpp
OBJS = $(SRCS:.cpp=.o)

.PHONY: all clean run test_small test_large

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp simulator.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) *.pgm

# Rulează pe laptop folosind 4 procese, grilă mică (cu Glider configurat static dacă rulezi cu ultimul parametru 0)
run: all
	mpirun -np 4 ./$(TARGET) 50 50 20 0

# Test scalabilitate pe o grilă medie spre mare
test_small: all
	mpirun -np 2 ./$(TARGET) 2000 2000 100 1

test_large: all
	mpirun -np 4 ./$(TARGET) 10000 10000 50 1