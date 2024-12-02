#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <numeric>
#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
using namespace std;

// Константы
const int POLYNOMIAL_ORDER = 4; // Порядок многочлена
const int POPULATION_SIZE = 1000; // Размер популяции
const int GENERATIONS = 1000; // Максимальное число поколений
const double BASE_MUTATION_RATE = 0.9; // Начальная вероятность мутации
const double TOLERANCE = 1e-6; // Порог ошибки

// Генерация случайных данных
pair<vector<double>, vector<double>> generateData(int points = 500) {
    thrust::random::default_random_engine rng;  // Генератор случайных чисел
    thrust::random::uniform_real_distribution<> dis(-10.0, 10.0);  // Диапазон для x
    thrust::random::uniform_real_distribution<> coef_dis(-100.0, 100.0);  // Диапазон для коэффициентов

    vector<double> coefficients(POLYNOMIAL_ORDER + 1);
    // Генерация случайных коэффициентов полинома
    for (int i = 0; i <= POLYNOMIAL_ORDER; ++i) {
        coefficients[i] = coef_dis(rng);
        cout << coefficients[i] << endl;
    }

    vector<double> x(points), y(points);
    // Генерация случайных точек x и вычисление соответствующих значений y
    for (int i = 0; i < points; ++i) {
        x[i] = dis(rng);
        y[i] = 0.0;
        for (int j = 0; j <= POLYNOMIAL_ORDER; ++j) {
            y[i] += coefficients[j] * pow(x[i], j);
        }
    }

    return { x, y };
}

// CUDA-ядро для вычисления приспособленности
__global__ void calculateFitness(double* population, double* x, double* y, double* fitness, int populationSize, int polynomialOrder, int points) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < populationSize) {
        double sumSquaredError = 0.0;
        for (int i = 0; i < points; ++i) {
            double approx = 0.0;
            for (int j = 0; j <= polynomialOrder; ++j) {
                approx += population[idx * (polynomialOrder + 1) + j] * pow(x[i], j);
            }
            double error = y[i] - approx;
            sumSquaredError += error * error;
        }
        fitness[idx] = sumSquaredError / points; // MSE: среднее по количеству точек
    }
}

// Оценка приспособленности на GPU
void gpuFitnessEvaluation(
    vector<vector<double>>& population,
    const vector<double>& x,
    const vector<double>& y,
    vector<double>& fitness
) {
    int populationSize = population.size();
    int polynomialOrder = POLYNOMIAL_ORDER;
    int points = x.size();

    // Размеры
    size_t populationSizeBytes = populationSize * (polynomialOrder + 1) * sizeof(double);
    size_t pointsBytes = points * sizeof(double);
    size_t fitnessBytes = populationSize * sizeof(double);

    // Указатели на устройстве
    double* d_population, * d_x, * d_y, * d_fitness;

    // Выделение памяти на устройстве
    cudaMalloc(&d_population, populationSizeBytes);
    cudaMalloc(&d_x, pointsBytes);
    cudaMalloc(&d_y, pointsBytes);
    cudaMalloc(&d_fitness, fitnessBytes);

    // Копирование данных на устройство
    vector<double> flatPopulation(populationSize * (polynomialOrder + 1));
    for (int i = 0; i < populationSize; ++i) {
        for (int j = 0; j <= polynomialOrder; ++j) {
            flatPopulation[i * (polynomialOrder + 1) + j] = population[i][j];
        }
    }
    cudaMemcpy(d_population, flatPopulation.data(), populationSizeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), pointsBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), pointsBytes, cudaMemcpyHostToDevice);

    // Конфигурация CUDA
    int threadsPerBlock = 256;
    int blocks = (populationSize + threadsPerBlock - 1) / threadsPerBlock;

    // Запуск ядра
    calculateFitness << <blocks, threadsPerBlock >> > (d_population, d_x, d_y, d_fitness, populationSize, polynomialOrder, points);

    // Копирование результатов обратно
    cudaMemcpy(fitness.data(), d_fitness, fitnessBytes, cudaMemcpyDeviceToHost);

    // Очистка памяти на устройстве
    cudaFree(d_population);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_fitness);
}

// Турнирный отбор
int tournamentSelection(const vector<double>& fitnessScores) {
    thrust::random::default_random_engine rng;
    thrust::random::uniform_int_distribution<> dis(0, POPULATION_SIZE - 1);
    int bestIndex = dis(rng);

    for (int i = 1; i < 5; ++i) {
        int competitor = dis(rng);
        if (fitnessScores[competitor] < fitnessScores[bestIndex]) {
            bestIndex = competitor;
        }
    }
    return bestIndex;
}

// Кроссовер
vector<double> crossover(const vector<double>& parent1, const vector<double>& parent2, double crossoverRate = 0.7) {
    vector<double> child(POLYNOMIAL_ORDER + 1);
    thrust::random::default_random_engine rng;
    thrust::random::uniform_real_distribution<> prob(0, 1);

    for (size_t i = 0; i < child.size(); ++i) {
        // С вероятностью crossoverRate выбираем ген из второго родителя
        if (prob(rng) < crossoverRate) {
            child[i] = parent2[i];
        }
        else {
            child[i] = parent1[i];
        }
    }
    return child;
}

// Основной генетический алгоритм
vector<double> geneticAlgorithm(const vector<double>& x, const vector<double>& y) {
    thrust::random::default_random_engine rng;
    thrust::random::uniform_real_distribution<> coef_dis(-1, 1);

    // Инициализация популяции
    vector<vector<double>> population(POPULATION_SIZE, vector<double>(POLYNOMIAL_ORDER + 1));
    for (auto& individual : population) {
        for (auto& gene : individual) {
            gene = coef_dis(rng);
        }
    }

    double mutationRate = 0;

    for (int generation = 0; generation < GENERATIONS; ++generation) {
        // Вычисление приспособленности на GPU
        vector<double> fitnessScores(POPULATION_SIZE);
        gpuFitnessEvaluation(population, x, y, fitnessScores);

        // Сортировка по приспособленности
        vector<int> indices(POPULATION_SIZE);
        iota(indices.begin(), indices.end(), 0);
        sort(indices.begin(), indices.end(), [&fitnessScores](int i, int j) {
            return fitnessScores[i] < fitnessScores[j];
            });

        cout << "Generation " << generation << ": Best fitness = " << fitnessScores[indices[0]] << endl;

        // Проверка условия завершения
        if (fitnessScores[indices[0]] < TOLERANCE) {
            return population[indices[0]];
        }

        // Динамическое уменьшение диапазона мутации
        mutationRate = BASE_MUTATION_RATE * exp(-generation / (double)GENERATIONS);

   
        // Создание новой популяции
        vector<vector<double>> newPopulation;

        // Элитизм
        newPopulation.push_back(population[indices[0]]);
        newPopulation.push_back(population[indices[1]]);

        // Кроссовер и мутация
        while (newPopulation.size() < POPULATION_SIZE) {
            int parent1 = tournamentSelection(fitnessScores);
            int parent2 = tournamentSelection(fitnessScores);
            auto child = crossover(population[parent1], population[parent2]);

            // Динамическое изменение диапазона мутации
            double mutationRange = ((1.0 - (double)generation / GENERATIONS) * 10);
            for (size_t j = 0; j < child.size(); ++j) {
                if (coef_dis(rng) < mutationRate) {
                    child[j] += coef_dis(rng) * mutationRange;
                }
            }
            newPopulation.push_back(child);
        }

        population = newPopulation;
    }

    return population[0];
}

// Функция для вычисления ошибки между коэффициентами
double calculateError(const vector<double>& original, const vector<double>& solution) {
    double error = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        error += pow(original[i] - solution[i], 2);
    }
    return sqrt(error);
}

// Точка входа
int main() {
    auto data = generateData(); // Получаем пару вектора x и y
    vector<double> x = data.first;  // Извлекаем x
    vector<double> y = data.second; // Извлекаем y

    // Запуск генетического алгоритма
    vector<double> solution = geneticAlgorithm(x, y);
    cout << "Final Solution: ";
    for (const double& coeff : solution) {
        cout << coeff << " ";
    }
    cout << endl;
    return 0;
}
