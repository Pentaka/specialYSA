#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>

// Aktivasyon fonksiyonu: Tanjant Hiperbolik (tanh)
double activationFunction(double x) {
    return std::tanh(x);
}

// Aktivasyon fonksiyonunun türevi
double activationFunctionDerivative(double x) {
    double tanh_x = std::tanh(x);
    return 1.0 - tanh_x * tanh_x;
}

// YSA sınıfı
class NeuralNetwork {
private:
    int inputSize;
    int hiddenSize;
    int outputSize;
    std::vector<double> inputLayer;
    std::vector<double> hiddenLayer;
    std::vector<double> outputLayer;
    std::vector<std::vector<double>> weightsInputHidden;
    std::vector<std::vector<double>> weightsHiddenOutput;
    double learningRate; // Öğrenme hızı
    double bias;

public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;
        this->learningRate = learningRate;

        // Katman vektörlerini boyutlandırma
        inputLayer.resize(inputSize, 0.0);
        hiddenLayer.resize(hiddenSize, 0.0);
        outputLayer.resize(outputSize, 0.0);

        // Ağırlık matrislerini boyutlandırma ve başlangıç değerlerini rastgele atama
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        weightsInputHidden.resize(inputSize+1);
        for (int i = 0; i < inputSize+1; i++) {
            weightsInputHidden[i].resize(hiddenSize, 0.0);
        }

        weightsHiddenOutput.resize(hiddenSize+1);
        for (int i = 0; i < hiddenSize+1; i++) {
            weightsHiddenOutput[i].resize(outputSize, 0.0);

        }

        // Bağlantıları manuel olarak belirleme
        weightsInputHidden[0][0] = dist(gen);
        weightsInputHidden[0][1] = dist(gen);
        weightsInputHidden[0][2] = 0.0;
        weightsInputHidden[0][3] = dist(gen);
        weightsInputHidden[0][4] = 0.0;
        weightsInputHidden[0][5] = 0.0;
        weightsInputHidden[0][6] = 0.0;
        weightsInputHidden[0][7] = dist(gen);

        weightsInputHidden[1][0] = 0.0;
        weightsInputHidden[1][1] = dist(gen);
        weightsInputHidden[1][2] = dist(gen);
        weightsInputHidden[1][3] = 0.0;
        weightsInputHidden[1][4] = 0.0;
        weightsInputHidden[1][5] = dist(gen);
        weightsInputHidden[1][6] = 0.0;
        weightsInputHidden[1][7] = 0.0;

        weightsInputHidden[2][0] = 0.0;
        weightsInputHidden[2][1] = 0.0;
        weightsInputHidden[2][2] = 0.0;
        weightsInputHidden[2][3] = dist(gen);
        weightsInputHidden[2][4] = dist(gen);
        weightsInputHidden[2][5] = 0.0;
        weightsInputHidden[2][6] = 0.0;
        weightsInputHidden[2][7] = 0.0;

        weightsInputHidden[3][0] = 0.0;
        weightsInputHidden[3][1] = 0.0;
        weightsInputHidden[3][2] = dist(gen);
        weightsInputHidden[3][3] = 0.0;
        weightsInputHidden[3][4] = 0.0;
        weightsInputHidden[3][5] = dist(gen);
        weightsInputHidden[3][6] = dist(gen);
        weightsInputHidden[3][7] = 0.0;

        weightsInputHidden[4][0] = dist(gen);
        weightsInputHidden[4][1] = 0.0;
        weightsInputHidden[4][2] = 0.0;
        weightsInputHidden[4][3] = 0.0;
        weightsInputHidden[4][4] = dist(gen);
        weightsInputHidden[4][5] = 0.0;
        weightsInputHidden[4][6] = 0.0;
        weightsInputHidden[4][7] = dist(gen);

        weightsInputHidden[5][0] = dist(gen);
        weightsInputHidden[5][1] = dist(gen);
        weightsInputHidden[5][2] = dist(gen);
        weightsInputHidden[5][3] = dist(gen);
        weightsInputHidden[5][4] = dist(gen);
        weightsInputHidden[5][5] = dist(gen);
        weightsInputHidden[5][6] = dist(gen);
        weightsInputHidden[5][7] = dist(gen);

        for (int i=0; i< hiddenSize; i++)
        {
            for (int j=0; j<outputSize; j++)
            {
                weightsHiddenOutput[i][j] = dist(gen);
            }
        }



        bias = 1.0;
    }

    // Feed-forward işlemi
    void feedForward() {
        // Gizli katman hesaplaması
        for (int i = 0; i < hiddenSize; i++) {
            double sum = bias * weightsInputHidden[inputSize][i];
            for (int j = 0; j < inputSize; j++) {
                sum += inputLayer[j] * weightsInputHidden[j][i];
            }
            hiddenLayer[i] = activationFunction(sum);
        }

        // Çıkış katmanı hesaplaması
        for (int i = 0; i < outputSize; i++) {
            double sum = bias * weightsHiddenOutput[hiddenSize][i];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hiddenLayer[j] * weightsHiddenOutput[j][i];
            }
            outputLayer[i] = activationFunction(sum);
        }
    }

    // MSE hesaplama
    double calculateMSE(std::vector<double>& target) {
        double sumSquaredError = 0.0;
        for (int i = 0; i < outputSize; i++) {
            double error = target[i] - outputLayer[i];
            sumSquaredError += error * error;
        }
        return sumSquaredError / 2;
    }

    // Eğitim fonksiyonu
    void train(std::vector<std::vector<double>>& input, std::vector<std::vector<double>>& target, double targetMSE) {
        double mse = 1.0;
        int epoch = 0;

        while (mse > targetMSE && epoch < 3000) {

            mse = 0.0;
            for (int k = 0; k < input.size(); k++) {
                // Giriş değerlerini atama
                for (int i = 0; i < inputSize; i++) {
                    inputLayer[i] = input[k][i];
                }


                // Hedef değerleri atama
                std::vector<double> currentTarget = target[k];

                // Feed-forward işlemi
                feedForward();

                // MSE hesaplama
                mse += calculateMSE(currentTarget);

                //std::cout << "Current Target: ";
               // for (int i = 0; i < currentTarget.size(); i++) {
               // std::cout << currentTarget[i] << " ";

                // Geri yayılım algoritması
                std::vector<double> outputErrors(outputSize, 0.0);
                for (int i = 0; i < outputSize; i++) {
                    double error = currentTarget[i] - outputLayer[i];
                    outputErrors[i] = error * activationFunctionDerivative(outputLayer[i]);
                }

                std::vector<double> hiddenErrors(hiddenSize, 0.0);
                for (int i = 0; i < hiddenSize; i++) {
                    double error = 0.0;
                    for (int j = 0; j < outputSize; j++) {
                        error += outputErrors[j] * weightsHiddenOutput[i][j];
                    }
                    hiddenErrors[i] = error * activationFunctionDerivative(hiddenLayer[i]);
                }

                // Ağırlık güncellemesi
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
            if (weightsInputHidden[i][j] != 0.0) { // Sadece sıfır olmayan bağlantılar için güncelleme yap
                weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * inputLayer[i] * bias;
            }
        }
    }

for (int i = 0; i < hiddenSize; i++) {
    for (int j = 0; j < outputSize; j++) {
        weightsHiddenOutput[i][j] += learningRate * outputErrors[j] * hiddenLayer[i]* bias;
    }
}
            }

            mse /= input.size();
            epoch++;

            // Her epoch sonunda MSE değerini yazdırma
            std::cout << "Epoch: " << epoch << ", MSE: " << std::fixed << std::setprecision(6) << mse << std::endl;
        }

        // Son ağırlık değerlerini yazdırma
        std::cout << "Son ağırlık değerleri:" << std::endl;
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                std::cout << "input-hidden[" << i << "][" << j << "]: " << weightsInputHidden[i][j] << std::endl;
            }
        }

        for (int i = 0; i < hiddenSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                std::cout << "hidden-output[" << i << "][" << j << "]: " << weightsHiddenOutput[i][j] << std::endl;
            }
        }
    }

    void test(std::vector<double>& input, int outputSize) {
        // Giriş değerlerini atama
        for (int i = 0; i < inputSize; i++) {
            inputLayer[i] = input[i];
        }

        // Feed-forward işlemi
        feedForward();

        // Çıkış değerlerini yazdırma
        std::cout << "Cikis: ";
        for (int i = 0; i < outputSize; i++) {
            std::cout << outputLayer[i] << " ";
        }
        std::cout << std::endl;
    }
};


std::vector<double> convertToBinary(const std::vector<double>& input) {
    std::vector<double> binaryInput;

    for (const auto& value : input) {
        // Değeri binary olarak dönüştürme
        std::vector<double> binaryValue;
        int intValue = static_cast<int>(value);

        while (intValue > 0) {
            binaryValue.insert(binaryValue.begin(), intValue % 2);
            intValue /= 2;
        }

        // 5 boyutlu binary vektörü oluşturma
        while (binaryValue.size() < 5) {
            binaryValue.insert(binaryValue.begin(), 0);
        }

        // Binary değeri ekrana yazdırma
        std::cout << "Binary donusumu: ";
        for (const auto& bit : binaryValue) {
            std::cout << bit << " ";
        }
        std::cout << std::endl;

        binaryInput.insert(binaryInput.end(), binaryValue.begin(), binaryValue.end());
    }

    return binaryInput;
}

int main() {
    // Eğitim veri seti
std::vector<std::vector<double>> input = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
};
std::vector<std::vector<double>> target = {
    {-1, -1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1},
    {1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, 1, 1}
};
    // Giriş değerlerini 5 boyutlu binary olarak çevirme

    std::vector<std::vector<double>> binaryInput;
    for (const auto& inputData : input) {
        std::cout << "Giris: ";
        for (const auto& value : inputData) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        std::vector<double> binaryData = convertToBinary(inputData);

        std::cout << "Binary: ";
        for (size_t i = 0; i < binaryData.size(); i += 5) {
            for (size_t j = 0; j < 5; j++) {
                std::cout << binaryData[i + j];
            }
            if (i + 5 < binaryData.size()) {
                std::cout << ",";
            }
        }
        std::cout << std::endl;

        binaryInput.push_back(binaryData);
    }


    // Yapay Sinir Ağı modeli oluşturma ve eğitim
    int inputSize = 5;
    int hiddenSize = 8;
    int outputSize = 2;
    double learningRate = 0.1;
    NeuralNetwork neuralNetwork(inputSize, hiddenSize, outputSize, learningRate);
    double targetMSE = 0.001;
    neuralNetwork.train(binaryInput, target, targetMSE);

    // Test
    std::vector<double> testInput = {10, 13, 15, 17, 19};
    std::vector<double> binaryTestInput = convertToBinary(testInput);

    neuralNetwork.test(binaryTestInput, outputSize);

    return 0;
}
