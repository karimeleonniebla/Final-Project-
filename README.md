# Final Project 
# Purpose: The purpose of this project is to implement a simple version of Multi-Head Attention using basic C++ without relying on any external libraries. I wanted to understand how attention mechanisms work, so I built everything from scratch using simple functions like dot product, softmax, and matrix generation. The program creates random query, key, and value matrices, performs attention across multiple heads, and prints the results to the console. This helped me get a better grasp of how transformer models process information, and it was a great way for me to practice both C++ and machine learning fundamentals.

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

float dot_product(const vector<float>& a, const vector<float>& b) {
    float result = 0;
    for (size_t i = 0; i < a.size(); i++)
        result += a[i] * b[i];
    return result;
}

vector<float> softmax(const vector<float>& input) {
    vector<float> output(input.size());
    float sum_exp = 0;
    for (float val : input)
        sum_exp += exp(val);
    for (size_t i = 0; i < input.size(); i++)
        output[i] = exp(input[i]) / sum_exp;
    return output;
}

vector<float> scale_vector(const vector<float>& vec, float scalar) {
    vector<float> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++)
        result[i] = vec[i] * scalar;
    return result;
}

vector<float> scaled_dot_product_attention(const vector<vector<float>>& Q,
                                           const vector<vector<float>>& K,
                                           const vector<vector<float>>& V) {
    size_t seq_len = Q.size();
    vector<float> output(seq_len, 0.0f);

    for (size_t i = 0; i < seq_len; i++) {
        vector<float> scores;
        for (size_t j = 0; j < seq_len; j++) {
            float score = dot_product(Q[i], K[j]);
            scores.push_back(score / sqrt(Q[0].size()));
        }
        vector<float> attn_weights = softmax(scores);
        for (size_t j = 0; j < seq_len; j++) {
            for (size_t k = 0; k < V[0].size(); k++) {
                output[i] += attn_weights[j] * V[j][k];
            }
        }
    }
    return output;
}

vector<vector<float>> generate_random_matrix(int rows, int cols) {
    vector<vector<float>> matrix(rows, vector<float>(cols));
    for (auto& row : matrix)
        for (auto& val : row)
            val = static_cast<float>(rand()) / RAND_MAX;
    return matrix;
}

void multi_head_attention(int num_heads, int seq_len, int depth) {
    cout << "Multi-head Attention Test\n";
    for (int head = 0; head < num_heads; head++) {
        cout << "\nHead " << head + 1 << ":\n";
        auto Q = generate_random_matrix(seq_len, depth);
        auto K = generate_random_matrix(seq_len, depth);
        auto V = generate_random_matrix(seq_len, depth);

        auto output = scaled_dot_product_attention(Q, K, V);
        cout << "Output: ";
        for (float val : output)
            cout << val << " ";
        cout << endl;
    }
}

int main() {
    srand(time(0));
    int num_heads = 2;
    int seq_len = 4;
    int depth = 3;
    multi_head_attention(num_heads, seq_len, depth);
    return 0;
}

