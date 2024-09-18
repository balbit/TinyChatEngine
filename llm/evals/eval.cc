// eval.cpp
#include "eval.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
// Include any necessary libraries for JSON parsing

// Load the VizWiz dataset
void loadVizWizData(const std::string& dataset_path, std::vector<std::pair<std::string, std::string>>& data) {
    // Pseudo-code for loading data
    // Open and parse the dataset file (e.g., JSON or CSV)
    // Store each image path and question in the data vector

    // Example: assuming dataset_path is a text file with image path and question
    std::ifstream infile(dataset_path);
    std::string img_path, question;
    while (infile >> img_path >> question) {
        data.push_back(std::make_pair(img_path, question));
    }
}

// Evaluate the model on the dataset
void evaluateModel(const std::string& llama_m_path, const std::string& clip_m_path, const std::string& dataset_path, const struct opt_params& generation_config) {
    // Load models
    void* llama_model = loadLlamaModel(llama_m_path); // Pseudo-function
    void* clip_model = loadClipModel(clip_m_path); // Pseudo-function

    // Load the dataset
    std::vector<std::pair<std::string, std::string>> vizwiz_data;
    loadVizWizData(dataset_path, vizwiz_data);

    std::vector<std::string> predictions;

    // Iterate over the dataset
    for (const auto& sample : vizwiz_data) {
        std::string img_path = sample.first;
        std::string question = sample.second;

        // Call LLaVAGenerate function
        std::string prediction = LLaVAGenerate(
            llama_m_path,
            llama_model,
            clip_m_path,
            clip_model,
            VILA_FP32,
            question,
            img_path,
            generation_config,
            "models/llama_vocab.bin",
            true,
            false,
            true
        );

        predictions.push_back(prediction);
    }

    // Evaluate predictions (compute accuracy, etc.)
    // Compare predictions with ground truth and calculate accuracy

    // Clean up models if needed
}