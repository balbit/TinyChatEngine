#ifndef EVAL_H
#define EVAL_H

#include <string>
#include <vector>

void loadVizWizData(const std::string& dataset_path, std::vector<std::pair<std::string, std::string>>& data);

// Function to evaluate the model on a given dataset
void evaluateModel(const std::string& llama_m_path, const std::string& clip_m_path, const std::string& dataset_path, const struct opt_params& generation_config);

#endif