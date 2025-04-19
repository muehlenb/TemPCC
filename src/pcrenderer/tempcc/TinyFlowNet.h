// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "src/pch.h"

#pragma once

/**
 * A Scaled Dot Product Attention Layer used in our TinyFlowNet below.
 */
struct ScaledDotProductAttention : torch::nn::Module {
    torch::nn::Linear query, key, value;
    double scale;
    int input_size;

    ScaledDotProductAttention(int64_t input_size, int64_t embed_size) :
        query(register_module("query", torch::nn::Linear(input_size, embed_size))),
        key(register_module("key", torch::nn::Linear(input_size, embed_size))),
        value(register_module("value", torch::nn::Linear(input_size, embed_size))),
        scale(sqrt(embed_size)),
        input_size(input_size)
    {
        query->to(torch::kCUDA);
        key->to(torch::kCUDA);
        value->to(torch::kCUDA);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), -1, input_size});

        auto Q = query->forward(x);
        auto K = key->forward(x);
        auto V = value->forward(x);

        auto scores = torch::matmul(Q, K.transpose(-2, -1)) / scale;
        auto weights = torch::softmax(scores, -1);
        return torch::matmul(weights, V).view({x.size(0),-1});
    }
};

/**
 * Our TinyFlowNet for flow prediction of occluded points. For details, see paper.
 */
struct TinyFlowNet : torch::nn::Module {
    bool attention_enabled = false;

    TinyFlowNet()
        : spatial_att(std::make_shared<ScaledDotProductAttention>(6,6))
        , spatial_fc1(torch::nn::Linear(588, 588))
        , spatial_fc2(torch::nn::Linear(588, 384))
        , temporal_layer(torch::nn::GRU(torch::nn::GRUOptions(4, 32).batch_first(true)))
        , temporal_fc1(torch::nn::Linear(32, 128))
        , merged_fc(torch::nn::LinearOptions(384+128, 512))
        , merged_fc2(torch::nn::LinearOptions(512, 512))
        , merged_fc3(torch::nn::LinearOptions(512, 512))
        , merged_fc4(torch::nn::LinearOptions(512, 4))
    {
        register_module("spatial_fc1", spatial_fc1);
        register_module("spatial_att", spatial_att);
        register_module("spatial_fc2", spatial_fc2);

        register_module("gru", temporal_layer);
        register_module("fc_temporal", temporal_fc1);

        register_module("merged_fc", merged_fc);
        register_module("merged_fc2", merged_fc2);
        register_module("merged_fc3", merged_fc3);
        register_module("merged_fc4", merged_fc4);
    }

    torch::Tensor forward(torch::Tensor temporal_data, torch::Tensor spatial_data) {
        auto spatial_processed_att = torch::relu(spatial_att->forward(spatial_data));
        auto spatial_processed_con = torch::cat({spatial_data, spatial_processed_att}, 1);

        auto spatial_processed = torch::relu(spatial_fc1->forward(spatial_processed_con));

        spatial_processed = torch::relu(spatial_fc2->forward(spatial_processed));

        auto temporal_processed = std::get<0>(temporal_layer->forward(temporal_data)).index({torch::indexing::Slice(), -1});
        temporal_processed = torch::relu(temporal_fc1->forward(temporal_processed));

        auto concatenation = torch::cat({temporal_processed, spatial_processed}, 1);

        auto merged_result = torch::relu(merged_fc->forward(concatenation));
        auto merged_final = torch::relu(merged_fc2->forward(merged_result));
        merged_final = torch::relu(merged_fc3->forward(merged_final));
        return merged_fc4->forward(merged_final);
    }

    void saveNet(std::string name = "temp_net.pt"){
        torch::serialize::OutputArchive archive;
        save(archive);
        archive.save_to(CMAKE_SOURCE_DIR "/data/net/" + name);
    }

    void loadNet(std::string name = "temp_net.pt"){
        torch::serialize::InputArchive archive;
        archive.load_from(CMAKE_SOURCE_DIR "/data/net/" + name);
        load(archive);
    }

    void setLearningEnabled(bool attention, bool other){
        for (auto& param : this->named_parameters()) {
            std::cout << param.key();
            if(param.key().find("lstm1.") == std::string::npos){
                if (param.key().find("attn.") != std::string::npos) {
                    param.value().set_requires_grad(attention);
                    std::cout << ": " << attention << std::endl;
                } else {
                    param.value().set_requires_grad(other);
                    std::cout << ": " << other << std::endl;
                }
            }
        }
    }

    int64_t getParamNum(){
        int64_t total_params = 0;
        for (const auto& p : parameters()) {
            total_params += p.numel();
        }

        return total_params;
    }

    torch::nn::Linear spatial_fc1;
    std::shared_ptr<ScaledDotProductAttention> spatial_att;
    torch::nn::Linear spatial_fc2;

    torch::nn::GRU temporal_layer;
    torch::nn::Linear temporal_fc1;

    torch::nn::Linear merged_fc;
    torch::nn::Linear merged_fc2;
    torch::nn::Linear merged_fc3;
    torch::nn::Linear merged_fc4;
};
