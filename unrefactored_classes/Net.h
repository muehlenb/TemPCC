// © 2025, CGVR (https://cgvr.informatik.uni-bremen.de/),
// Author: Andre Mühlenbrock (muehlenb@uni-bremen.de)

#include "src/pch.h"

#pragma once

// CURRENTLY NOT USED, FlowPointNet is below!
class TNet : public torch::nn::Module {
public:
    TNet()
        : conv1(torch::nn::Conv1dOptions(6, 64, 1)),
        conv2(torch::nn::Conv1dOptions(64, 256, 1)),
        fc1(torch::nn::LinearOptions(256, 64)),
        fc2(torch::nn::LinearOptions(64, 36))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto batchsize = x.size(0);
        x = torch::relu((conv1(x)));
        x = torch::relu((conv2(x)));
        x = std::get<0>(torch::max(x, 2, true));

        x = x.view({-1, 256});

        x = torch::relu((fc1(x)));
        x = fc2(x);

        auto iden = torch::tensor({
            1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1
        }).view({1, 36}).repeat({batchsize, 1});

        if (x.is_cuda()) {
            iden = iden.to(torch::kCUDA);
        }
        x = x + iden;
        x = x.view({-1, 6, 6});
        return x;
    }

private:
    torch::nn::Conv1d conv1, conv2;
    torch::nn::Linear fc1, fc2;
};

class PointNetEncoder : public torch::nn::Module {
public:
    PointNetEncoder()
        : tNet(std::make_shared<TNet>())
        , conv1(torch::nn::Conv1dOptions(6, 64, 1))
        , conv2(torch::nn::Conv1dOptions(64, 256, 1))
    {
        register_module("tNet", tNet);
        register_module("conv1", conv1);
        register_module("conv2", conv2);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto trans = tNet->forward(x);
        x = x.transpose(2, 1);
        x = torch::bmm(x, trans);
        x = x.transpose(2, 1);
        x = torch::relu((conv1(x)));
        x = (conv2(x));
        x = std::get<0>(torch::max(x, 2, true));
        x = x.view({-1, 256});
        return x;
    }

private:
    std::shared_ptr<TNet> tNet;
    torch::nn::Conv1d conv1, conv2;
    //torch::nn::BatchNorm1d bn1, bn2;
};




struct SimpleAttention : torch::nn::Module {
    torch::nn::Linear fc;
    SimpleAttention(int64_t embed_size) : fc(register_module("attn", torch::nn::Linear(embed_size, embed_size))) {
        fc->to(torch::kCUDA);
    }

    torch::Tensor forward(torch::Tensor x) {
        auto scores = torch::softmax(fc->forward(x), -1);
        return scores * x;
    }
};

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

struct FlowPointNet : torch::nn::Module {
    bool attention_enabled = false;

    FlowPointNet()
         //pointNetEncoder(std::make_shared<PointNetEncoder>())
        : spatial_att(std::make_shared<ScaledDotProductAttention>(6,6))
        , spatial_fc1(torch::nn::Linear(588, 588))
        , spatial_fc2(torch::nn::Linear(588, 384))
        , temporal_layer(torch::nn::GRU(torch::nn::GRUOptions(4, 32).batch_first(true)))
        , temporal_fc1(torch::nn::Linear(32, 128))
        , merged_fc(torch::nn::LinearOptions(384+128, 512))
        , merged_att(std::make_shared<ScaledDotProductAttention>(16,16))
        , merged_fc2(torch::nn::LinearOptions(512, 512))
        , merged_fc3(torch::nn::LinearOptions(512, 512))
        , merged_fc4(torch::nn::LinearOptions(512, 4))
    {
        //register_module("pointNetEncoder", pointNetEncoder);
        register_module("spatial_fc1", spatial_fc1);
        register_module("spatial_att", spatial_att);
        register_module("spatial_fc2", spatial_fc2);

        register_module("gru", temporal_layer);
        register_module("fc_temporal", temporal_fc1);

        register_module("merged_fc", merged_fc);
        register_module("merged_att", merged_att);
        register_module("merged_fc2", merged_fc2);
        register_module("merged_fc3", merged_fc3);
        register_module("merged_fc4", merged_fc4);
    }

    torch::Tensor forward(torch::Tensor temporal_data, torch::Tensor spatial_data) {
        /*
        // Überprüfen, ob die feature_size ein Vielfaches von 6 ist
        int feature_size = spatial_data.size(1);
        if (feature_size % 6 != 0) {
            throw std::runtime_error("feature_size must be a multiple of 6");
        }

        int batch_size = spatial_data.size(0);

        // Umformen von spatial_data in die erwartete Dimension [batch_size, 6, num_points]
        torch::Tensor x = spatial_data.view({batch_size, 6, 49});

        // Inferieren:
        auto spatial_final = pointNetEncoder->forward(x);
*/
        auto spatial_processed_att = torch::relu(spatial_att->forward(spatial_data));
        auto spatial_processed_con = torch::cat({spatial_data, spatial_processed_att}, 1);

        auto spatial_processed = torch::relu(spatial_fc1->forward(spatial_processed_con));

        spatial_processed = torch::relu(spatial_fc2->forward(spatial_processed));

        // GRU forward pass
        auto temporal_processed = std::get<0>(temporal_layer->forward(temporal_data)).index({torch::indexing::Slice(), -1});
        temporal_processed = torch::relu(temporal_fc1->forward(temporal_processed));

        auto concatenation = torch::cat({temporal_processed, spatial_processed}, 1);

        auto merged_result = torch::relu(merged_fc->forward(concatenation));
        //auto merged_result_att = torch::relu(merged_att->forward(merged_result));
        //auto merged_processed_con = torch::cat({merged_result, merged_result_att}, 1);

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
            if(param.key().find("lstm1.") == std::string::npos){// && param.key().find("fc1.") == std::string::npos){
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
        // Gesamtanzahl der Parameter
        int64_t total_params = 0;

        // Durch alle Parameter des Modells iterieren
        for (const auto& p : parameters()) {
            // Größe jedes Parameters ermitteln und zur Gesamtanzahl hinzufügen
            total_params += p.numel();
        }

        return total_params;
    }

    //std::shared_ptr<PointNetEncoder> pointNetEncoder;

    torch::nn::Linear spatial_fc1;
    std::shared_ptr<ScaledDotProductAttention> spatial_att;
    torch::nn::Linear spatial_fc2;

    torch::nn::GRU temporal_layer;
    torch::nn::Linear temporal_fc1;

    torch::nn::Linear merged_fc;
    std::shared_ptr<ScaledDotProductAttention> merged_att;
    torch::nn::Linear merged_fc2;
    torch::nn::Linear merged_fc3;
    torch::nn::Linear merged_fc4;
};

struct Net3 : torch::nn::Module {
    bool attention_enabled = false;


    Net3() {
        transform1 = register_module("transform1", torch::nn::Linear(6, 64));

    }

    torch::nn::Linear transform1{nullptr};
    torch::nn::Linear transform2{nullptr};
};



struct Net2 : torch::nn::Module {
    bool attention_enabled = false;

    Net2() {
        tmp = register_module("gru", torch::nn::LSTM(torch::nn::LSTMOptions(3, 32).batch_first(true)));
        fc_temporal = register_module("fc_temporal", torch::nn::Linear(32, 64));
        fc_temporal2 = register_module("fc_temporal2", torch::nn::Linear(64, 64));

        attention_layer = register_module("attention", std::make_shared<ScaledDotProductAttention>(6,6));
        fc_spatial = register_module("fc_spatial", torch::nn::Linear(49*6, 256));
        fc_spatial2 = register_module("fc_spatial2", torch::nn::Linear(256, 64 + 128));


        simpleAttn = register_module("simpleAtt", std::make_shared<SimpleAttention>(256));

        fc_merge = register_module("fc_merge", torch::nn::Linear(256, 256)); // Changed to 64 because we concatenate two 32 outputs
        //merge_weight_net = register_module("merge_weight_net", torch::nn::Linear(49 * 6, 1));
        fc_merged = register_module("fc_merged", torch::nn::Linear(256, 256));
        fc_merged2 = register_module("fc_merged2", torch::nn::Linear(256, 128));

        output_layer = register_module("output_layer", torch::nn::Linear(128, 4)); // Final output layer to return 4 floats
    }

    torch::Tensor forward(torch::Tensor temporal_data, torch::Tensor spatial_data) {
        temporal_data.zero_();

        // GRU forward pass
        auto gru_output = std::get<0>(tmp->forward(temporal_data)).index({torch::indexing::Slice(), -1});
        auto temporal_processed = torch::relu(fc_temporal->forward(gru_output));
        auto temporal_processed2 = torch::relu(fc_temporal2->forward(temporal_processed));

        // Attention forward pass
        auto spatial_attention_output = attention_layer->forward(spatial_data);
        auto spatial_processed = torch::relu(fc_spatial->forward(spatial_attention_output));
        auto spatial_processed2 = torch::relu(fc_spatial2->forward(spatial_processed));

        // Calculate merge weights based on spatial positions (first 3 features)
        //auto merge_weights = torch::sigmoid(merge_weight_net->forward(spatial_data));

        // Weighted merge of both pathways
        //auto merged_output = merge_weights * temporal_processed + (1 - merge_weights) * spatial_processed;
        //auto final_merge = torch::cat({merge_weights * temporal_processed, (1 - merge_weights) * spatial_processed}, 1);
        auto concatenation = torch::relu(simpleAttn->forward(torch::cat({temporal_processed2, spatial_processed2}, 1)));

        auto merge = torch::relu(fc_merge->forward(concatenation));
        auto merged = torch::relu(fc_merged->forward(merge));
        auto merged2 = torch::relu(fc_merged2->forward(merged));

        // Testen, die Gewichtung alleine durch die Anzahl an der Punkte, die nah bei sind, vorzunehmen.
        // D.h., wenn keine nahen Punkte vorhanden sind, immer nur Temporal gewichten. Und dann vielleicht später
        // lernen (aber erst einmal testen, wie es ist, wenn man die Gewichtung nicht lernt, sondern fest vorgibt.
        //auto final_att_output = attention_end->forward(merge);

        // Process merged data through final layers:
        return output_layer->forward(merged2);
    }

    /**
     * Saves the trained weight.
     *
     * @param name
     */
    void saveNet(std::string name = "temp_net.pt"){
        torch::serialize::OutputArchive archive;
        save(archive);
        archive.save_to(CMAKE_SOURCE_DIR "/data/net/" + name);
    }

    /**
     * Loads the network.
     *
     * @param name
     */
    void loadNet(std::string name = "temp_net.pt"){
        torch::serialize::InputArchive archive;
        archive.load_from(CMAKE_SOURCE_DIR "/data/net/" + name);
        load(archive);
    }

    void setLearningEnabled(bool attention, bool other){
        for (auto& param : this->named_parameters()) {
            std::cout << param.key();
            if(param.key().find("lstm1.") == std::string::npos){// && param.key().find("fc1.") == std::string::npos){
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

    /**
     *
     */
    int64_t getParamNum(){
        // Gesamtanzahl der Parameter
        int64_t total_params = 0;

        // Durch alle Parameter des Modells iterieren
        for (const auto& p : parameters()) {
            // Größe jedes Parameters ermitteln und zur Gesamtanzahl hinzufügen
            total_params += p.numel();
        }

        return total_params;
    }

    std::shared_ptr<SimpleAttention> simpleAttn;
    torch::nn::LSTM tmp{nullptr};
    torch::nn::Linear fc_temporal{nullptr}, fc_spatial{nullptr}, fc_merge{nullptr}, merge_weight_net{nullptr}, output_layer{nullptr};
    std::shared_ptr<ScaledDotProductAttention> attention_layer, attention_end;
    torch::nn::Linear fc_temporal2{nullptr}, fc_spatial2{nullptr};
    torch::nn::Linear fc_merged{nullptr}, fc_merged2{nullptr};
};

/**
 *
 */
struct Net : torch::nn::Module {
    bool attention_enabled = false;

    Net(int64_t gru_size, int64_t layer1_size, int64_t layer2_size, int64_t layer3_size) :
        attn(register_module("attn", std::make_shared<SimpleAttention>(layer1_size + gru_size))) {

        gru1 = register_module("gru1", torch::nn::GRU(torch::nn::GRUOptions(3, gru_size).batch_first(true)));
        fc1 = register_module("fc1", torch::nn::Linear(98 * 3, layer1_size));

        mfc1 = register_module("mfc1", torch::nn::Linear(layer1_size + gru_size, layer2_size));
        mfc2 = register_module("mfc2", torch::nn::Linear(layer2_size, layer3_size));
        mfc3 = register_module("mfc3", torch::nn::Linear(layer3_size, 4));
    }

    torch::Tensor forward(torch::Tensor temporal_data, torch::Tensor spatial_data) {
        auto gru_out1 = std::get<0>(gru1->forward(temporal_data)).squeeze(1);
        auto gru_out_last = gru_out1.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});

        auto fc1_out = torch::relu(fc1->forward(spatial_data));
        auto merged = torch::cat({gru_out_last, fc1_out}, 1);

        auto attn_out = merged;
        // Apply attention
        if(attention_enabled)
            attn_out = attn->forward(merged);

        auto fc2_out = torch::relu(mfc1->forward(attn_out));
        auto fc3_out = torch::relu(mfc2->forward(fc2_out));
        auto out = mfc3->forward(fc3_out);

        return out;
    }

    /**
     * Saves the trained weight.
     *
     * @param name
     */
    void saveNet(std::string name = "temp_net.pt"){
        torch::serialize::OutputArchive archive;
        save(archive);
        archive.save_to(CMAKE_SOURCE_DIR "/data/net/" + name);
    }

    /**
     * Loads the network.
     *
     * @param name
     */
    void loadNet(std::string name = "temp_net.pt"){
        torch::serialize::InputArchive archive;
        archive.load_from(CMAKE_SOURCE_DIR "/data/net/" + name);
        load(archive);
    }

    void setLearningEnabled(bool attention, bool other){
        for (auto& param : this->named_parameters()) {
            std::cout << param.key();
            if(param.key().find("lstm1.") == std::string::npos){// && param.key().find("fc1.") == std::string::npos){
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

    /**
     *
     */
    int64_t getParamNum(){
        // Gesamtanzahl der Parameter
        int64_t total_params = 0;

        // Durch alle Parameter des Modells iterieren
        for (const auto& p : parameters()) {
            // Größe jedes Parameters ermitteln und zur Gesamtanzahl hinzufügen
            total_params += p.numel();
        }

        return total_params;
    }

    std::shared_ptr<SimpleAttention> attn;

    torch::nn::GRU gru1{nullptr};
    torch::nn::Linear fc1{nullptr}, mfc1{nullptr}, mfc2{nullptr}, mfc3{nullptr};
};
