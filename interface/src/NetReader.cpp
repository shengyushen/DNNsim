
#include <interface/NetReader.h>

namespace interface {

    bool ReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto) {
        int fd = open(filename, O_RDONLY);
        auto input = new google::protobuf::io::FileInputStream(fd);
        bool success = google::protobuf::TextFormat::Parse(input, proto);
        delete input;
        close(fd);
        return success;
    }

    template <typename T>
    void NetReader<T>::check_path(const std::string &path) {
        std::ifstream file(path);
        if(!file.good()) {
            throw std::runtime_error("The path " + path + " does not exist.");
        }
    }

    template <typename T>
    std::string NetReader<T>::inputName() {
        std::string output_name = this->name;
        std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time
        output_name += "-" + type;
        return output_name;
    }

    template <typename T>
    core::Layer<T> NetReader<T>::read_layer_caffe(const caffe::LayerParameter &layer_caffe) {
        int Nn = -1, Kx = -1, Ky = -1, stride = -1, padding = -1;

        if(layer_caffe.type() == "Convolution") {
            Nn = layer_caffe.convolution_param().num_output();
            Kx = layer_caffe.convolution_param().kernel_size(0);
            Ky = layer_caffe.convolution_param().kernel_size(0);
            stride = layer_caffe.convolution_param().stride_size() == 0 ? 1 : layer_caffe.convolution_param().stride(0);
            padding = layer_caffe.convolution_param().pad_size() == 0 ? 0 : layer_caffe.convolution_param().pad(0);
        } else if (layer_caffe.type() == "InnerProduct") {
            Nn = layer_caffe.inner_product_param().num_output();
            Kx = 1; Ky = 1; stride = 1; padding = 0;

        } else if (layer_caffe.type() == "Pooling") {
            Kx = layer_caffe.pooling_param().kernel_size();
            Ky = layer_caffe.pooling_param().kernel_size();
            stride = layer_caffe.pooling_param().stride();
        }

        std::string name = layer_caffe.name();
        std::replace( name.begin(), name.end(), '/', '-'); // Sanitize name
        return core::Layer<T>(layer_caffe.type(),name,layer_caffe.bottom(0), Nn, Kx, Ky, stride, padding);
    }

    template <typename T>
    core::Network<T> NetReader<T>::read_network_caffe() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        caffe::NetParameter network;

        check_path("models/" + this->name);
        std::string path = "models/" + this->name + "/train_val.prototxt";
        check_path(path);
        if (!ReadProtoFromTextFile(path.c_str(),&network)) {
            throw std::runtime_error("Failed to read prototxt");
        }

        for(const auto &layer : network.layer()) {
            if(this->layers_allowed.find(layer.type()) != this->layers_allowed.end()) {
                layers.emplace_back(read_layer_caffe(layer));
            }
        }

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    core::Layer<T> NetReader<T>::read_layer_proto(const protobuf::Network_Layer &layer_proto) {
        core::Layer<T> layer = core::Layer<T>(layer_proto.type(),layer_proto.name(),layer_proto.input(),
        layer_proto.nn(),layer_proto.kx(),layer_proto.ky(),layer_proto.stride(),layer_proto.padding(),
        std::make_tuple<int,int>(layer_proto.act_mag(),layer_proto.act_prec()),
        std::make_tuple<int,int>(layer_proto.wgt_mag(),layer_proto.wgt_prec()));

        // Read weights, activations, and output activations only to the desired layers
        if(this->layers_data.find(layer_proto.type()) != this->layers_data.end()) {

            std::string type = typeid(T).name() + std::to_string(sizeof(T));// Get template type in run-time

            std::vector<size_t> weights_shape;
            for (const int value : layer_proto.wgt_shape())
                weights_shape.push_back((size_t) value);

            #ifdef BIAS
            std::vector<size_t> biases_shape;
            for (const int value : layer_proto.bias_shape())
                biases_shape.push_back((size_t) value);
            #endif

            std::vector<size_t> activations_shape;
            for (const int value : layer_proto.act_shape())
                activations_shape.push_back((size_t) value);

            #ifdef OUTPUT_ACTIVATIONS
            std::vector<size_t> out_activations_shape;
            for (const int value : layer_proto.out_act_shape())
                out_activations_shape.push_back((size_t) value);
            #endif

            std::vector<T> weights_data;
            std::vector<T> biases_data;
            std::vector<T> activations_data;
            std::vector<T> out_activations_data;


            if (type == "f4") {
                for (const auto &value : layer_proto.wgt_data_flt())
                    weights_data.push_back(value);

                #ifdef BIAS
                for (const auto value : layer_proto.bias_data_flt())
                    biases_data.push_back(value);
                #endif

                for (const auto value : layer_proto.act_data_flt())
                    activations_data.push_back(value);

                #ifdef OUTPUT_ACTIVATIONS
                for (const auto value : layer_proto.out_act_data_flt())
                                    out_activations_data.push_back(value);
                #endif
            } else if (type == "t2") {
                for (const auto &value : layer_proto.wgt_data_fxd())
                    weights_data.push_back(value);

                #ifdef BIAS
                for (const auto value : layer_proto.bias_data_fxd())
                    biases_data.push_back(value);
                #endif

                for (const auto value : layer_proto.act_data_fxd())
                    activations_data.push_back(value);

                #ifdef OUTPUT_ACTIVATIONS
                for (const auto value : layer_proto.out_act_data_fxd())
                    out_activations_data.push_back(value);
                #endif
            }

            cnpy::Array<T> weights; weights.set_values(weights_data,weights_shape);
            layer.setWeights(weights);

            #ifdef BIAS
            cnpy::Array<T> biases; biases.set_values(biases_data,biases_shape);
            layer.setBias(biases);
            #endif

            cnpy::Array<T> activations; activations.set_values(activations_data,activations_shape);
            layer.setActivations(activations);

            #ifdef OUTPUT_ACTIVATIONS
            cnpy::Array<T> out_activations; out_activations.set_values(out_activations_data,out_activations_shape);
            layer.setOutput_activations(out_activations);
            #endif

        }

        return layer;
    }

    template <typename T>
    core::Network<T> NetReader<T>::read_network_protobuf() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        protobuf::Network network_proto;

        {
            // Read the existing network.
            check_path("net_traces/" + this->name);
            std::string path = "net_traces/" + this->name + '/' + inputName() + ".proto";
            check_path(path);
            std::fstream input(path,std::ios::in | std::ios::binary);
            if (!network_proto.ParseFromIstream(&input)) {
                throw std::runtime_error("Failed to read protobuf");
            }
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network<T>(this->name,layers);
    }


    template <typename T>
    core::Network<T> NetReader<T>::read_network_gzip() {
        GOOGLE_PROTOBUF_VERIFY_VERSION;

        std::vector<core::Layer<T>> layers;
        protobuf::Network network_proto;

        // Read the existing network.
        check_path("net_traces/" + this->name);
        std::string path = "net_traces/" + this->name + '/' + inputName();
        check_path(path);
        std::fstream input(path, std::ios::in | std::ios::binary);

        google::protobuf::io::IstreamInputStream inputFileStream(&input);
        google::protobuf::io::GzipInputStream gzipInputStream(&inputFileStream);

        if (!network_proto.ParseFromZeroCopyStream(&gzipInputStream)) {
            throw std::runtime_error("Failed to read Gzip protobuf");
        }

        std::string name = network_proto.name();

        for(const protobuf::Network_Layer &layer_proto : network_proto.layers())
            layers.emplace_back(read_layer_proto(layer_proto));

        google::protobuf::ShutdownProtobufLibrary();

        return core::Network<T>(this->name,layers);
    }

    template <typename T>
    void NetReader<T>::read_weights_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/wgt-" + layer.getName() + ".npy" ;
                cnpy::Array<T> weights; weights.set_values("net_traces/" + this->name + file);
                layer.setWeights(weights);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_bias_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/bias-" + layer.getName() + ".npy" ;
                cnpy::Array<T> bias; bias.set_values("net_traces/" + this->name + file);
                layer.setBias(bias);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_activations_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/act-" + layer.getName() + "-0.npy";
                cnpy::Array<T> activations; activations.set_values("net_traces/" + this->name + file);
                layer.setActivations(activations);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_output_activations_npy(core::Network<T> &network) {
        check_path("net_traces/" + this->name);
        for(core::Layer<T> &layer : network.updateLayers()) {
            if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                std::string file = "/act-" + layer.getName() + "-0-out.npy" ;
                cnpy::Array<T> activations; activations.set_values("net_traces/" + this->name + file);
                layer.setOutput_activations(activations);
            }
        }
    }

    template <typename T>
    void NetReader<T>::read_precision(core::Network<T> &network) {

        std::string line;
        std::stringstream ss_line;
        std::vector<int> act_mag;
        std::vector<int> act_prec;
        std::vector<int> wgt_mag;
        std::vector<int> wgt_prec;

        std::ifstream myfile ("models/" + this->name + "/precision.txt");
        if (myfile.is_open()) {
            std::string word;

            getline(myfile,line); // Remove first line

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                act_mag.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                act_prec.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                wgt_mag.push_back(stoi(word));

            getline(myfile,line);
            ss_line = std::stringstream(line);
            while (getline(ss_line,word,';'))
                wgt_prec.push_back(stoi(word));

            myfile.close();

            int i = 0;
            for(core::Layer<T> &layer : network.updateLayers()) {
                if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                    layer.setAct_precision(std::make_tuple(act_mag[i], act_prec[i]));
                    layer.setWgt_precision(std::make_tuple(wgt_mag[i], wgt_prec[i]));
                    i++;
                } else {
                    layer.setAct_precision(std::make_tuple(0,0));
                    layer.setWgt_precision(std::make_tuple(0,0));
                }
            }

        } else {
            // Generic precision
            int i = 0;
            for(core::Layer<T> &layer : network.updateLayers()) {
                if(this->layers_data.find(layer.getType()) != this->layers_data.end()) {
                    layer.setAct_precision(std::make_tuple(13, 2));
                    layer.setWgt_precision(std::make_tuple(0, 15));
                    i++;
                } else {
                    layer.setAct_precision(std::make_tuple(0,0));
                    layer.setWgt_precision(std::make_tuple(0,0));
                }
            }
        }
    }

    INITIALISE_DATA_TYPES(NetReader);

}