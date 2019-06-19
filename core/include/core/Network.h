#ifndef DNNSIM_NETWORK_H
#define DNNSIM_NETWORK_H

#include <sys/common.h>
#include <core/Layer.h>

namespace core {

    template <typename T>
    class Network {

    private:

        /* Name of the network */
        std::string name;

        /* Set of layers of the network*/
        std::vector<Layer<T>> layers;

        /* Max number of bits for the network*/
        uint32_t network_bits;

        /* Tensorflow 8b quantization */
        bool tensorflow_8b;

    public:

        /* Default constructor */
        Network() = default;

        /* Constructor
         * @param _name             The name of the network
         * @param _network_bits     Max number of bits of the network
         * @param _tensorflow_8b    Active tensorflow 8b quantization
         */
        explicit Network(const std::string &_name, uint32_t _network_bits = 16, bool _tensorflow_8b = false) :
                network_bits(_network_bits), tensorflow_8b(_tensorflow_8b) {
            name = _name;
        }

        /* Constructor
         * @param _name             The name of the network
         * @param _layers           Vector of layers
         * @param _network_bits     Max number of bits of the network
         * @param _tensorflow_8b    Active tensorflow 8b quantization
         */
        Network(const std::string &_name, const std::vector<Layer<T>> &_layers, uint32_t _network_bits = 16,
                bool _tensorflow_8b = false) : network_bits(_network_bits), tensorflow_8b(_tensorflow_8b) {
            name = _name; layers = _layers;
        }

        /* Getters */
        const std::string &getName() const { return name; }
        const std::vector<Layer<T>> &getLayers() const { return layers; }
        int getNetwork_bits() const { return network_bits; }
        bool isTensorflow_8b() const { return tensorflow_8b; }

        /* Setters */
        std::vector<Layer<T>> &updateLayers() { return layers; }
        void setNetwork_bits(uint32_t network_bits) { Network::network_bits = network_bits; }
        void setTensorflow_8b(bool tensorflow_8b) { Network::tensorflow_8b = tensorflow_8b; }

        /* Return a network in fixed point given a floating point network
         * @param network   Network in floating point
         */
        Network<uint16_t> fixed_point() {
            auto fixed_network = Network<uint16_t>(name,network_bits,tensorflow_8b);

            for(auto &layer : layers) {
                auto fixed_layer = Layer<uint16_t>(layer.getType(),layer.getName(),layer.getInput(),layer.getNn(),
                        layer.getKx(),layer.getKy(),layer.getStride(),layer.getPadding(),layer.getActPrecision(),
                        layer.getActMagnitude(),layer.getActFraction(),layer.getWgtPrecision(),layer.getWgtMagnitude(),
                        layer.getWgtFraction());

                if(tensorflow_8b) fixed_layer.setActivations(layer.getActivations().tensorflow_fixed_point());
                else fixed_layer.setActivations(layer.getActivations().profiled_fixed_point(layer.getActMagnitude(),
                        layer.getActFraction()));
                layer.setActivations(cnpy::Array<T>());

                if(tensorflow_8b) fixed_layer.setWeights(layer.getWeights().tensorflow_fixed_point());
                else fixed_layer.setWeights(layer.getWeights().profiled_fixed_point(layer.getWgtMagnitude(),
                        layer.getWgtFraction()));
                layer.setWeights(cnpy::Array<T>());

                fixed_network.updateLayers().emplace_back(fixed_layer);
            }

            return fixed_network;
        }

    };

}

#endif //DNNSIM_NETWORK_H
