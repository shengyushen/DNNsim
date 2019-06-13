#ifndef DNNSIM_LAYER_H
#define DNNSIM_LAYER_H

#include <sys/common.h>
#include <cnpy/Array.h>

namespace core {

    template <typename T>
    class Layer {

    private:

        /* Type of the layer */
        std::string type;

        /* Name of the network */
        std::string name;

        /* Set of layers of the network*/
        std::string input;

        /* Number of outputs */
        int Nn;

        /* Filters X size */
        int Kx;

        /* Filters Y size */
        int Ky;

        /* Stride */
        int stride;

        /* Padding */
        int padding;

        /* Activations precision: magnitude + sign + frac */
        int act_precision;

        /* Activations magnitude */
        int act_magnitude;

        /* Activations fraction */
        int act_fraction;

        /* Weights precision: magnitude + sign + frac */
        int wgt_precision;

        /* Weights magnitude */
        int wgt_magnitude;

        /* Weights fraction */
        int wgt_fraction;

        /* numpy array containing the weights for the layer */
        cnpy::Array<T> weights;

        /* numpy array containing the activations for the layer */
        cnpy::Array<T> activations;

    public:

        /* Constructor
         * @param _type     Type of the network
         * @param _name     Name of the layer
         * @param _input    Name of the input layer
         * @param _Nn       Number of outputs
         * @param _Kx       Filters X size
         * @param _Ky       Filters Y size
         * @param _stride   Stride
         * @param _padding  Padding
         */
        Layer(const std::string &_type, const std::string &_name, const std::string &_input, int _Nn, int _Kx, int _Ky,
              int _stride, int _padding) : Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding) {
            type = _type; name = _name; input = _input;
            act_precision = -1; act_magnitude = -1; act_fraction = -1;
            wgt_precision = -1; wgt_magnitude = -1; wgt_fraction = -1;
        }

        /* Constructor
         * @param _type             Type of the network
         * @param _name             Name of the layer
         * @param _input            Name of the input layer
         * @param _Nn               Number of outputs
         * @param _Kx               Filters X size
         * @param _Ky               Filters Y size
         * @param _stride           Stride
         * @param _padding          Padding
         * @param _act_precision    Activations precision
         * @param _act_magnitude    Activations magnitude
         * @param _act_fraction     Activations fraction
         * @param _wgt_precision    Weights precision
         * @param _wgt_magnitude    Weights magnitude
         * @param _wgt_fraction     Weights fraction
         */
        Layer(const std::string &_type, const std::string &_name, const std::string &_input, int _Nn, int _Kx, int _Ky,
              int _stride, int _padding, int _act_precision, int _act_magnitude, int _act_fraction, int _wgt_precision,
              int _wgt_magnitude, int _wgt_fraction) : Nn(_Nn), Kx(_Kx), Ky(_Ky), stride(_stride), padding(_padding),
              act_precision(_act_precision), act_magnitude(_act_magnitude), act_fraction(_act_fraction),
              wgt_precision(_wgt_precision), wgt_magnitude(_wgt_magnitude), wgt_fraction(_wgt_fraction) {
            type = _type; name = _name; input = _input;
        }

        /* Getters */
        std::string getType() const { return type; }
        const std::string &getName() const { return name; }
        const std::string &getInput() const { return input; }
        int getNn() const { return Nn; }
        int getKx() const { return Kx; }
        int getKy() const { return Ky; }
        int getStride() const { return stride; }
        int getPadding() const { return padding; }
        int getActPrecision() const { return act_precision; }
        int getActMagnitude() const { return act_magnitude; }
        int getActFraction() const { return act_fraction; }
        int getWgtPrecision() const { return wgt_precision; }
        int getWgtMagnitude() const { return wgt_magnitude; }
        int getWgtFraction() const { return wgt_fraction; }
        const cnpy::Array<T> &getWeights() const { return weights; }
        const cnpy::Array<T> &getActivations() const { return activations; }

        /* Setters */
        void setWeights(const cnpy::Array<T> &weights) { Layer::weights = weights; }
        void setActivations(const cnpy::Array<T> &activations) { Layer::activations = activations; }
        void setAct_precision(int act_precision, int act_magnitude, int act_fraction) {
            Layer::act_precision = act_precision;
            Layer::act_magnitude = act_magnitude;
            Layer::act_fraction = act_fraction;
        }
        void setWgt_precision(int wgt_precision, int wgt_magnitude, int wgt_fraction) {
            Layer::wgt_precision = wgt_precision;
            Layer::wgt_magnitude = wgt_magnitude;
            Layer::wgt_fraction = wgt_fraction;
        }

    };

}

#endif //DNNSIM_LAYER_H
