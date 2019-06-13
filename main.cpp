/*
Copyright (c) 2019 Isak Edo Vivancos, Andreas Moshovos
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <sys/common.h>
#include <sys/cxxopts.h>
#include <sys/Batch.h>

#include <interface/NetReader.h>
#include <interface/NetWriter.h>
#include <interface/StatsWriter.h>

#include <core/Network.h>
#include <core/Stripes.h>
#include <core/DynamicStripes.h>
#include <core/Loom.h>
#include <core/BitPragmatic.h>
#include <core/Laconic.h>
#include <core/BitTacticalP.h>
#include <core/BitTacticalE.h>
#include <core/SCNN.h>
#include <core/BitFusion.h>

template <typename T>
core::Network<T> read(const std::string &input_type, const std::string &network_name, int batch, bool QUIET) {

    // Read the network
    core::Network<T> network;
    interface::NetReader<T> reader = interface::NetReader<T>(network_name, batch, QUIET);
    if (input_type == "Caffe") {
        network = reader.read_network_caffe();
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);

    } else if (input_type == "Trace") {
        network = reader.read_network_trace_params();
        reader.read_precision(network);
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);

    } else if (input_type == "CParams") {
        network = reader.read_network_conv_params();
        reader.read_weights_npy(network);
        reader.read_activations_npy(network);

    } else if (input_type == "Protobuf") {
        network = reader.read_network_protobuf();
    } else {
		throw std::runtime_error("Input model option not recognized");
	}

    return network;

}

template <typename T>
void write(const core::Network<T> &network, bool QUIET) {

    // Write network
    interface::NetWriter<T> writer = interface::NetWriter<T>(network.getName(),QUIET);
    writer.write_network_protobuf(network);

}

template <typename T>
std::vector<schedule> read_schedule(const std::string &network_name, const std::string &arch,
        const sys::Batch::Simulate::Experiment &experiment, bool QUIET) {

    interface::NetReader<T> reader = interface::NetReader<T>(network_name, 0, QUIET);
    int mux_entries = experiment.lookahead_h + experiment.lookaside_d + 1;
    std::string schedule_type = arch + "_" + experiment.search_shape + std::to_string(mux_entries) + "("
            + std::to_string(experiment.lookahead_h) + "-" + std::to_string(experiment.lookaside_d) + ")";
    return reader.read_schedule_protobuf(schedule_type);
}

template <typename T>
void write_schedule(const core::Network<T> &network, core::BitTactical<T> &DNNsim, const std::string &arch,
        const sys::Batch::Simulate::Experiment &experiment, bool QUIET) {
    const auto &network_schedule = DNNsim.network_scheduler(network);
    interface::NetWriter<uint16_t> writer = interface::NetWriter<uint16_t>(network.getName(),QUIET);
    int mux_entries = experiment.lookahead_h + experiment.lookaside_d + 1;
    std::string schedule_type = arch + "_" + experiment.search_shape + std::to_string(mux_entries) + "("
            + std::to_string(experiment.lookahead_h) + "-" + std::to_string(experiment.lookaside_d) + ")";
    writer.write_schedule_protobuf(network_schedule,schedule_type);
}

void check_options(const cxxopts::Options &options)
{
    if(options.count("batch") == 0) {
        throw std::runtime_error("Please provide a batch file with instructions. Examples in folder \"examples\"");
    } else {
        std::string batch_path = options["batch"].as<std::string>();
        std::ifstream file(batch_path);
        if(!file.good()) {
            throw std::runtime_error("The path " + batch_path + " does not exist.");
        }
    }

    if(options.count("threads") == 1 && options["threads"].as<uint16_t>() < 1) {
        throw std::runtime_error("The number of threads must be at least one.");
    }

}

cxxopts::Options parse_options(int argc, char *argv[]) {
    cxxopts::Options options("DNNsim", "Deep Neural Network simulator");

    // help-related options
    options.add_options("help")("h,help", "Print this help message", cxxopts::value<bool>(), "");

    options.add_options("batch")
    ("batch", "Specify a batch file with instructions. Examples in folder \"examples\"",cxxopts::value<std::string>(),
            "<Prototxt file>");

    options.add_options("simulation")
    ("t,threads", "Specify the number of threads",cxxopts::value<uint16_t>(),"<Positive number>")
    ("q,quiet", "Don't show stdout progress messages",cxxopts::value<bool>(),"<Boolean>")
    ("fast_mode", "Enable fast mode: simulate only one image",cxxopts::value<bool>(),"<Boolean>")
    ("store_fixed_point_protobuf", "Store the fixed point network in an intermediate Protobuf file",
    		cxxopts::value<bool>(),"<Boolean>");

    options.parse_positional("batch");

    options.parse(argc, argv);

    return options;
}

int main(int argc, char *argv[]) {
    try {
        auto const options = parse_options(argc, argv);

        // Help
        if (options.count("h") == 1) {
            std::cout << options.help({"help", "batch", "simulation"}) << std::endl;
            return 0;
        }

        check_options(options);

        uint8_t N_THREADS = options.count("threads") == 0 ? (uint8_t)1 : (uint8_t)options["threads"].as<uint16_t>();
        bool QUIET = options.count("quiet") == 0 ? false : options["quiet"].as<bool>();
        bool FAST_MODE = options.count("fast_mode") == 0 ? false : options["fast_mode"].as<bool>();
        bool STORE_PROTOBUF = options.count("store_fixed_point_protobuf") == 0 ? false :
        		options["store_fixed_point_protobuf"].as<bool>();
        std::string batch_path = options["batch"].as<std::string>();
        sys::Batch batch = sys::Batch(batch_path);
        batch.read_batch();

        for(const auto &simulate : batch.getSimulations()) {

            if(!QUIET) std::cout << "Network: " << simulate.network << std::endl;

            try {
                if (simulate.data_type == "Float32") {
                    core::Network<float> network;
                    network = read<float>(simulate.model, simulate.network, simulate.batch, QUIET);
                    network.setNetwork_bits(simulate.network_bits);
                    for(const auto &experiment : simulate.experiments) {
                        if(experiment.architecture == "None") {
                            core::Simulator<float> DNNsim(N_THREADS,FAST_MODE);
                            if (experiment.task == "Sparsity") DNNsim.sparsity(network);
                        } else if (experiment.architecture == "SCNN") {
                            core::SCNN<float> DNNsim(experiment.Wt, experiment.Ht, experiment.I, experiment.F,
                                    experiment.out_acc_size, experiment.banks, N_THREADS, FAST_MODE);
                            if (experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);
                        }
                    }
                } else if (simulate.data_type == "Fixed16") {
                    core::Network<uint16_t> network;
                    if (simulate.model != "Protobuf") {
                        core::Network<float> tmp_network;
                        tmp_network = read<float>(simulate.model, simulate.network, simulate.batch, QUIET);
                        tmp_network.setTensorflow_8b(simulate.tensorflow_8b);
                        network = tmp_network.fixed_point();
                    } else {
                        network = read<uint16_t>(simulate.model, simulate.network, simulate.batch, QUIET);
                    }
                    network.setNetwork_bits(simulate.network_bits);
                    network.setTensorflow_8b(simulate.tensorflow_8b);

                    if(STORE_PROTOBUF) write(network,QUIET);

                    for(const auto &experiment : simulate.experiments) {

                        if(!QUIET) std::cout << "Starting simulation " << experiment.task << " for architecture "
                                             << experiment.architecture << std::endl;

                        if(experiment.architecture == "None") {
                            core::Simulator<uint16_t> DNNsim(N_THREADS,FAST_MODE);
                            if(experiment.task == "Sparsity") DNNsim.sparsity(network);
                            else if(experiment.task == "BitSparsity") DNNsim.bit_sparsity(network);

                        } else if(experiment.architecture == "BitPragmatic") {
                            core::BitPragmatic<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.bits_first_stage,experiment.column_registers,experiment.diffy,
                                    N_THREADS,FAST_MODE);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if(experiment.architecture == "Stripes") {
                            core::Stripes<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,experiment.bits_pe,
                                                           N_THREADS,FAST_MODE);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if(experiment.architecture == "DynamicStripes") {
                            core::DynamicStripes<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.precision_granularity, experiment.column_registers, experiment.bits_pe,
                                    experiment.leading_bit, experiment.diffy, N_THREADS,FAST_MODE);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if(experiment.architecture == "Loom") {
                            core::Loom<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.precision_granularity, experiment.pe_serial_bits, experiment.leading_bit,
                                    experiment.dynamic_weights, N_THREADS,FAST_MODE);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "Laconic") {
                            core::Laconic<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,N_THREADS, FAST_MODE);
                            if(experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "BitTacticalP") {
                            core::BitTacticalP<uint16_t> DNNsim(experiment.n_columns, experiment.n_rows,
                                    experiment.precision_granularity, experiment.column_registers,
                                    experiment.lookahead_h, experiment.lookaside_d, experiment.search_shape,
                                    experiment.leading_bit, N_THREADS, FAST_MODE);
                            if(experiment.task == "Cycles" && experiment.read_schedule) {
                                auto dense_schedule = read_schedule<uint16_t>(network.getName(),"BitTactical",
                                        experiment,QUIET);
                                DNNsim.run(network, dense_schedule);
                            }
                            else if (experiment.task == "Schedule")
                                write_schedule<uint16_t>(network,DNNsim,"BitTactical",experiment,QUIET);
                            else if (experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "BitTacticalE") {
                            core::BitTacticalE<uint16_t> DNNsim(experiment.n_columns,experiment.n_rows,
                                    experiment.bits_first_stage, experiment.column_registers, experiment.lookahead_h,
                                    experiment.lookaside_d, experiment.search_shape,N_THREADS,FAST_MODE);
                            if(experiment.task == "Cycles" && experiment.read_schedule) {
                                auto dense_schedule = read_schedule<uint16_t>(network.getName(),"BitTactical",
                                                                              experiment,QUIET);
                                DNNsim.run(network, dense_schedule);
                            }
                            else if (experiment.task == "Schedule")
                                write_schedule<uint16_t>(network,DNNsim,"BitTactical",experiment,QUIET);
                            else if (experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "SCNN") {
                            core::SCNN<uint16_t> DNNsim(experiment.Wt, experiment.Ht, experiment.I, experiment.F,
                                                        experiment.out_acc_size, experiment.banks, N_THREADS, FAST_MODE);
                            if (experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);

                        } else if (experiment.architecture == "BitFusion") {
                            core::BitFusion<uint16_t> DNNsim(experiment.M, experiment.N, experiment.pmax,
                                    experiment.pmin, N_THREADS, FAST_MODE);
                            if (experiment.task == "Cycles") DNNsim.run(network);
                            else if (experiment.task == "Potentials") DNNsim.potentials(network);
                        }
                        sys::Statistics::updateFlagsLastStat(simulate.tensorflow_8b);
                    }
                }
			} catch (std::exception &exception) {
                std::cerr << "Simulation error: " << exception.what() << std::endl;
                #ifdef STOP_AFTER_ERROR
                exit(1);
                #endif
            }
        }

        //Dump statistics
        auto writer = interface::StatsWriter(QUIET);
        writer.dump_csv();

    } catch (std::exception &exception) {
        std::cerr << "Error: " << exception.what() << std::endl;
        exit(1);
    }
    return 0;
}
