# DNNsim 

### Requeriments
*   Cmake version >= 3.10
*   GNU C++ compiler version >= 5.1
*   Google Protobuf for C++. Installation link:
    *   https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

### Allowed input files

*   The architecture of the net in a train_val.prototxt file (without weights and activations)
*   The architecture of the net in a trace_params.csv file (without weights and activations)
*   The architecture of the net in a conv_params.csv file (without weights and activations)
*   Weights and Inputs activations in a *.npy file using the following format
*   Full network in a Google protobuf format file
*   Tactical schedule in a protobuf format file

### Compilation:
Command line compilation. First we need to configure the project:
    
    cmake -H. -Bcmake-build-release -DCMAKE_BUILD_TYPE=Release

Then, we can proceed to build the project

    cmake --build cmake-build-release/ --target all

### Set up directories

Create folder **models** including a folder for each network. Every network must include:
   *  train_val.prototxt
   *  trace_params.csv (Instead of the prototxt file)
      *  _Header_: \<Layer\>:\<Input layer\*\>:\<Output channels\>:\<Kernel X\>:\<Kernel Y\>:\<Stride\>:\<Padding\>
      *  \* Input layer is optional
   *  conv_params.csv (Instead of the prototxt file and the precision.txt) 
      *  _Header_: \<Network\>:\<Layer\>:\<Type(conv|fc|lstm)\>:\<Output channels\>:\<Weight channels\>:\<Kernel X\>: \\  
                   \<Kernel Y\>:\<Kernel size\>:\<Stride\>:\<Padding\>:\<Precision\>:\<Magnitude (without sign)\>
      *  Weights are generic precision 0:15                
   *  precision.txt (Optional, contain 5 lines as the example, first line is skipped)
        *   If this file does not exist the precisions are 13:2 for activations and 0:15 for weights
   
   ```
   magnitude (including sign), fraction, wgt_magnitude, wgt_fraction
   9;9;8;9;9;8;6;4;
   -1;-2;-3;-3;-3;-3;-1;0;
   2;1;1;1;1;-3;-4;-1;
   7;8;7;8;8;9;8;8;
   ```
    
Create folder **net_traces** including a folder for each network. 
In the case of **inference** simulation, every network must include:
   * wgt-$LAYER.npy
   * act-$LAYER-$BATCH.npy
       
### Test

Print help:

    ./DNNsim -h
    
The simulator instructions are defined in prototxt files. Example files can be found [here](examples/README.md).

Results from simulations can be found inside the results folder. One csv file for each simulation 
containing one line for each layer which are grouped per batch. After that, one line for the each layer is shown with the 
average results for all batches. Finally, the last line corresponds to the total of the network. 
(In the case of training the results are grouped per epoch)

### Command line options

* Option **--threads <positive_num>** indicates the number of simultaneous threads that can be executed. The code is 
parallelized per batch using OpenMP library
* Option **--quiet** remove stdout messages from simulations.
* Option **--fast_mode** makes the simulation execute only one batch per network, the first one.
* Option **--store_fixed_point_protobuf** store the fixed point network in a intermediate Protobuf file.

### Allowed simulations

*  Allowed model types for the simulations:

| model | Description |
|:---:|:---:|
| Caffe | Load network model from *train_val.prototxt*, precisions from *precision.txt*, and traces from numpy arrays |
| Trace | Load network model from *trace_params.csv*, precisions from *precision.txt*, and traces from numpy arrays | 
| CParams | Load network model and precisions from *conv_params.csv*, and traces from numpy arrays | 
| Protobuf | Load network model, precisions, and traces from a protobuf file |

*  Allowed architectures for the experiments:

| Architecture | Description | 
|:---:|:---:|
| None | Special generic architecture |
| BitPragmatic | **Ae**: Exploits bit-level sparsity of activations |
| Stripes | **Ap**: Exploits precision requirements of activations |
| DynamicStripes | **Ap**: Exploits dynamic precision requirements of a group of activations | 
| Loom | **Wp + Ap**: Exploits precision requirements of weights and dynamic group of activations |
| Laconic | **We + Ae**: Exploits bit-level sparsity of both weights and activations |
| BitTacticalP | **W + Ap**: Skips zero weights and exploits precision requirements of activations | 
| BitTacticalE | **W + Ae**: Skips zero weights and exploits bit-level sparsity of activations | 
| SCNN | **W + A**: Skips zero weights and zero activations |
| BitFusion | **Wp + Ap**: Exploits precision requirements of activations and weights for powers of two |

Input parameters are the parameters that can be changed for each architecture in the prototxt batch file.
Default parameters are defined in the header of each architecture, they can be changed in the specific file.
Data type indicates the possible data types allowed: 
Float32 for 4bytes floating point, and Fixed16 for 2bytes quantized integer   

| Architecture | Input Parameters | Default Parameters\* | Data type |
|:---:|:---:|:---:|:---:|
| BitPragmatic | N_COLUMNS, N_ROWS, BITS_FIRST_STAGE, COLUMN_REGISTERS, DIFFY | BOOTH_ENCODING, ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| Stripes | N_COLUMNS, N_ROWS, BITS_PE | FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| DynamicStripes | N_COLUMNS, N_ROWS, PRECISION_GRANULARITY, COLUMN_REGISTERS, LEADING_BIT, MINOR_BIT, DIFFY | FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| Loom | N_COLUMNS, N_ROWS, PRECISION_GRANULARITY, PE_SERIAL_BITS, LEADING_BIT, MINOR_BIT, DYNAMIC_WEIGHTS | FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| Laconic | N_COLUMNS, N_ROWS | BOOTH_ENCODING, ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| BitTacticalP | N_COLUMNS, N_ROWS, LOOKAHEAD_H, LOOKASIDE_D, SEARCH_SHAPE, PRECISION_GRANULARITY, COLUMN_REGISTERS, LEADING_BIT, MINOR_BIT | ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| BitTacticalE | N_COLUMNS, N_ROWS, LOOKAHEAD_H, LOOKASIDE_D, SEARCH_SHAPE, BITS_FIRST_STAGE, COLUMN_REGISTERS | BOOTH_ENCODING, ZERO_COUNT, FC_MULTIPLEX_COLUMNS, WEIGHT_LANES | Fixed16 |
| SCNN | Wt, Ht, I, F, out_acc_size, BANKS | ZERO_COUNT | Fixed16, Float32 |
| BitFusion | M, N, PMAX, PMIN | - | Fixed16 |

*\*Default features can be removed in their specific header file*

*  Allowed tasks for these architectures:

| Task | Description | 
|:---:|:---:|
| Cycles | Simulate number of cycles and memory accesses | 
| Potentials | Calculate ideal speedup and work reduction | 
| Schedule | Schedule weights statically (Only for BitTactical architecture) |

* Allowed task for the special architecture "None":

| Task | Description | Data type |
|:---:|:---:|:---:|
| Sparsity | Calculate sparsity for actiations and weights, number of zero values | Fixed16, Float32 |
| BitSparsity | Calculate bit sparsity for activations and weights, number of zero bits | Fixed16 |

### Default Parameters Description   

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| BOOTH_ENCODING | bool | Activate booth encoding | True-False | True |
| ZERO_COUNT | bool | Zero values count as one cycle | True-False | True | 
| FC_MULTIPLEX_COLUMNS | bool | Fully connected layers are time-multiplexed across the columns | True-False | True |
| WEIGHT_LANES | uint32 | Number of weights per PE | Positive Numbers | 16 |
   
### Input Parameters Description   

The batch file can be constructed as follows for the simulation tool:

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| network | string | Name of the network as in the folder models | Valid path | N/A |
| batch | uint32 | Corresponding batch for the Numpy traces | Positive numbers | 0 | 
| inputType | string | Format of the input model definition and traces | Trace-Caffe-CParams-Protobuf-Gzip | N/A |
| inputDataType | string | Data type of the input traces | Float32-Fixed16-BFloat16 | N/A |
| network_bits | uint32 | Number of baseline bits of the network | Positive Number | 16 |
| tensorflow_8b | bool | Use tensorflow 8bits quantization | True-False | False |

Experiments for the simulation tool can contain the following parameters.

| Name | Data Type | Description | Valid Options | Default |
|:---:|:---:|:---:|:---:|:---:|
| architecture | string | Name of the architecture to simulate | Allowed architectures | N/A |
| task | string | Name of the task to simulate | Allowed tasks | N/A |
| | | | | |
| n_columns | uint32 | Number of columns/windows in the tile | Positive number | 16 |
| n_rows | uint32 | Number of rows/filters in the tile | Positive number | 16 |
| column_registers | uint32 | Number of registers per column to run-ahead | Positive number | 0 |
| precision_granularity | uint32 | Size of the group of values | Positive number | 16 |
| leading bit | bool | Only the leading bit for dynamic precisions | True-False | False |
| minor bit | bool | Only the minor bit for dynamic precisions | True-False | False |
| bits_first_stage | uint32 | Number of bits of the first stage shifter | Positive number | 0 |
| bits_pe | uint32 | Number of bits per PE | Positive number | 16 |
| lookahead_h |uint32 | Lookahead window size | Positive number | 2 |
| lookaside_d |uint32 | Lookaside window size | Positive number | 5 | 
| search_shape | string | Shape of the scheduler search | L-T | L |
| read_schedule_from_proto | bool | Read the scheduled weights from a Protobuf file | True-False | False |
| diffy | bool | Simulate Diffy in top of the architecture | True-False | False |
| pe_serial_bits | uint32 | Number of serial bits per PE | Positive Number | 1 |
| dynamic_weights | bool | Use dynamic precision for the weights | True-False | False |
| | | | | |
| Wt | uint32 | Number of PE columns | Positive number | 8 |
| Ht | uint32 | Number of PE rows | Positive number | 8 |
| I | uint32 | Column multipliers per PE | Positive number | 4 |
| F | uint32 | Number of PE columns | Positive number | 4 |
| out_acc_size | uint32 | Size of the output accumulator per PE | Positive number | 1024 |
| banks | uint32 | Number of banks in the output accumulator per PE | Positive number | 32 |
| | | | | |
| M | uint32 | Systolic array width (Parallel filters)| Positive number | 32 |
| N | uint32 | Systolic array height (Parallel windows)| Positive number | 16 |
| PMAX | uint32 | Maximum precision allowed per PE | Positive number | 8 |
| PMIN | uint32 | Minimum precision allowed per PE | Positive number | 2 |

### Structure:
*   **sys**: Folder for system libraries
    *   common: contains common definitions for all classes
    *   cxxopts: library to read options from the console (c) Jarryd Beck
    *   Statistic: container for the statistics of the simulation
    *   Batch: support to load batch files
*   **cnpy**: Folder for supporting math libraries
    *   cnpy: library to read Numpy arrays
    *   Array: class to store and operate with flatten arrays
*   **core**: Folder for the main classes of the simulator
    *   Network: class to store the network
    *   Layer: class to store the layer of the network
    *   Inference: class that defines the behaviour of a standard deep learning inference simulation
    *   Stripes: class for the Stripes accelerator
    *   DynamicStripes: class for the Dynamic-Stripes accelerator
    *   Loom: class for the Loom accelerator
    *   BitPragmatic: class for the Bit-Pragmatic accelerator
    *   Laconic: class for the Laconic accelerator
    *   BitTactical: common class for both BitTactical behaviors
    *   BitTacticalP: class for the Bit-Tactical version p accelerator
    *   BitTacticalE: class for the Bit-Tactical version e accelerator
    *   SCNN: class for the SCNN accelerator and common behaviour for SCNN-like architectures
    *   BitFusion: class for the BitFusion accelerator
*   **interface**: Folder to interface with input/output operations
    *   Interface: common class for the reader and writer classes
    *   NetReader: class to read and load a network using different formats
    *   NetWriter: class to write and dump a network using different formats
    *   StatsWriter: class to dump simulation statistics in different formats
*   **proto**: Folder for protobuf definition
    *   network.proto: Google protobuf definition for the network
    *   caffe.proto: Caffe protobuf definition for Caffe networks
    *   batch.proto: Google protobuf definition for the batch file
    *   schedule.proto: Tactical schedule protobuf definition
    
