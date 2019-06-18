### Example files

Example batch files are given in this folder. These are the following:

   * **simulator_example**: Example file for launching simulations:
       * GoogLeNet read from Caffe Prototxt model:
           1. Laconic timing simulation for a 16x16 tile configuration
           2. BitPragmatic potential/work reduction
           3. Laconic potential/work reduction
       * AlexNet read from Caffe Prototxt model (It is internally converted into Quantized Fixed point Protobuf):
           1. BitPragmatic timing simulation for a 16x16 tile configuration and first shift size of 2bits
           2. BitPragmatic timing simulation for a 16x256 tile configuration (default first shift size of 0bits)
   * **SCNN_example**: Example file for launching SCNN simulations:
       * GoogLeNet read from Caffe Prototxt model (It is internally converted into Quantized Fixed point Protobuf)