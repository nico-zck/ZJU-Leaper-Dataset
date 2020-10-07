% Script of examples that reproduces results from metadata structures previously saved in One-D/Scripts.
% The metadata were obtained by running the other test scripts, and are stored as additional material in MCALab.
% This script requires a little extra information to be ran.

clear all

% 1D MCA, no inpainting.
x = GenSignal('EegfMRI');
load figMCAEEGmetadata
options
parts = firefrommetadata(x,[],options);

clear all

% 1D MCA, no inpainting.
x = GenSignal('Bumps-LCosine',1024);
load figMCABumpsLocalSinesmetadata
options
parts = firefrommetadata(x,[],options);

