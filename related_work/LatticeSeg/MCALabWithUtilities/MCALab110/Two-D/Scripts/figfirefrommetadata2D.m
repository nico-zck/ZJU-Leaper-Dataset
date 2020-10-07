% Script of examples that reproduces results from metadata structures previously saved in Two-D/Scripts.
% The metadata were obtained by running the other test scripts, and are stored as additional material in MCALab.
% This script requires a little extra information to be ran.

clear all

% 2D MCA, no inpainting.
load linesgaussians
load figMCAlinesgaussiansmetadata
options
parts = firefrommetadata(img,[],options);

clear all

% 2D MCA inpainting.
load texturegaussians
mask=double(imread('mask_texturegaussians.bmp'));mask=mask(:,:,1);mask(mask~=0)=1;
load figInpainttexturegaussiansMCAmetadata
optionsMCA
parts = firefrommetadata(img.*mask,mask,optionsMCA);

clear all

% 2D ECM inpainting.
load linesgaussians
mask=double(imread('mask_texturegaussians.bmp'));mask=mask(:,:,1);mask(mask~=0)=1;
load figInpaintlinesgaussiansECMmetadata
optionsECM
parts = firefrommetadata(img.*mask,mask,optionsECM);
