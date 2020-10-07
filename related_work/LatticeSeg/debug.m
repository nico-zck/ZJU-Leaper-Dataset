P_NAME = 'star';

DATASET_DIR = 'D:/Dataset/Textile/HKU/';
IMG_SIZE = [256, 256];

%% test on test set images with d_star
test_names = dir([DATASET_DIR, char(P_NAME), '/defective/', '*.bmp']);
test_names = {test_names.name};
img_base_path = [DATASET_DIR, char(P_NAME), '/defective/%s'];

disp('##### start test #####');
mask_pred = zeros([length(test_names), IMG_SIZE]);
dist_matrix = cell(1, length(test_names));
for i = 18:length(test_names)
    name = test_names{i};
    disp(name);
    img = im2double(rgb2gray(imread(sprintf(img_base_path, name))));
%     img = imresize(img, IMG_SIZE);
    [D, binary_result] = LSG_on_img(img, 0);
    dist_matrix{i} = D;
    mask_pred(i, :, :) = squeeze(binary_result);
end
disp('##### end test #####');
