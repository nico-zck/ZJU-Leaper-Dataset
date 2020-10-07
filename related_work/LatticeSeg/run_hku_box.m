P_NAME = 'box';

DATASET_DIR = 'D:/Dataset/Textile/HKU/';
RESULT_DIR = './result/'; mkdir(RESULT_DIR);
IMG_SIZE = [256, 256];

%% find d_star on normal images
train_names = dir([DATASET_DIR, P_NAME, '/normal/', '*.bmp']);
train_names = {train_names.name};
img_base_path = [DATASET_DIR, P_NAME, '/normal/%s'];

disp('##### start training #####');
train_dist_matrix = cell(1, length(train_names));
parfor i = 1:length(train_names)
    name = train_names{i};
    disp(name);
    good_img =  im2double(rgb2gray(imread(sprintf(img_base_path, name))));
    good_img = imresize(good_img, IMG_SIZE);
    D = LSG_on_img(good_img);
    train_dist_matrix{i} = D;
end
disp('##### end training #####');

% figure out d_star from all distance matrices
d_star = cellfun(@(x) max(x(:)), train_dist_matrix);
d_star = mean(d_star(~isnan(d_star)));

%% test on test set images with d_star
test_names = dir([DATASET_DIR, P_NAME, '/defective/', '*.bmp']);
test_names = {test_names.name};
img_base_path = [DATASET_DIR, P_NAME, '/defective/%s'];

disp('##### start test #####');
dist_matrix = cell(1, length(test_names));
mask_pred = zeros([length(test_names), IMG_SIZE]);
parfor i = 1:length(test_names)
    name = test_names{i};
    disp(name);
    img = im2double(rgb2gray(imread(sprintf(img_base_path, name))));
    img = imresize(img, IMG_SIZE);
    [D, binary_result] = LSG_on_img(img, d_star);
    dist_matrix{i} = D;
    mask_pred(i, :, :) = squeeze(binary_result);
end
disp('##### end test #####');

%% save result
mat_path = [RESULT_DIR, 'LS_pattern_', P_NAME, '.mat'];
names = test_names;
save(mat_path, 'names', 'mask_pred', 'dist_matrix', 'train_dist_matrix');
