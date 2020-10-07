P_NAME = 'star';

DATASET_DIR = 'D:/Dataset/Textile/HKU/';
RESULT_DIR = './result/'; mkdir(RESULT_DIR);
IMG_SIZE = [256, 256];

%% only test defective images of test set
test_names = dir([DATASET_DIR, P_NAME, '/defective/', '*.bmp']);
test_names = {test_names.name};
img_base_path = [DATASET_DIR, P_NAME, '/defective/%s'];
mask_base_path = [DATASET_DIR, P_NAME, '/defective/mask/%s'];

disp('##### start test #####');
sparse_matrix = cell(1, length(test_names));
mask_pred = zeros([length(test_names), IMG_SIZE]);
parfor i = 1:length(test_names)
    name = test_names{i};
    disp(name);
    % load image
    img = im2double(rgb2gray(imread(sprintf(img_base_path, name))));
    %     img = imresize(img, IMG_SIZE);
    % load mask as defect prior
    mask = double(imread(sprintf(mask_base_path, name)));
    %     mask = imresize(mask, IMG_SIZE, 'nearest');
    % ADM with ALM to solve the problem
    [A, E, N] = LRD_NP(img, mask, 0.03, 0.2);
    % obtain saliency map
    S = abs(E);
    S_hat = imgaussfilt(S .* S);
    S_hat = (S_hat - min(S_hat(:))) / (max(S_hat(:)) - min(S_hat(:)));
    % obtain binary map
    binary_map = imbinarize(S_hat); % global Otsu
%     binary_map = imbinarize(S_hat, 'adaptive'); % local adaptive
%     binary_map = imbinarize(S_hat, adaThresh(S_hat*255)/255); % adaptive
    % save result
    sparse_matrix{i} = E;
    mask_pred(i, :, :) = squeeze(binary_map);
end
disp('##### end test #####');

%% save result
mat_path = [RESULT_DIR, 'LR_pattern_', P_NAME, '.mat'];
names = test_names;
save(mat_path, 'names', 'mask_pred', 'sparse_matrix');
