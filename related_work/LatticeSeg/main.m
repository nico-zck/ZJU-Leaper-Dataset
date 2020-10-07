P_ID = 1;

DATASET_DIR = 'D:/Dataset/ZJU/Dataset-Fabric/FabricFinal/';
RESULT_DIR = './result/'; mkdir(RESULT_DIR);
IMG_SIZE = [256, 256];

for P_ID = 1:4
    img_base_path = [DATASET_DIR, 'Images/%s.jpg'];
    json_path = [DATASET_DIR, 'ImageSets/Patterns/pattern', num2str(P_ID), '.json'];
    names = loadjson(json_path);

    %% find d_star on normal images
    train_names = names.normal.train;
    train_names = train_names(1:100:end);
    
    D_all = {};
    for name = train_names
        good_img =  im2double(rgb2gray(imread(sprintf(img_base_path, char(name)))));
        good_img = imresize(good_img, IMG_SIZE);
        D = LSG_on_img(good_img);
        D_all{end + 1} = D;
    end
    
    % figure out d_star from all distance matrices
    d_star = cellfun(@(x) max(x(:)), D_all);
    d_star = mean(d_star(~isnan(d_star)));
    
    %% test on test set images with d_star
    test_names = [names.normal.test, names.defect.test];
    test_names = test_names(1:100:end);
    
    result_all = zeros([length(test_names), IMG_SIZE]);
    for i = 1:length(test_names)
        name = test_names(i);
        img = im2double(rgb2gray(imread(sprintf(img_base_path, char(name)))));
        img = imresize(img, IMG_SIZE);
        binary_result = LSG_on_img(img, d_star);
        result_all(i, :, :) = squeeze(binary_result);
    end
    
    %% save result
    mat_path = [RESULT_DIR, 'pattern', num2str(P_ID) '.mat'];
    save(mat_path, 'dist_matrix', 'test_names', 'mask_pred');

end