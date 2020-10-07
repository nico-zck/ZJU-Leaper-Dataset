clc;close all;clear;

DATASET_DIR = 'D:/Dataset/ZJU/Dataset-Fabric/FabricFinal/';
% DATASET_DIR = 'D:/zck/Dataset/FabricFinal/';
RESULT_DIR = './result/'; mkdir(RESULT_DIR);
IMG_SIZE = [256, 256];

%%%%%%%%%%%%%%%%%%% Test images %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for P_ID=1:4
    disp(['pattern', num2str(P_ID)]);
    
    img_base_path = [DATASET_DIR, 'Images/%s.jpg'];
    json_path = [DATASET_DIR, 'ImageSets/Patterns/pattern', num2str(P_ID), '.json'];
    all_names = loadjson(json_path);

    test_names = [all_names.normal.test, all_names.defect.test];
    % test_names = test_names(1:100:end);
    
    names = {};
    raw_pred = [];
    disp('##### start test #####');
    parfor idx = 1:length(test_names)  %number of image
        name = test_names{idx};        
        disp(name);
        
        img = imread(sprintf(img_base_path, name));
        img = imresize(img, IMG_SIZE);
        I = im2double(img); x0 = rgb2gray(I); 

        tau=1.5; mu=1; beta1=50; beta2=1;
        %%%% solver
        p =inf;
        opts = struct('MaxIt', 100, 'tau', tau, 'mu', mu, 'beta1', beta1, 'beta2', beta2);
        [u_S,v_S] = ADM(x0, p, opts);
%       fprintf([name, '\t the running is over!\n']);

        %%%%%%% display defects
    %     figure;
    %     subplot(121); imshow(I,[]); title('original')
    %     subplot(122); imshow(u_S,[]);title('detected defect'); 

        names{idx} = name;
        raw_pred(idx,:,:) = u_S;
    end

    save([RESULT_DIR, 'ID_pattern_', num2str(P_ID), '.mat'], 'names', 'raw_pred');
end