clc;close all;clear;

data_type = {'box', 'dot', 'star'};

DATASET_DIR = 'D:/Dataset/Textile/HKU/';
RESULT_DIR = './result/'; mkdir(RESULT_DIR);
IMG_SIZE = [256, 256];

%%%%%%%%%%%%%%%%%%% Test images %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for p_id=1:length(data_type)
    P_NAME = data_type{p_id};
    disp(P_NAME);
    
    test_names = dir([DATASET_DIR, P_NAME, '/defective/', '*.bmp']);
    test_names = {test_names.name};
    img_base_path = [DATASET_DIR, P_NAME, '/defective/%s'];

    names = {};
    raw_pred = [];
    disp('##### start test #####');
    parfor idx = 1:length(test_names)  %number of image
        name = test_names{idx};
        [a, b] = regexpi(name, '_[a-z]+_');
        defect_type = name(a+1:b-1);
        
        disp([name, ': ', defect_type]);
        
        img = imread(sprintf(img_base_path, name));
        img = imresize(img, IMG_SIZE);
        I = im2double(img); x0 = rgb2gray(I); 

        switch P_NAME
            %%%%%% Box pattern
            case 'box'
                switch defect_type
                    case {'bn','tk'}
                        tau=2; mu=1; beta1=80; beta2=1;   %%  BrokenEnd, thickbar
                    case 'tn'
                        tau=1.5; mu=1; beta1=10; beta2=0.1;  % thinbar
                    otherwise
                        tau=0.6; mu=1.5; beta1=50; beta2=1;
                end
            %%%%%% Dot pattern
            case 'dot'
                switch defect_type
                    case {'b','h'}
                        tau=0.3; mu=1; beta1=50; beta2=1;  %BrokenEnd,Hole 
                    case 'k'
                        tau=0.5; mu=1; beta1=100; beta2=1; %Knot
                    otherwise 
                        tau=0.2; mu=1; beta1=80; beta2=1;  %other defects
                end
            %%%%%% Star pattern
            case 'star'
                switch defect_type
                    case 'bn'
                        tau=1.5; mu=1; beta1=80; beta2=1;  %BrokenEnd
                    otherwise
                        tau=1.5; mu=1; beta1=50; beta2=1;  %Hole,NettingMultiple，thickbar, thinbar
                end
            otherwise
                tau=1.5; mu=1; beta1=50; beta2=1;
        end

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

    save([RESULT_DIR, 'ID_pattern_', P_NAME, '.mat'], 'names', 'raw_pred');
end