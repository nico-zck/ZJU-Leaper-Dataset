%%% This code handle the defect detection problem for star-patterned fabric
%%% by a simple image decomposion rationale.
%%%% Star package:
%%%%    'bn' --BrokenEnd;     'he'--Hole;      'nn'--NettingMultiple
%%%%    'tk' --ThickBar;      'tn'--ThinBar;
%%%% written by Wenxing Zhang, wenxing84@gmail.com 2012
clc;
close all;
clear;

%%%%%%%%%%%%%%%%%%% Test images %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Star package:
types = {'bn', 'he', 'nn', 'tk', 'tn'};
idx = 1;
for t=1:length(types)
    d_type = types{t};   %type of defect
    for i = 1:5  %number of image
        name = ['bad_', d_type, int2str(i), '.png'];
        IM = ['../dataset/star/test/', name];
        %%%%%%%%%% preconditioning: hist(J) or hist(J,2) %%%%%%%%%%%%
        I = im2double(imread(IM)); I = rgb2gray(I);
        x0 = histeq(I,2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %% %%%%%%%%%%%%%%%%%Parameters selection  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        switch d_type
            case 'bn'
                tau=1.5; mu=1; beta1=80; beta2=1;  %BrokenEnd
            otherwise
                tau=1.5; mu=1; beta1=50; beta2=1;  %Hole,NettingMultiple£¬thickbar, thinbar
        end

        %%%%%%%%%%%% solver
        opts.MaxIt = 100; p =inf;
        opts.tau = tau; opts.mu = mu; opts.beta1=beta1; opts.beta2=beta2;
        [u_S,v_S] = ADM(x0,p,opts);

        results(idx,:,:) = u_S;
        names(idx) = string(name);
        idx = idx + 1;

%         figure;
%         subplot(121); imshow(I,[]); title('original')
%         subplot(122); imshow(u_S,[]);title('defected defect');
    end
end

save('ID_star.mat', 'results', 'names');