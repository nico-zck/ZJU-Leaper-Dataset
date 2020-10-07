%%% This code handle the defect detection problem for box-patterned
%%% fabric (most difficult fabric to defect by ID method because the fabric
%%% are not quite like texture. Or the motif is too large)
%%%  by a simple image decomposion rationale. The datasets are tested including
%%%% Box package:
%%%%    'bn' --BrokenEnd;   'tk' --ThickBar;      'tn'--ThinBar;
%%%% written by Wenxing Zhang, wenxing84@gmail.com 2012
clc;
close all;
clear;

%%%%%%%%%%%%%%%%%%% Test images %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% types = {'bn', 'tk', 'tn'};
idx = 1;
types = {'bn', 'he', 'nn', 'tk', 'tn'};
for t=1:length(types)
    d_type = types{t};
    for i = 1:5  %number of image
        name = ['bad_', d_type, int2str(i), '.png'];
        IM = ['../dataset/box/test/', name];
        
        %%%%%%%%%% preconditioning: hist(J) or hist(J,2) %%%%%%%%%%%%
        I = im2double(imread(IM)); I = rgb2gray(I);
        x0 = histeq(I);
        
        %% %%%%%%%%%%%%%%%%%Parameters selection  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        switch d_type
            case {'bn','tk'}
                tau=2; mu=1; beta1=80; beta2=1;   %%  BrokenEnd, thickbar
            case 'tn'
                tau=1.5; mu=1; beta1=10; beta2=0.1;  % thinbar
            otherwise
                tau=0.6; mu=1.5; beta1=50; beta2=1;
        end
        
        %%%%%%%%%%%% solver
        opts.MaxIt = 100; p = inf;
        opts.tau = tau; opts.mu = mu; opts.beta1=beta1; opts.beta2=beta2;
        [u_S,v_S] = ADM(x0,p,opts);
        
        results(idx,:,:) = u_S;
        names(idx) = string(name);
        idx = idx + 1;
        
%         figure(1);
%         subplot(2,3,i); imshow(I,[]); title('original');
%         figure(2);
%         subplot(2,3,i); imshow(u_S,[]);title('defected defect');
    end
end

save('ID_box.mat', 'results', 'names');

